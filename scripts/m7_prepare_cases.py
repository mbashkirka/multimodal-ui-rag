from __future__ import annotations

import argparse
import json
from pathlib import Path

import faiss
import numpy as np

from ui_rag.config import PROJECT_ROOT
from ui_rag.evaluation.dataset import read_queries_jsonl
from ui_rag.indexing.manifest import read_json, sha256_file
from ui_rag.reranking.multires import reciprocal_rank_fusion_order


def _load_map(path: Path) -> dict[int, str]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {int(k): str(v) for k, v in raw.items()}


def _read_excluded(paths: list[str]) -> set[str]:
    out: set[str] = set()
    for raw in paths:
        path = Path(raw)
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                item = json.loads(line)
                out.add(str(item.get("case_id", item.get("query_id", ""))))
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base-manifest", required=True)
    p.add_argument("--rerank-stage", required=True)
    p.add_argument("--queries", required=True)
    p.add_argument("--query-embeddings", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--exclude-cases", action="append", default=[])
    p.add_argument("--candidate-k", type=int, default=20)
    p.add_argument("--evidence-k", type=int, default=4)
    p.add_argument("--base-weight", type=float, default=0.5)
    p.add_argument("--rrf-constant", type=int, default=60)
    p.add_argument("--rank1-exact", type=int, default=10)
    p.add_argument("--top4-nonrank1", type=int, default=10)
    p.add_argument("--miss", type=int, default=10)
    p.add_argument("--threads", type=int, default=1)
    args = p.parse_args()

    manifest = read_json(Path(args.base_manifest))
    stage = read_json(Path(args.rerank_stage))
    if manifest["mapping"]["artifact_sha256"] != stage["mapping"]["artifact_sha256"]:
        raise RuntimeError("base/rerank mapping mismatch")
    index_path = PROJECT_ROOT / manifest["index"]["path"]
    map_path = PROJECT_ROOT / manifest["mapping"]["artifact_path"]
    rerank_path = PROJECT_ROOT / stage["embedding"]["path"]
    for path, digest in [
        (index_path, manifest["index"]["sha256"]),
        (map_path, manifest["mapping"]["artifact_sha256"]),
        (rerank_path, stage["embedding"]["sha256"]),
    ]:
        if sha256_file(path) != digest:
            raise RuntimeError(f"artifact hash mismatch: {path}")

    queries = read_queries_jsonl(Path(args.queries))
    q = np.load(args.query_embeddings, mmap_mode="r")
    rerank = np.load(rerank_path, mmap_mode="r")
    image_map = _load_map(map_path)
    filename_to_id = {name: idx for idx, name in image_map.items()}
    excluded = _read_excluded(args.exclude_cases)

    faiss.omp_set_num_threads(args.threads)
    index = faiss.read_index(str(index_path))
    _, base_ids = index.search(np.ascontiguousarray(q, dtype=np.float32), args.candidate_k)

    buckets: dict[str, list[dict]] = {"rank1_exact": [], "top4_nonrank1": [], "miss": []}
    limits = {
        "rank1_exact": args.rank1_exact,
        "top4_nonrank1": args.top4_nonrank1,
        "miss": args.miss,
    }

    for i, query in enumerate(queries):
        case_id = str(query.get("query_id", i))
        if case_id in excluded:
            continue
        ids = np.asarray(base_ids[i], dtype=np.int64)
        fine_scores = np.asarray(rerank[ids] @ np.asarray(q[i], dtype=np.float32), dtype=np.float32)
        ordered, fused = reciprocal_rank_fusion_order(ids, fine_scores, base_weight=args.base_weight, rrf_constant=args.rrf_constant)
        chosen = ordered[: args.evidence_k]
        target_id = filename_to_id[query["target_filename"]]
        evidence = []
        target_evidence_id = None
        for rank, faiss_id in enumerate(chosen, start=1):
            filename = image_map[int(faiss_id)]
            eid = f"E{rank}"
            if int(faiss_id) == target_id:
                target_evidence_id = eid
            evidence.append({
                "evidence_id": eid,
                "rank": rank,
                "faiss_id": int(faiss_id),
                "filename": filename,
                "path": str(PROJECT_ROOT / "data_storage" / "RICO_dataset" / filename),
                "retrieval_score": float(fused[rank - 1]),
            })
        if target_evidence_id == "E1":
            slice_name = "rank1_exact"
        elif target_evidence_id is not None:
            slice_name = "top4_nonrank1"
        else:
            slice_name = "miss"
        if len(buckets[slice_name]) >= limits[slice_name]:
            continue
        buckets[slice_name].append({
            "case_id": case_id,
            "query": query["query"],
            "target_filename": query["target_filename"],
            "target_evidence_id": target_evidence_id,
            "retrieval_slice": slice_name,
            "evidence": evidence,
        })
        if all(len(buckets[k]) >= limits[k] for k in limits):
            break

    if any(len(buckets[k]) < limits[k] for k in limits):
        raise RuntimeError(f"could not fill requested M7 slices: { {k: len(v) for k,v in buckets.items()} }")

    cases = buckets["rank1_exact"] + buckets["top4_nonrank1"] + buckets["miss"]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in cases), encoding="utf-8")
    meta = {
        "milestone": "M7",
        "protocol": "generation holdout stratified by frozen retrieval outcome",
        "num_cases": len(cases),
        "slices": {k: len(v) for k, v in buckets.items()},
        "excluded_case_ids": len(excluded),
        "evidence_policy": "deterministic rank1",
        "note": "M7 cases are drawn from Screen2Words dev but exclude all persisted M6 prompt-design/evaluation cases supplied via --exclude-cases.",
    }
    output.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))
    print(f"cases={output}")


if __name__ == "__main__":
    main()
