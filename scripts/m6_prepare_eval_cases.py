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


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base-manifest", required=True)
    p.add_argument("--rerank-stage", required=True)
    p.add_argument("--queries", required=True)
    p.add_argument("--query-embeddings", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--evidence-k", type=int, default=4)
    p.add_argument("--candidate-k", type=int, default=20)
    p.add_argument("--base-weight", type=float, default=0.5)
    p.add_argument("--rrf-constant", type=int, default=60)
    p.add_argument("--hit-cases", type=int, default=20)
    p.add_argument("--miss-cases", type=int, default=20)
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
    faiss.omp_set_num_threads(args.threads)
    index = faiss.read_index(str(index_path))
    _, base_ids = index.search(np.ascontiguousarray(q, dtype=np.float32), args.candidate_k)

    hits: list[dict] = []
    misses: list[dict] = []
    for i, query in enumerate(queries):
        ids = np.asarray(base_ids[i], dtype=np.int64)
        fine_scores = np.asarray(rerank[ids] @ np.asarray(q[i], dtype=np.float32), dtype=np.float32)
        ordered, fused = reciprocal_rank_fusion_order(
            ids,
            fine_scores,
            base_weight=args.base_weight,
            rrf_constant=args.rrf_constant,
        )
        chosen = ordered[: args.evidence_k]
        target_id = filename_to_id[query["target_filename"]]
        evidence = []
        target_evidence_id = None
        for rank, faiss_id in enumerate(chosen, start=1):
            filename = image_map[int(faiss_id)]
            evidence_id = f"E{rank}"
            if int(faiss_id) == target_id:
                target_evidence_id = evidence_id
            evidence.append(
                {
                    "evidence_id": evidence_id,
                    "rank": rank,
                    "faiss_id": int(faiss_id),
                    "filename": filename,
                    "path": str(PROJECT_ROOT / "data_storage" / "RICO_dataset" / filename),
                    "retrieval_score": float(fused[rank - 1]),
                }
            )
        case = {
            "case_id": str(query.get("query_id", i)),
            "query": query["query"],
            "target_filename": query["target_filename"],
            "target_evidence_id": target_evidence_id,
            "retrieval_hit": target_evidence_id is not None,
            "evidence": evidence,
        }
        bucket = hits if target_evidence_id is not None else misses
        limit = args.hit_cases if target_evidence_id is not None else args.miss_cases
        if len(bucket) < limit:
            bucket.append(case)
        if len(hits) >= args.hit_cases and len(misses) >= args.miss_cases:
            break

    cases = hits + misses
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for case in cases:
            handle.write(json.dumps(case, ensure_ascii=False) + "\n")
    meta = {
        "milestone": "M6",
        "protocol": "balanced evidence-selection diagnostic",
        "num_cases": len(cases),
        "hit_cases": len(hits),
        "miss_cases": len(misses),
        "evidence_k": args.evidence_k,
        "candidate_k": args.candidate_k,
        "base_weight": args.base_weight,
        "rrf_constant": args.rrf_constant,
        "note": "Selection accuracy is scored only on retrieval-hit cases; misses are used for grounding/abstention diagnostics, not semantic correctness.",
    }
    output.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))
    print(f"cases={output}")


if __name__ == "__main__":
    main()
