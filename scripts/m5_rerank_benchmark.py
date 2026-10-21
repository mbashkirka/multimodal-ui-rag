from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import faiss
import numpy as np

from ui_rag.config import PROJECT_ROOT
from ui_rag.evaluation.dataset import read_queries_jsonl
from ui_rag.evaluation.metrics import summarize_ranks
from ui_rag.indexing.manifest import read_json, sha256_file, write_json
from ui_rag.reranking.multires import RerankConfig, reciprocal_rank_fusion_order, target_rank


def _load_map(path: Path) -> dict[int, str]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {int(k): v for k, v in raw.items()}


def _validate_contract(
    exact_manifest: dict,
    rerank_stage: dict,
    exact_manifest_path: Path,
    rerank_stage_path: Path,
) -> tuple[Path, Path, Path]:
    index_path = PROJECT_ROOT / exact_manifest["index"]["path"]
    exact_map_path = PROJECT_ROOT / exact_manifest["mapping"]["artifact_path"]
    rerank_emb_path = PROJECT_ROOT / rerank_stage["embedding"]["path"]
    rerank_map_path = PROJECT_ROOT / rerank_stage["mapping"]["artifact_path"]

    for path in (exact_manifest_path, rerank_stage_path, index_path, exact_map_path, rerank_emb_path, rerank_map_path):
        if not path.exists():
            raise FileNotFoundError(path)

    if sha256_file(index_path) != exact_manifest["index"]["sha256"]:
        raise RuntimeError("exact index hash mismatch")
    if sha256_file(exact_map_path) != exact_manifest["mapping"]["artifact_sha256"]:
        raise RuntimeError("exact mapping hash mismatch")
    if sha256_file(rerank_emb_path) != rerank_stage["embedding"]["sha256"]:
        raise RuntimeError("reranker embedding hash mismatch")
    if sha256_file(rerank_map_path) != rerank_stage["mapping"]["artifact_sha256"]:
        raise RuntimeError("reranker mapping hash mismatch")

    if exact_manifest["mapping"]["artifact_sha256"] != rerank_stage["mapping"]["artifact_sha256"]:
        raise RuntimeError("p256 and reranker mappings differ; FAISS ids are not aligned")
    if exact_manifest["embedding"]["model_name"] != rerank_stage["embedding"]["model_name"]:
        raise RuntimeError("M5 multi-resolution reranker must use the same checkpoint as stage 1")
    if exact_manifest["embedding"].get("backend") != rerank_stage["embedding"].get("backend"):
        raise RuntimeError("stage-1/reranker encoder backend mismatch")
    if exact_manifest["embedding"]["dimension"] != rerank_stage["embedding"]["dimension"]:
        raise RuntimeError("stage-1/reranker embedding dimension mismatch")
    if exact_manifest["embedding"].get("max_num_patches") == rerank_stage["embedding"].get("max_num_patches"):
        raise RuntimeError("reranker must use a distinct image resolution/patch budget")
    return index_path, exact_map_path, rerank_emb_path


def _best_config(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("no reranking configurations")
    return max(
        rows,
        key=lambda row: (
            row["metrics"]["recall_at_10"],
            row["metrics"]["mrr_at_10"],
            row["metrics"]["ndcg_at_10"],
            -row["candidate_k"],
            -row["base_weight"],
        ),
    )


def evaluate(args: argparse.Namespace) -> dict:
    exact_manifest_path = Path(args.exact_manifest)
    rerank_stage_path = Path(args.rerank_stage)
    queries_path = Path(args.queries)
    query_embeddings_path = Path(args.query_embeddings)
    query_stage_path = Path(args.query_stage)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    exact_manifest = read_json(exact_manifest_path)
    rerank_stage = read_json(rerank_stage_path)
    query_stage = read_json(query_stage_path)
    index_path, map_path, rerank_emb_path = _validate_contract(
        exact_manifest, rerank_stage, exact_manifest_path, rerank_stage_path
    )

    if sha256_file(queries_path) != query_stage["queries_sha256"]:
        raise RuntimeError("query file changed after embedding")
    if sha256_file(query_embeddings_path) != query_stage["output_sha256"]:
        raise RuntimeError("query embedding hash mismatch")

    queries = read_queries_jsonl(queries_path)
    q = np.load(query_embeddings_path, mmap_mode="r")
    rerank_emb = np.load(rerank_emb_path, mmap_mode="r")
    image_map = _load_map(map_path)
    filename_to_id = {name: idx for idx, name in image_map.items()}

    dim = int(exact_manifest["embedding"]["dimension"])
    n = int(exact_manifest["index"]["num_vectors"])
    if q.shape != (len(queries), dim):
        raise RuntimeError(f"query shape mismatch: {q.shape}")
    if rerank_emb.shape != (n, dim):
        raise RuntimeError(f"reranker corpus shape mismatch: {rerank_emb.shape}")

    target_ids = np.asarray([filename_to_id[x["target_filename"]] for x in queries], dtype=np.int64)
    index = faiss.read_index(str(index_path))
    faiss.omp_set_num_threads(args.threads)

    if args.selection:
        selection = read_json(Path(args.selection))
        configs = [
            RerankConfig(
                candidate_k=int(selection["selected"]["candidate_k"]),
                base_weight=float(selection["selected"]["base_weight"]),
                rrf_constant=int(selection["selected"].get("rrf_constant", args.rrf_constant)),
            )
        ]
        mode = "frozen_test"
    else:
        configs = [
            RerankConfig(k, weight, args.rrf_constant)
            for k in args.candidate_k
            for weight in args.base_weight
        ]
        mode = "dev_selection"
    for cfg in configs:
        cfg.validate()

    max_k = max(cfg.candidate_k for cfg in configs)
    started = time.perf_counter()
    _base_scores, base_ids = index.search(np.ascontiguousarray(q, dtype=np.float32), max_k)
    stage1_seconds = time.perf_counter() - started

    baseline_ranks = [target_rank(base_ids[i], int(target_ids[i])) for i in range(len(queries))]
    baseline_metrics = summarize_ranks(baseline_ranks, max_k)

    rows: list[dict] = []
    for cfg in configs:
        ranks: list[int | None] = []
        candidate_hits = 0
        rerank_started = time.perf_counter()
        for i in range(len(queries)):
            ids = np.asarray(base_ids[i, : cfg.candidate_k], dtype=np.int64)
            if int(target_ids[i]) in ids:
                candidate_hits += 1
            fine_scores = np.asarray(rerank_emb[ids] @ np.asarray(q[i], dtype=np.float32), dtype=np.float32)
            ordered_ids, _ = reciprocal_rank_fusion_order(
                ids,
                fine_scores,
                base_weight=cfg.base_weight,
                rrf_constant=cfg.rrf_constant,
            )
            ranks.append(target_rank(ordered_ids, int(target_ids[i])))
        rerank_seconds = time.perf_counter() - rerank_started
        metrics = summarize_ranks(ranks, cfg.candidate_k)
        rows.append(
            {
                "candidate_k": cfg.candidate_k,
                "base_weight": cfg.base_weight,
                "fine_weight": 1.0 - cfg.base_weight,
                "rrf_constant": cfg.rrf_constant,
                "candidate_oracle_recall": candidate_hits / len(queries) if queries else 0.0,
                "metrics": metrics,
                "timing": {
                    "rerank_total_seconds": rerank_seconds,
                    "rerank_amortized_ms_per_query": rerank_seconds * 1000.0 / len(queries) if queries else None,
                },
            }
        )

    selected = rows[0] if args.selection else _best_config(rows)
    payload = {
        "milestone": "M5",
        "track": "multi_resolution_reranking",
        "mode": mode,
        "protocol": {
            "stage1": "SigLIP2 Base NaFlex p256 + exact FlatIP",
            "stage2": f"same checkpoint at max_num_patches={rerank_stage['embedding'].get('max_num_patches')}",
            "fusion": "weighted reciprocal-rank fusion; raw cosine scores are not mixed",
            "selection_split": "dev only" if mode == "dev_selection" else "frozen from dev selection",
        },
        "num_queries": len(queries),
        "baseline": {
            "candidate_k": max_k,
            "metrics": baseline_metrics,
            "stage1_search_seconds": stage1_seconds,
        },
        "grid": rows,
        "selected": {
            "candidate_k": selected["candidate_k"],
            "base_weight": selected["base_weight"],
            "fine_weight": selected["fine_weight"],
            "rrf_constant": selected["rrf_constant"],
            "candidate_oracle_recall": selected["candidate_oracle_recall"],
            "metrics": selected["metrics"],
            "timing": selected["timing"],
        },
        "artifacts": {
            "exact_manifest": str(exact_manifest_path),
            "rerank_stage": str(rerank_stage_path),
            "queries": str(queries_path),
            "query_embeddings": str(query_embeddings_path),
        },
    }
    write_json(output_dir / "comparison.json", payload)
    with (output_dir / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["candidate_k", "base_weight", "oracle_recall", "R@1", "R@5", "R@10", "MRR@10", "nDCG@10", "rerank_ms"])
        for row in rows:
            m = row["metrics"]
            writer.writerow([
                row["candidate_k"], row["base_weight"], row["candidate_oracle_recall"],
                m["recall_at_1"], m["recall_at_5"], m["recall_at_10"], m["mrr_at_10"], m["ndcg_at_10"],
                row["timing"]["rerank_amortized_ms_per_query"],
            ])
    if not args.selection:
        write_json(
            output_dir / "selection.json",
            {
                "selected_on": "Screen2Words dev",
                "primary_metric": "recall_at_10",
                "tie_breakers": ["mrr_at_10", "ndcg_at_10", "smaller_candidate_k"],
                "selected": payload["selected"],
            },
        )

    print("M5 multi-resolution reranking")
    print("candidate_k base_w oracleR  R@1    R@5    R@10   MRR@10 nDCG@10 rerank_ms")
    for row in rows:
        m = row["metrics"]
        print(
            f"{row['candidate_k']:>11} {row['base_weight']:>6.2f} "
            f"{row['candidate_oracle_recall']*100:>6.2f}% {m['recall_at_1']*100:>6.2f}% "
            f"{m['recall_at_5']*100:>6.2f}% {m['recall_at_10']*100:>6.2f}% "
            f"{m['mrr_at_10']:>7.4f} {m['ndcg_at_10']:>8.4f} "
            f"{row['timing']['rerank_amortized_ms_per_query']:>9.4f}"
        )
    print(
        "selected="
        f"k{payload['selected']['candidate_k']}/base_w={payload['selected']['base_weight']:.2f} "
        f"R@10={payload['selected']['metrics']['recall_at_10']*100:.2f}%"
    )
    print(f"json={output_dir / 'comparison.json'}")
    return payload


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--exact-manifest", required=True)
    p.add_argument("--rerank-stage", required=True)
    p.add_argument("--queries", required=True)
    p.add_argument("--query-embeddings", required=True)
    p.add_argument("--query-stage", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--candidate-k", nargs="+", type=int, default=[20, 50, 100])
    p.add_argument("--base-weight", nargs="+", type=float, default=[0.0, 0.25, 0.5, 0.75])
    p.add_argument("--rrf-constant", type=int, default=60)
    p.add_argument("--selection", default=None)
    p.add_argument("--threads", type=int, default=1)
    return p


if __name__ == "__main__":
    evaluate(build_parser().parse_args())
