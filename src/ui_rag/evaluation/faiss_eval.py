from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import faiss
import numpy as np

from ui_rag.config import PROJECT_ROOT
from ui_rag.evaluation.dataset import read_queries_jsonl
from ui_rag.evaluation.metrics import percentile, summarize_ranks
from ui_rag.indexing.manifest import read_json, sha256_file, write_json


def _load_image_map(path: Path) -> dict[int, str]:
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    return {int(key): value for key, value in raw.items()}


def evaluate_faiss_benchmark(
    *,
    manifest_path: Path,
    queries_path: Path,
    query_embeddings_path: Path,
    query_stage_path: Path,
    output_dir: Path,
    search_k: int = 100,
    batch_size: int = 256,
    verify_hashes: bool = True,
) -> dict:
    """Evaluate benchmark in a FAISS/NumPy-only process."""
    if search_k <= 0 or batch_size <= 0:
        raise ValueError("search_k and batch_size must be positive")

    manifest = read_json(manifest_path)
    query_stage = read_json(query_stage_path)
    queries = read_queries_jsonl(queries_path)

    index_path = PROJECT_ROOT / manifest["index"]["path"]
    mapping_path = PROJECT_ROOT / manifest["mapping"]["artifact_path"]
    if verify_hashes:
        if sha256_file(index_path) != manifest["index"]["sha256"]:
            raise RuntimeError("FAISS index hash does not match manifest")
        if sha256_file(mapping_path) != manifest["mapping"]["artifact_sha256"]:
            raise RuntimeError("image mapping hash does not match manifest")
        if sha256_file(queries_path) != query_stage["queries_sha256"]:
            raise RuntimeError("benchmark query file changed after embedding")
        if sha256_file(query_embeddings_path) != query_stage["output_sha256"]:
            raise RuntimeError("benchmark query embeddings hash mismatch")

    index = faiss.read_index(str(index_path))
    image_map = _load_image_map(mapping_path)
    filename_to_faiss_id = {filename: faiss_id for faiss_id, filename in image_map.items()}
    embeddings = np.load(query_embeddings_path, mmap_mode="r")

    expected_dim = int(manifest["embedding"]["dimension"])
    if embeddings.shape != (len(queries), expected_dim):
        raise RuntimeError(
            f"query embedding shape mismatch: {embeddings.shape} != {(len(queries), expected_dim)}"
        )
    if index.ntotal != len(image_map):
        raise RuntimeError("FAISS index/map count mismatch")
    if index.d != expected_dim:
        raise RuntimeError("FAISS dimension mismatch")

    missing_targets = [q["target_filename"] for q in queries if q["target_filename"] not in filename_to_faiss_id]
    if missing_targets:
        raise RuntimeError(f"benchmark contains {len(missing_targets)} targets absent from index mapping")

    search_k = min(search_k, int(index.ntotal))
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "predictions.jsonl"
    failures_path = output_dir / "failures.csv"
    metrics_path = output_dir / "metrics.json"

    ranks: list[int | None] = []
    batch_latencies_ms: list[float] = []
    prediction_records: list[dict] = []

    started = time.perf_counter()
    for start in range(0, len(queries), batch_size):
        end = min(start + batch_size, len(queries))
        batch = np.ascontiguousarray(embeddings[start:end], dtype=np.float32)
        batch_started = time.perf_counter()
        scores, indices = index.search(batch, search_k)
        batch_latencies_ms.append((time.perf_counter() - batch_started) * 1000.0)

        for local_idx, query in enumerate(queries[start:end]):
            target_id = filename_to_faiss_id[query["target_filename"]]
            row_ids = indices[local_idx]
            positions = np.flatnonzero(row_ids == target_id)
            rank = int(positions[0]) + 1 if len(positions) else None
            ranks.append(rank)

            top10: list[dict] = []
            for rank_idx in range(min(10, search_k)):
                faiss_id = int(row_ids[rank_idx])
                if faiss_id < 0:
                    continue
                top10.append(
                    {
                        "rank": rank_idx + 1,
                        "faiss_id": faiss_id,
                        "filename": image_map[faiss_id],
                        "score": float(scores[local_idx][rank_idx]),
                    }
                )
            prediction_records.append(
                {
                    **query,
                    "target_faiss_id": target_id,
                    "target_rank_within_search_k": rank,
                    "top10": top10,
                }
            )

    total_search_seconds = time.perf_counter() - started
    with predictions_path.open("w", encoding="utf-8") as handle:
        for record in prediction_records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    failed = [record for record in prediction_records if record["target_rank_within_search_k"] is None or record["target_rank_within_search_k"] > 10]
    failed.sort(
        key=lambda record: (
            record["target_rank_within_search_k"] is None,
            record["target_rank_within_search_k"] or 10**9,
        ),
        reverse=True,
    )
    with failures_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["query_id", "screen_id", "query", "target_filename", "target_rank_within_search_k", "top1_filename", "top1_score"])
        for record in failed[:500]:
            top1 = record["top10"][0] if record["top10"] else {}
            writer.writerow(
                [
                    record["query_id"],
                    record["screen_id"],
                    record["query"],
                    record["target_filename"],
                    record["target_rank_within_search_k"] or "MISS",
                    top1.get("filename", ""),
                    top1.get("score", ""),
                ]
            )

    rank_metrics = summarize_ranks(ranks, search_k)
    num_queries = len(queries)
    metrics = {
        "benchmark": {
            "name": "Screen2Words text-to-RICO instance retrieval",
            "num_queries": num_queries,
            "unique_target_screens": len({q["screen_id"] for q in queries}),
            "one_relevant_screen_per_query": True,
            "search_k": search_k,
        },
        "retriever": {
            "backend": manifest["embedding"].get("backend", "legacy"),
            "model_name": manifest["embedding"]["model_name"],
            "pretrained": manifest["embedding"].get("pretrained"),
            "max_num_patches": manifest["embedding"].get("max_num_patches"),
            "index_type": manifest["index"]["type"],
            "corpus_vectors": manifest["index"]["num_vectors"],
            "dimension": manifest["embedding"]["dimension"],
        },
        "metrics": rank_metrics,
        "timing": {
            "faiss_search_total_seconds": total_search_seconds,
            "queries_per_second": num_queries / total_search_seconds if total_search_seconds else None,
            "amortized_search_ms_per_query": (total_search_seconds * 1000.0 / num_queries) if num_queries else None,
            "search_batch_size": batch_size,
            "batch_latency_ms_p50": percentile(batch_latencies_ms, 0.50),
            "batch_latency_ms_p95": percentile(batch_latencies_ms, 0.95),
            "query_embedding": query_stage.get("timing", {}),
        },
        "artifacts": {
            "queries": str(queries_path),
            "query_embeddings": str(query_embeddings_path),
            "predictions": str(predictions_path),
            "failures": str(failures_path),
        },
        "interpretation_note": (
            "Ground truth is exact RICO screen identity. Semantically interchangeable screens are "
            "still counted as negatives, so this is an instance-retrieval benchmark and may "
            "underestimate subjective semantic relevance."
        ),
    }
    write_json(metrics_path, metrics)
    return metrics
