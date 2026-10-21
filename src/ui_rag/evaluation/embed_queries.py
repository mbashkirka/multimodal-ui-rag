from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from ui_rag.embeddings.factory import create_encoder, encoder_kwargs_from_manifest
from ui_rag.evaluation.dataset import read_queries_jsonl
from ui_rag.evaluation.metrics import percentile
from ui_rag.indexing.manifest import read_json, sha256_file, write_json


def embed_benchmark_queries(
    *,
    manifest_path: Path,
    queries_path: Path,
    output_path: Path,
    stage_path: Path,
    batch_size: int = 128,
    device: str | None = None,
) -> dict:
    """Encode benchmark text in the encoder-only child process."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    manifest = read_json(manifest_path)
    queries = read_queries_jsonl(queries_path)
    cfg = manifest["embedding"]
    encoder = create_encoder(**encoder_kwargs_from_manifest(cfg), device=device)

    num_queries = len(queries)
    dim = int(cfg["dimension"])
    embeddings = np.empty((num_queries, dim), dtype=np.float32)
    batch_latencies_ms: list[float] = []

    started = time.perf_counter()
    for start in range(0, num_queries, batch_size):
        end = min(start + batch_size, num_queries)
        texts = [item["query"] for item in queries[start:end]]
        batch_started = time.perf_counter()
        encoded = encoder.encode_text(texts)
        batch_latencies_ms.append((time.perf_counter() - batch_started) * 1000.0)
        if encoded.shape != (end - start, dim):
            raise RuntimeError(f"unexpected query embedding shape: {encoded.shape}")
        embeddings[start:end] = encoded

    total_seconds = time.perf_counter() - started
    if not np.isfinite(embeddings).all():
        raise RuntimeError("query embeddings contain non-finite values")
    if not np.allclose(np.linalg.norm(embeddings, axis=1), 1.0, atol=2e-4):
        raise RuntimeError("query embeddings are not L2 normalized")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, np.ascontiguousarray(embeddings, dtype=np.float32))
    stage = {
        "stage": "benchmark_query_embeddings",
        "manifest": str(manifest_path),
        "queries": str(queries_path),
        "queries_sha256": sha256_file(queries_path),
        "output": str(output_path),
        "output_sha256": sha256_file(output_path),
        "num_queries": num_queries,
        "dimension": dim,
        "dtype": "float32",
        "l2_normalized": True,
        "device": encoder.device,
        "encoder": encoder.metadata(),
        "batch_size": batch_size,
        "timing": {
            "total_seconds": total_seconds,
            "queries_per_second": num_queries / total_seconds if total_seconds else None,
            "amortized_ms_per_query": (total_seconds * 1000.0 / num_queries) if num_queries else None,
            "batch_latency_ms_p50": percentile(batch_latencies_ms, 0.50),
            "batch_latency_ms_p95": percentile(batch_latencies_ms, 0.95),
            "num_batches": len(batch_latencies_ms),
        },
    }
    write_json(stage_path, stage)
    return stage
