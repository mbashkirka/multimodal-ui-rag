from __future__ import annotations

import argparse
import time
from pathlib import Path

import faiss
import numpy as np

from ui_rag.config import PROJECT_ROOT
from ui_rag.indexing.manifest import read_json, sha256_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--m", type=int, default=32)
    parser.add_argument("--ef-construction", type=int, default=100)
    parser.add_argument("--ef-search", type=int, default=64)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    manifest = read_json(manifest_path)
    embedding_path = PROJECT_ROOT / manifest["embedding"]["path"]
    if sha256_file(embedding_path) != manifest["embedding"]["artifact_sha256"]:
        raise RuntimeError("embedding hash mismatch")

    vectors = np.load(embedding_path, mmap_mode="r")
    limit = min(args.limit, int(vectors.shape[0]))
    if limit < 2:
        raise ValueError("smoke limit must be >= 2")
    subset = np.ascontiguousarray(vectors[:limit], dtype=np.float32)

    faiss.omp_set_num_threads(args.threads)
    index = faiss.IndexHNSWFlat(subset.shape[1], args.m, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = args.ef_construction
    started = time.perf_counter()
    index.add(subset)
    build_seconds = time.perf_counter() - started
    index.hnsw.efSearch = args.ef_search

    scores, ids = index.search(np.ascontiguousarray(subset[0:1]), 5)
    if int(ids[0, 0]) != 0:
        raise RuntimeError(f"HNSW self-retrieval failed: expected id 0, got {ids[0, 0]}")
    if float(scores[0, 0]) < 0.999:
        raise RuntimeError(f"HNSW self-retrieval score too low: {scores[0, 0]}")

    print("M4 HNSW smoke PASSED")
    print(f"vectors={limit}")
    print(f"dimension={subset.shape[1]}")
    print(f"M={args.m}")
    print(f"efConstruction={args.ef_construction}")
    print(f"efSearch={args.ef_search}")
    print(f"threads={args.threads}")
    print(f"build_seconds={build_seconds:.3f}")
    print(f"self_rank1_id={int(ids[0, 0])}")
    print(f"self_rank1_score={float(scores[0, 0]):.6f}")


if __name__ == "__main__":
    main()
