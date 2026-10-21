from __future__ import annotations

import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import faiss
import numpy as np

from ui_rag.config import INDEX_DIR, METADATA_DIR, PROJECT_ROOT
from ui_rag.indexing.manifest import read_json, sha256_file, write_json


def build_hnsw_from_manifest(
    *,
    source_manifest_path: Path,
    artifact_name: str,
    m: int,
    ef_construction: int,
    add_batch_size: int = 4096,
    threads: int = 0,
) -> dict:
    """Build an IndexHNSWFlat from an already-persisted embedding artifact.

    This is intentionally a FAISS/NumPy-only stage. M4 reuses the frozen M3 NaFlex
    embeddings and never re-encodes the RICO corpus while changing ANN parameters.
    """
    if m <= 0:
        raise ValueError("m must be positive")
    if ef_construction <= 0:
        raise ValueError("ef_construction must be positive")
    if add_batch_size <= 0:
        raise ValueError("add_batch_size must be positive")
    if threads < 0:
        raise ValueError("threads must be >= 0")

    source_manifest_path = source_manifest_path.resolve()
    source = read_json(source_manifest_path)
    if source["index"]["type"] != "IndexFlatIP":
        raise ValueError("M4 source manifest must be the frozen exact IndexFlatIP baseline")
    if source["index"]["metric"] != "inner_product_on_l2_normalized_vectors":
        raise ValueError("M4 requires normalized inner-product/cosine embeddings")

    embedding_path = PROJECT_ROOT / source["embedding"]["path"]
    mapping_path = PROJECT_ROOT / source["mapping"]["artifact_path"]
    if sha256_file(embedding_path) != source["embedding"]["artifact_sha256"]:
        raise RuntimeError("source embedding hash does not match the frozen M3 manifest")
    if sha256_file(mapping_path) != source["mapping"]["artifact_sha256"]:
        raise RuntimeError("source mapping hash does not match the frozen M3 manifest")

    vectors = np.load(embedding_path, mmap_mode="r")
    if vectors.ndim != 2 or vectors.dtype != np.float32:
        raise RuntimeError(f"invalid embedding artifact: shape={vectors.shape} dtype={vectors.dtype}")
    num_vectors, dimension = map(int, vectors.shape)
    if num_vectors != int(source["embedding"]["num_vectors"]):
        raise RuntimeError("embedding count does not match source manifest")
    if dimension != int(source["embedding"]["dimension"]):
        raise RuntimeError("embedding dimension does not match source manifest")

    if threads:
        faiss.omp_set_num_threads(threads)

    index = faiss.IndexHNSWFlat(dimension, m, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = int(ef_construction)

    started = time.perf_counter()
    for start in range(0, num_vectors, add_batch_size):
        batch = np.ascontiguousarray(vectors[start : start + add_batch_size], dtype=np.float32)
        index.add(batch)
    build_seconds = time.perf_counter() - started

    if int(index.ntotal) != num_vectors:
        raise RuntimeError("HNSW index/vector count mismatch after build")

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    index_path = INDEX_DIR / f"{artifact_name}.faiss"
    manifest_path = METADATA_DIR / f"{artifact_name}-manifest.json"
    faiss.write_index(index, str(index_path))
    index_size_mib = index_path.stat().st_size / (1024.0 * 1024.0)

    manifest = {
        "artifact_name": artifact_name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "milestone": "M4",
        "architecture": {
            "macos_native_isolation": True,
            "embedding_stage_process": "reused frozen M3 artifact; no torch loaded",
            "faiss_stage_process": "faiss+numpy only",
        },
        "source_exact_manifest": {
            "path": str(source_manifest_path.relative_to(PROJECT_ROOT)),
            "sha256": sha256_file(source_manifest_path),
            "artifact_name": source["artifact_name"],
            "index_sha256": source["index"]["sha256"],
        },
        "embedding": source["embedding"],
        "index": {
            "type": "IndexHNSWFlat",
            "metric": "inner_product_on_l2_normalized_vectors",
            "num_vectors": int(index.ntotal),
            "dimension": dimension,
            "m": int(m),
            "ef_construction": int(ef_construction),
            "ef_search_runtime_parameter": True,
            "path": str(index_path.relative_to(PROJECT_ROOT)),
            "sha256": sha256_file(index_path),
            "size_mib": index_size_mib,
            "build_seconds": build_seconds,
            "build_threads": int(threads),
        },
        "mapping": source["mapping"],
        "dataset": source["dataset"],
        "versions": {
            "python": platform.python_version(),
            "faiss": getattr(faiss, "__version__", "unknown"),
            "numpy": np.__version__,
        },
    }
    write_json(manifest_path, manifest)
    return manifest
