from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np

from ui_rag.config import PROJECT_ROOT, RICO_DIR
from ui_rag.indexing.manifest import read_json, sha256_file


def search_faiss_from_file(
    *,
    manifest_path: Path,
    query_vector_path: Path,
    top_k: int = 5,
    verify_hashes: bool = True,
) -> list[dict]:
    """Search FAISS in a faiss/numpy-only child process."""
    if top_k <= 0:
        raise ValueError("top_k must be positive")

    manifest = read_json(manifest_path)
    index_path = PROJECT_ROOT / manifest["index"]["path"]
    mapping_path = PROJECT_ROOT / manifest["mapping"]["artifact_path"]
    if verify_hashes:
        if sha256_file(index_path) != manifest["index"]["sha256"]:
            raise RuntimeError("FAISS index hash does not match manifest")
        if sha256_file(mapping_path) != manifest["mapping"]["artifact_sha256"]:
            raise RuntimeError("image mapping hash does not match manifest")

    index = faiss.read_index(str(index_path))
    with mapping_path.open("r", encoding="utf-8") as handle:
        image_map = json.load(handle)

    if index.d != int(manifest["embedding"]["dimension"]):
        raise RuntimeError("index dimension does not match manifest")
    if index.ntotal != len(image_map):
        raise RuntimeError("index/map count mismatch")

    query_vector = np.load(query_vector_path)
    query_vector = np.ascontiguousarray(query_vector, dtype=np.float32)
    if query_vector.shape != (1, int(index.d)):
        raise RuntimeError(f"invalid query vector shape: {query_vector.shape}")

    top_k = min(top_k, int(index.ntotal))
    scores, indices = index.search(query_vector, top_k)
    results: list[dict] = []
    for rank, (score, faiss_id) in enumerate(zip(scores[0], indices[0]), start=1):
        if int(faiss_id) < 0:
            continue
        filename = image_map[str(int(faiss_id))]
        results.append(
            {
                "rank": rank,
                "faiss_id": int(faiss_id),
                "filename": filename,
                "path": str(RICO_DIR / filename),
                "score": float(score),
            }
        )
    return results
