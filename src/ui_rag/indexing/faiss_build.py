from __future__ import annotations

import platform
import shutil
from datetime import datetime, timezone

import faiss
import numpy as np

from ui_rag.config import INDEX_DIR, METADATA_DIR, PROJECT_ROOT
from ui_rag.indexing.manifest import read_json, sha256_file, write_json


def build_faiss_index(
    *,
    artifact_name: str,
    add_batch_size: int = 4096,
) -> dict:
    """Build IndexFlatIP from a persisted embedding artifact.

    This process intentionally imports FAISS but not torch/selected-encoder. Keeping the
    native stacks in separate interpreters avoids the macOS ARM OpenMP conflict
    between current PyTorch and faiss-cpu wheels.
    """
    if add_batch_size <= 0:
        raise ValueError("add_batch_size must be positive")

    stage_path = METADATA_DIR / f"{artifact_name}-embedding-stage.json"
    stage = read_json(stage_path)
    embedding_path = PROJECT_ROOT / stage["embedding"]["path"]
    mapping_path = PROJECT_ROOT / stage["mapping"]["artifact_path"]

    if sha256_file(embedding_path) != stage["embedding"]["sha256"]:
        raise RuntimeError("embedding artifact hash does not match embedding-stage metadata")
    if sha256_file(mapping_path) != stage["mapping"]["artifact_sha256"]:
        raise RuntimeError("image mapping hash does not match embedding-stage metadata")

    vectors = np.load(embedding_path, mmap_mode="r")
    if vectors.ndim != 2 or vectors.dtype != np.float32:
        raise RuntimeError(
            f"invalid embedding artifact: shape={vectors.shape} dtype={vectors.dtype}"
        )

    num_vectors, dimension = map(int, vectors.shape)
    if num_vectors != int(stage["embedding"]["num_vectors"]):
        raise RuntimeError("embedding count does not match stage metadata")
    if dimension != int(stage["embedding"]["dimension"]):
        raise RuntimeError("embedding dimension does not match stage metadata")

    index = faiss.IndexFlatIP(dimension)
    for start in range(0, num_vectors, add_batch_size):
        batch = np.ascontiguousarray(vectors[start : start + add_batch_size], dtype=np.float32)
        index.add(batch)

    if int(index.ntotal) != num_vectors:
        raise RuntimeError("FAISS index/vector count mismatch after build")

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    index_path = INDEX_DIR / f"{artifact_name}-flatip.faiss"
    manifest_path = METADATA_DIR / f"{artifact_name}-manifest.json"
    faiss.write_index(index, str(index_path))

    manifest = {
        "artifact_name": artifact_name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "architecture": {
            "macos_native_isolation": True,
            "embedding_stage_process": "torch+selected_encoder only",
            "faiss_stage_process": "faiss+numpy only",
            "resumable_embedding_stage": bool(stage.get("resumable_build", False)),
        },
        "embedding": {
            **stage["embedding"],
            "artifact_sha256": stage["embedding"]["sha256"],
        },
        "index": {
            "type": "IndexFlatIP",
            "metric": "inner_product_on_l2_normalized_vectors",
            "num_vectors": int(index.ntotal),
            "path": str(index_path.relative_to(PROJECT_ROOT)),
            "sha256": sha256_file(index_path),
        },
        "mapping": stage["mapping"],
        "dataset": stage["dataset"],
        "versions": {
            "embedding_stage": stage["versions"],
            "faiss_stage": {
                "python": platform.python_version(),
                "faiss": getattr(faiss, "__version__", "unknown"),
                "numpy": np.__version__,
            },
        },
    }
    write_json(manifest_path, manifest)
    shutil.copy2(manifest_path, METADATA_DIR / "latest-manifest.json")
    return manifest
