from __future__ import annotations

import os
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from numpy.lib.format import open_memmap
from tqdm import tqdm

from ui_rag.config import (
    DEFAULT_ARTIFACT_NAME,
    DEFAULT_MODEL_NAME,
    DEFAULT_PRETRAINED,
    EMBEDDING_DIR,
    LEGACY_IMAGE_MAP,
    METADATA_DIR,
    PROJECT_ROOT,
    RICO_DIR,
)
from ui_rag.dataset.mapping import load_canonical_mapping, load_legacy_mapping
from ui_rag.embeddings.factory import create_encoder, infer_backend
from ui_rag.indexing.manifest import read_json, sha256_file, write_json


def _ordered_mapping(
    mapping_path: Path, limit: int | None
) -> tuple[list[tuple[int, str]], list[dict]]:
    ordered, aliases = load_canonical_mapping(mapping_path)
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be positive")
        ordered = ordered[:limit]
    return ordered, aliases


def _progress_contract(
    *,
    artifact_name: str,
    mapping_path: Path,
    model_name: str,
    pretrained: str,
    encoder_backend: str,
    max_num_patches: int | None,
    num_vectors: int,
) -> dict:
    return {
        "artifact_name": artifact_name,
        "mapping_sha256": sha256_file(mapping_path),
        "model_name": model_name,
        "pretrained": pretrained,
        "backend": encoder_backend,
        "max_num_patches": max_num_patches,
        "num_vectors": num_vectors,
        "dtype": "float32",
    }


def _validate_progress(progress: dict, expected: dict) -> None:
    for key, value in expected.items():
        if progress.get(key) != value:
            raise RuntimeError(
                f"partial embedding artifact contract mismatch for {key}: "
                f"expected={value!r} got={progress.get(key)!r}. "
                "Run with --restart to discard the partial artifact."
            )


def _complete_contract(stage: dict) -> dict:
    emb = stage.get("embedding", {})
    return {
        "artifact_name": stage.get("artifact_name"),
        "mapping_sha256": stage.get("mapping", {}).get("source_sha256"),
        "model_name": emb.get("model_name"),
        "pretrained": emb.get("pretrained"),
        "backend": emb.get("backend", "open_clip"),
        "max_num_patches": emb.get("max_num_patches"),
        "num_vectors": emb.get("num_vectors"),
        "dtype": emb.get("dtype"),
    }


def build_embeddings(
    *,
    image_dir: Path = RICO_DIR,
    mapping_path: Path = LEGACY_IMAGE_MAP,
    model_name: str = DEFAULT_MODEL_NAME,
    pretrained: str = DEFAULT_PRETRAINED,
    encoder_backend: str | None = None,
    max_num_patches: int | None = None,
    artifact_name: str = DEFAULT_ARTIFACT_NAME,
    batch_size: int = 16,
    device: str | None = None,
    limit: int | None = None,
    restart: bool = False,
) -> dict:
    """Encode RICO with a selected dual encoder without importing FAISS.

    M3 keeps the M1 macOS isolation contract: this child owns PyTorch plus one
    vision-language encoder backend; a separate child later owns FAISS.  The
    embedding stage remains resumable at flushed batch boundaries.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if max_num_patches is not None and max_num_patches <= 0:
        raise ValueError("max_num_patches must be positive")

    resolved_backend = infer_backend(model_name, encoder_backend)
    ordered, aliases = _ordered_mapping(mapping_path, limit)
    image_paths = [image_dir / filename for _, filename in ordered]
    missing = [str(path) for path in image_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} canonical screenshots are missing. First paths:\n"
            + "\n".join(missing[:10])
        )

    EMBEDDING_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    final_embedding_path = EMBEDDING_DIR / f"{artifact_name}.npy"
    partial_embedding_path = EMBEDDING_DIR / f".{artifact_name}.partial.npy"
    progress_path = METADATA_DIR / f"{artifact_name}-embedding-progress.json"
    artifact_mapping_path = METADATA_DIR / f"{artifact_name}-image-map.json"
    stage_path = METADATA_DIR / f"{artifact_name}-embedding-stage.json"

    expected_contract = _progress_contract(
        artifact_name=artifact_name,
        mapping_path=mapping_path,
        model_name=model_name,
        pretrained=pretrained,
        encoder_backend=resolved_backend,
        max_num_patches=max_num_patches,
        num_vectors=len(ordered),
    )

    if restart:
        partial_embedding_path.unlink(missing_ok=True)
        progress_path.unlink(missing_ok=True)

    if final_embedding_path.exists() and not restart:
        if stage_path.exists():
            existing = read_json(stage_path)
            if (
                _complete_contract(existing) == expected_contract
                and sha256_file(final_embedding_path) == existing["embedding"]["sha256"]
            ):
                return existing
        raise RuntimeError(
            f"complete embedding artifact already exists at {final_embedding_path}, "
            "but its contract cannot be safely reused. Use --restart to rebuild it."
        )
    if restart:
        final_embedding_path.unlink(missing_ok=True)
        stage_path.unlink(missing_ok=True)
        artifact_mapping_path.unlink(missing_ok=True)

    encoder = create_encoder(
        model_name=model_name,
        pretrained=pretrained,
        backend=resolved_backend,
        device=device,
        max_num_patches=max_num_patches,
    )
    processed = 0
    dimension: int | None = None
    output = None

    if partial_embedding_path.exists() or progress_path.exists():
        if not (partial_embedding_path.exists() and progress_path.exists()):
            raise RuntimeError(
                "partial embedding state is incomplete. Run with --restart to rebuild it."
            )
        progress = read_json(progress_path)
        _validate_progress(progress, expected_contract)
        processed = int(progress.get("processed", 0))
        dimension = int(progress["dimension"])
        output = np.load(partial_embedding_path, mmap_mode="r+")
        if output.shape != (len(image_paths), dimension) or output.dtype != np.float32:
            raise RuntimeError(
                "partial embedding file shape/dtype does not match progress metadata. "
                "Run with --restart."
            )
        if not 0 <= processed <= len(image_paths):
            raise RuntimeError("invalid processed row count in progress metadata")
        print(f"Resuming embeddings at row {processed}/{len(image_paths)}")

    starts = range(processed, len(image_paths), batch_size)
    for start in tqdm(starts, desc=f"Embedding RICO [{resolved_backend}]"):
        vectors = encoder.encode_images(image_paths[start : start + batch_size])
        if vectors.ndim != 2 or vectors.dtype != np.float32 or not np.isfinite(vectors).all():
            raise RuntimeError(
                f"invalid embedding batch: shape={vectors.shape} dtype={vectors.dtype}"
            )

        norms = np.linalg.norm(vectors, axis=1)
        if not np.allclose(norms, 1.0, atol=2e-4):
            raise RuntimeError("encoder returned non-normalized corpus embeddings")

        if dimension is None:
            dimension = int(vectors.shape[1])
            output = open_memmap(
                partial_embedding_path,
                mode="w+",
                dtype=np.float32,
                shape=(len(image_paths), dimension),
            )
        elif vectors.shape[1] != dimension:
            raise RuntimeError("embedding dimension changed between batches")

        assert output is not None
        output[start : start + len(vectors)] = vectors
        output.flush()
        processed = start + len(vectors)
        write_json(
            progress_path,
            {
                **expected_contract,
                "dimension": dimension,
                "processed": processed,
                "device": encoder.device,
                "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            },
        )

    if output is None or dimension is None or processed != len(image_paths):
        raise RuntimeError("embedding build failed consistency checks")
    output.flush()
    del output
    os.replace(partial_embedding_path, final_embedding_path)
    progress_path.unlink(missing_ok=True)

    write_json(
        artifact_mapping_path,
        {str(i): filename for i, (_, filename) in enumerate(ordered)},
    )

    encoder_meta = encoder.metadata()
    stage = {
        "artifact_name": artifact_name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "embeddings",
        "resumable_build": True,
        "embedding": {
            "backend": encoder_meta["backend"],
            "model_name": model_name,
            "pretrained": pretrained,
            "max_num_patches": max_num_patches,
            "dimension": dimension,
            "l2_normalized": True,
            "dtype": "float32",
            "num_vectors": len(ordered),
            "device_used_for_build": encoder.device,
            "path": str(final_embedding_path.relative_to(PROJECT_ROOT)),
            "sha256": sha256_file(final_embedding_path),
        },
        "mapping": {
            "source_path": str(mapping_path),
            "source_sha256": sha256_file(mapping_path),
            "legacy_rows": len(load_legacy_mapping(mapping_path)),
            "canonical_rows_indexed": len(ordered),
            "legacy_alias_records_detected": len(aliases),
            "artifact_path": str(artifact_mapping_path.relative_to(PROJECT_ROOT)),
            "artifact_sha256": sha256_file(artifact_mapping_path),
        },
        "dataset": {
            "image_dir": str(image_dir),
            "limit": limit,
            "canonicalized_legacy_duplicates": True,
        },
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "encoder_library": encoder_meta.get("library"),
            "encoder_library_version": encoder_meta.get("library_version"),
        },
    }
    write_json(stage_path, stage)
    return stage
