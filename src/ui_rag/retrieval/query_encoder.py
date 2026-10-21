from __future__ import annotations

from pathlib import Path

import numpy as np

from ui_rag.embeddings.factory import create_encoder, encoder_kwargs_from_manifest
from ui_rag.indexing.manifest import read_json


def encode_query_to_file(
    *,
    manifest_path: Path,
    output_path: Path,
    query: str | None = None,
    image_path: Path | None = None,
    text_weight: float = 0.5,
    device: str | None = None,
) -> dict:
    """Encode a query in an encoder-only child process."""
    if not query and image_path is None:
        raise ValueError("provide at least query text or image_path")

    manifest = read_json(manifest_path)
    cfg = manifest["embedding"]
    encoder = create_encoder(**encoder_kwargs_from_manifest(cfg), device=device)

    text_embedding = encoder.encode_text(query) if query else None
    image_embedding = encoder.encode_images(image_path) if image_path is not None else None
    if text_embedding is not None and image_embedding is not None:
        vector = encoder.fuse(text_embedding, image_embedding, text_weight=text_weight)
    elif text_embedding is not None:
        vector = text_embedding
    else:
        assert image_embedding is not None
        vector = image_embedding

    expected_dim = int(cfg["dimension"])
    if vector.shape != (1, expected_dim):
        raise RuntimeError(f"query embedding shape mismatch: {vector.shape} != (1, {expected_dim})")
    if vector.dtype != np.float32 or not np.isfinite(vector).all():
        raise RuntimeError("query embedding must be finite float32")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, np.ascontiguousarray(vector))
    return {
        "device": encoder.device,
        "dimension": expected_dim,
        "query_type": (
            "multimodal" if query and image_path is not None else "text" if query else "image"
        ),
        "encoder": encoder.metadata(),
        "output": str(output_path),
    }
