from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import numpy as np


class RetrievalEncoder(Protocol):
    """Minimal dual-encoder contract used by indexing and retrieval."""

    device: str
    model_name: str
    pretrained: str
    backend: str

    def encode_text(self, texts: str | Sequence[str]) -> np.ndarray: ...

    def encode_images(
        self, image_paths: str | Path | Sequence[str | Path]
    ) -> np.ndarray: ...

    def fuse(
        self,
        text_embedding: np.ndarray,
        image_embedding: np.ndarray,
        text_weight: float = 0.5,
    ) -> np.ndarray: ...

    def metadata(self) -> dict: ...


def fuse_normalized(
    text_embedding: np.ndarray,
    image_embedding: np.ndarray,
    text_weight: float = 0.5,
) -> np.ndarray:
    if not 0.0 <= text_weight <= 1.0:
        raise ValueError("text_weight must be between 0 and 1")
    if text_embedding.shape != image_embedding.shape:
        raise ValueError("text and image embeddings must have identical shapes")
    fused = text_weight * text_embedding + (1.0 - text_weight) * image_embedding
    fused /= np.clip(np.linalg.norm(fused, axis=1, keepdims=True), 1e-12, None)
    return np.ascontiguousarray(fused.astype(np.float32))
