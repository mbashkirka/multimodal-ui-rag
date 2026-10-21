from __future__ import annotations

import numpy as np


def fuse_text_and_reference(
    text_embeddings: np.ndarray,
    reference_embeddings: np.ndarray,
    *,
    text_weight: float,
) -> np.ndarray:
    if text_embeddings.shape != reference_embeddings.shape:
        raise ValueError("text/reference embedding shape mismatch")
    if not 0.0 <= text_weight <= 1.0:
        raise ValueError("text_weight must be in [0, 1]")
    fused = text_weight * text_embeddings + (1.0 - text_weight) * reference_embeddings
    norms = np.linalg.norm(fused, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return np.ascontiguousarray((fused / norms).astype(np.float32))


def choose_nonself_reference_ids(neighbor_ids: np.ndarray, target_ids: np.ndarray) -> np.ndarray:
    """Choose the first nearest-neighbour id that is not the target itself."""
    if neighbor_ids.ndim != 2:
        raise ValueError("neighbor_ids must be rank-2")
    if target_ids.ndim != 1 or len(target_ids) != len(neighbor_ids):
        raise ValueError("target_ids must align with neighbor rows")
    selected = np.full(len(target_ids), -1, dtype=np.int64)
    for row_idx, target in enumerate(target_ids):
        for candidate in neighbor_ids[row_idx]:
            candidate = int(candidate)
            if candidate >= 0 and candidate != int(target):
                selected[row_idx] = candidate
                break
    if np.any(selected < 0):
        raise RuntimeError("could not find a non-self visual reference for every query")
    return selected
