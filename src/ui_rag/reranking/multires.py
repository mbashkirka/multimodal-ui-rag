from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RerankConfig:
    candidate_k: int
    base_weight: float
    rrf_constant: int = 60

    def validate(self) -> None:
        if self.candidate_k <= 0:
            raise ValueError("candidate_k must be positive")
        if not 0.0 <= self.base_weight <= 1.0:
            raise ValueError("base_weight must be in [0, 1]")
        if self.rrf_constant < 0:
            raise ValueError("rrf_constant must be non-negative")


def reciprocal_rank_fusion_order(
    base_ids: np.ndarray,
    fine_scores: np.ndarray,
    *,
    base_weight: float,
    rrf_constant: int = 60,
) -> tuple[np.ndarray, np.ndarray]:
    """Return candidate ids ordered by weighted reciprocal-rank fusion.

    ``base_ids`` must already be in stage-1 rank order. ``fine_scores`` are the
    stage-2 scores for those same ids. Rank fusion is deliberately used rather
    than raw-score interpolation because p256 and p512 cosine distributions are
    not assumed to be calibrated identically.
    """
    if base_ids.ndim != 1 or fine_scores.ndim != 1:
        raise ValueError("base_ids and fine_scores must be one-dimensional")
    if len(base_ids) != len(fine_scores):
        raise ValueError("base_ids and fine_scores length mismatch")
    if not 0.0 <= base_weight <= 1.0:
        raise ValueError("base_weight must be in [0, 1]")
    if rrf_constant < 0:
        raise ValueError("rrf_constant must be non-negative")

    n = len(base_ids)
    base_ranks = np.arange(1, n + 1, dtype=np.float64)
    fine_order = np.argsort(-fine_scores, kind="stable")
    fine_ranks = np.empty(n, dtype=np.float64)
    fine_ranks[fine_order] = np.arange(1, n + 1, dtype=np.float64)
    fused = (
        base_weight / (rrf_constant + base_ranks)
        + (1.0 - base_weight) / (rrf_constant + fine_ranks)
    )
    # Stable sort preserves stage-1 order when fused scores tie.
    order = np.argsort(-fused, kind="stable")
    return base_ids[order], fused[order]


def target_rank(ids: np.ndarray, target_id: int, *, top_k: int | None = None) -> int | None:
    if top_k is not None:
        ids = ids[:top_k]
    positions = np.flatnonzero(ids == target_id)
    return int(positions[0]) + 1 if len(positions) else None
