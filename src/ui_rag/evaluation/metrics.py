from __future__ import annotations

import math
from statistics import mean, median


def recall_at_k(ranks: list[int | None], k: int) -> float:
    if not ranks:
        return 0.0
    return mean(1.0 if rank is not None and rank <= k else 0.0 for rank in ranks)


def mrr_at_k(ranks: list[int | None], k: int) -> float:
    if not ranks:
        return 0.0
    return mean(1.0 / rank if rank is not None and rank <= k else 0.0 for rank in ranks)


def ndcg_at_k(ranks: list[int | None], k: int) -> float:
    """nDCG for a benchmark with exactly one binary-relevant document per query."""
    if not ranks:
        return 0.0
    return mean(
        1.0 / math.log2(rank + 1) if rank is not None and rank <= k else 0.0
        for rank in ranks
    )


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    if not 0.0 <= q <= 1.0:
        raise ValueError("q must be in [0, 1]")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = q * (len(ordered) - 1)
    lo = math.floor(position)
    hi = math.ceil(position)
    if lo == hi:
        return ordered[lo]
    weight = position - lo
    return ordered[lo] * (1.0 - weight) + ordered[hi] * weight


def summarize_ranks(ranks: list[int | None], search_k: int) -> dict:
    hits = [rank for rank in ranks if rank is not None]
    return {
        "recall_at_1": recall_at_k(ranks, 1),
        "recall_at_5": recall_at_k(ranks, 5),
        "recall_at_10": recall_at_k(ranks, 10),
        "recall_at_50": recall_at_k(ranks, min(50, search_k)),
        "recall_at_100": recall_at_k(ranks, min(100, search_k)),
        "mrr_at_10": mrr_at_k(ranks, 10),
        "ndcg_at_10": ndcg_at_k(ranks, 10),
        "hits_within_search_k": len(hits),
        "misses_within_search_k": len(ranks) - len(hits),
        "median_rank_among_hits": float(median(hits)) if hits else None,
        "mean_rank_among_hits": float(mean(hits)) if hits else None,
    }
