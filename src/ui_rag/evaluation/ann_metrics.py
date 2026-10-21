from __future__ import annotations

from statistics import mean


def exact_neighbor_recall(exact_ids: list[list[int]], ann_ids: list[list[int]], k: int) -> float:
    """Average fraction of exact top-k neighbours recovered by ANN top-k."""
    if k <= 0:
        raise ValueError("k must be positive")
    if len(exact_ids) != len(ann_ids):
        raise ValueError("exact_ids and ann_ids must have equal length")
    if not exact_ids:
        return 0.0

    recalls: list[float] = []
    for exact_row, ann_row in zip(exact_ids, ann_ids, strict=True):
        exact_top = [idx for idx in exact_row[:k] if idx >= 0]
        ann_top = {idx for idx in ann_row[:k] if idx >= 0}
        denominator = len(exact_top)
        recalls.append(
            (len(set(exact_top).intersection(ann_top)) / denominator) if denominator else 0.0
        )
    return mean(recalls)


def top1_agreement(exact_ids: list[list[int]], ann_ids: list[list[int]]) -> float:
    if len(exact_ids) != len(ann_ids):
        raise ValueError("exact_ids and ann_ids must have equal length")
    if not exact_ids:
        return 0.0
    agreements = []
    for exact_row, ann_row in zip(exact_ids, ann_ids, strict=True):
        exact_top = exact_row[0] if exact_row else -1
        ann_top = ann_row[0] if ann_row else -1
        agreements.append(1.0 if exact_top >= 0 and exact_top == ann_top else 0.0)
    return mean(agreements)


def task_rank_metrics(ranks: list[int | None]) -> dict[str, float]:
    """Top-10 task metrics for a single-relevant-document benchmark."""
    if not ranks:
        return {
            "recall_at_1": 0.0,
            "recall_at_5": 0.0,
            "recall_at_10": 0.0,
            "mrr_at_10": 0.0,
            "ndcg_at_10": 0.0,
        }

    import math

    n = len(ranks)
    return {
        "recall_at_1": sum(rank is not None and rank <= 1 for rank in ranks) / n,
        "recall_at_5": sum(rank is not None and rank <= 5 for rank in ranks) / n,
        "recall_at_10": sum(rank is not None and rank <= 10 for rank in ranks) / n,
        "mrr_at_10": sum(
            (1.0 / rank) if rank is not None and rank <= 10 else 0.0 for rank in ranks
        )
        / n,
        "ndcg_at_10": sum(
            (1.0 / math.log2(rank + 1)) if rank is not None and rank <= 10 else 0.0
            for rank in ranks
        )
        / n,
    }


def choose_ann_configuration(
    rows: list[dict],
    *,
    min_neighbor_recall_at_10: float = 0.95,
    min_task_recall10_retention: float = 0.98,
    min_single_query_speedup: float = 1.20,
) -> dict:
    """Choose an ANN configuration under explicit quality-retention constraints.

    If no ANN row satisfies both quality constraints and the minimum online speedup,
    the function recommends the exact FlatIP baseline. This makes a negative M4 result
    a valid engineering outcome instead of forcing ANN into production.
    """
    exact_rows = [row for row in rows if row.get("kind") == "exact"]
    if len(exact_rows) != 1:
        raise ValueError("rows must contain exactly one exact baseline")
    exact = exact_rows[0]

    eligible: list[dict] = []
    for row in rows:
        if row.get("kind") != "hnsw":
            continue
        if row["ann_quality"]["exact_neighbor_recall_at_10"] < min_neighbor_recall_at_10:
            continue
        if row["task"]["recall_at_10_retention_vs_exact"] < min_task_recall10_retention:
            continue
        if row["latency"]["single_query_p50_speedup_vs_exact"] < min_single_query_speedup:
            continue
        eligible.append(row)

    if not eligible:
        return {
            "recommended_kind": "exact",
            "recommended_label": exact["label"],
            "reason": (
                "No HNSW configuration met the required neighbour recall, task-retention, "
                "and online-latency speedup constraints."
            ),
        }

    eligible.sort(
        key=lambda row: (
            row["latency"]["single_query_p50_speedup_vs_exact"],
            row["ann_quality"]["exact_neighbor_recall_at_10"],
            -row["index"]["size_mib"],
        ),
        reverse=True,
    )
    winner = eligible[0]
    return {
        "recommended_kind": "hnsw",
        "recommended_label": winner["label"],
        "reason": "Fastest eligible HNSW configuration under the frozen quality constraints.",
    }
