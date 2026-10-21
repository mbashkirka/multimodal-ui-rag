from ui_rag.evaluation.ann_metrics import (
    choose_ann_configuration,
    exact_neighbor_recall,
    task_rank_metrics,
    top1_agreement,
)


def test_exact_neighbor_recall_and_top1_agreement():
    exact = [[1, 2, 3, 4, 5], [10, 11, 12, 13, 14]]
    ann = [[1, 2, 9, 4, 8], [11, 10, 12, 99, 98]]
    assert exact_neighbor_recall(exact, ann, 5) == 0.6
    assert top1_agreement(exact, ann) == 0.5


def test_task_rank_metrics_top10():
    metrics = task_rank_metrics([1, 5, 10, 11, None])
    assert metrics["recall_at_1"] == 0.2
    assert metrics["recall_at_5"] == 0.4
    assert metrics["recall_at_10"] == 0.6
    assert metrics["mrr_at_10"] > 0
    assert metrics["ndcg_at_10"] > metrics["mrr_at_10"]


def _row(label, kind, neighbor, retention, speedup, size=220.0):
    return {
        "label": label,
        "kind": kind,
        "ann_quality": {"exact_neighbor_recall_at_10": neighbor},
        "task": {"recall_at_10_retention_vs_exact": retention},
        "latency": {"single_query_p50_speedup_vs_exact": speedup},
        "index": {"size_mib": size},
    }


def test_selection_can_keep_exact_when_ann_tradeoff_is_not_worth_it():
    rows = [
        _row("FlatIP exact", "exact", 1.0, 1.0, 1.0, 195.0),
        _row("HNSW", "hnsw", 0.99, 0.99, 1.10),
    ]
    choice = choose_ann_configuration(rows)
    assert choice["recommended_kind"] == "exact"


def test_selection_prefers_fastest_eligible_hnsw():
    rows = [
        _row("FlatIP exact", "exact", 1.0, 1.0, 1.0, 195.0),
        _row("HNSW-safe", "hnsw", 0.97, 0.99, 1.8, 220.0),
        _row("HNSW-fast", "hnsw", 0.96, 0.985, 2.2, 230.0),
        _row("HNSW-lossy", "hnsw", 0.90, 0.99, 3.0, 210.0),
    ]
    choice = choose_ann_configuration(rows)
    assert choice["recommended_kind"] == "hnsw"
    assert choice["recommended_label"] == "HNSW-fast"
