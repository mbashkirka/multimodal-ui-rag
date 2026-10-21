from ui_rag.evaluation.metrics import mrr_at_k, ndcg_at_k, recall_at_k, summarize_ranks


def test_single_relevant_metrics():
    ranks = [1, 2, 7, None]
    assert recall_at_k(ranks, 1) == 0.25
    assert recall_at_k(ranks, 5) == 0.5
    assert mrr_at_k(ranks, 10) == (1 + 0.5 + 1 / 7 + 0) / 4
    assert 0 < ndcg_at_k(ranks, 10) < 1


def test_rank_summary_counts_misses():
    summary = summarize_ranks([1, 5, None], search_k=100)
    assert summary["recall_at_1"] == 1 / 3
    assert summary["recall_at_5"] == 2 / 3
    assert summary["hits_within_search_k"] == 2
    assert summary["misses_within_search_k"] == 1
