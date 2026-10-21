import numpy as np

from ui_rag.reranking.multires import RerankConfig, reciprocal_rank_fusion_order, target_rank


def test_weighted_rrf_can_promote_fine_reranker_choice():
    base_ids = np.array([10, 20, 30], dtype=np.int64)
    fine_scores = np.array([0.1, 0.9, 0.2], dtype=np.float32)
    ordered, scores = reciprocal_rank_fusion_order(
        base_ids, fine_scores, base_weight=0.0, rrf_constant=60
    )
    assert ordered.tolist() == [20, 30, 10]
    assert scores[0] > scores[1] > scores[2]
    assert target_rank(ordered, 20) == 1


def test_weighted_rrf_base_weight_one_preserves_stage1_order():
    base_ids = np.array([10, 20, 30], dtype=np.int64)
    fine_scores = np.array([0.1, 0.9, 0.2], dtype=np.float32)
    ordered, _ = reciprocal_rank_fusion_order(
        base_ids, fine_scores, base_weight=1.0, rrf_constant=60
    )
    assert ordered.tolist() == base_ids.tolist()


def test_rerank_config_validation():
    RerankConfig(candidate_k=50, base_weight=0.25).validate()
