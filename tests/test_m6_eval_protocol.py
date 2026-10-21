from pathlib import Path


def test_eval_scores_selection_only_conditionally_on_retrieval_hit():
    source = Path("scripts/m6_vlm_eval.py").read_text(encoding="utf-8")
    assert "evidence_selection_accuracy_on_hits" in source
    assert "retrieval_hit_cases" in source
    assert "abstain_rate_on_misses_diagnostic" in source
    assert "semantically similar non-target screens" in source
