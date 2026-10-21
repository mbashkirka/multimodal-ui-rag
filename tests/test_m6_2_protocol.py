from pathlib import Path


def test_m62_orchestrator_uses_single_mlx_child_and_deterministic_citations() -> None:
    source = Path("src/ui_rag/rag/orchestrator.py").read_text(encoding="utf-8")
    assert '"vlm-ground"' in source
    assert '"--evidence-policy"' in source
    assert 'evidence_policy: str = "rank1"' in source


def test_m62_eval_is_selector_only() -> None:
    source = Path("scripts/m6_selector_eval.py").read_text(encoding="utf-8")
    assert "engine.select_evidence" in source
    assert "retrieval_rank1_baseline_accuracy_on_hits" in source
    assert "selector_parse_valid_rate" in source
    assert "answer_selected_evidence" not in source


def test_makefile_m62_eval_uses_dev_cache() -> None:
    source = Path("Makefile").read_text(encoding="utf-8")
    assert "M6_DEV_DIR ?= artifacts/evaluation/screen2words-dev-primary-siglip2-naflex" in source
    assert "scripts/m6_selector_eval.py" in source
