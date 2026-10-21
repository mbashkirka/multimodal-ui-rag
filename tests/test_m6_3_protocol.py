from pathlib import Path


def test_m63_defaults_to_rank1_policy() -> None:
    orchestrator = Path("src/ui_rag/rag/orchestrator.py").read_text(encoding="utf-8")
    cli = Path("src/ui_rag/cli.py").read_text(encoding="utf-8")
    assert 'evidence_policy: str = "rank1"' in orchestrator
    assert 'choices=["rank1", "vlm-selector"], default="rank1"' in cli
    assert '"deterministic-rank1 -> single-image grounded VLM answer"' in cli


def test_m63_keeps_selector_as_diagnostic_ablation() -> None:
    cli = Path("src/ui_rag/cli.py").read_text(encoding="utf-8")
    provider = Path("src/ui_rag/rag/providers/mlx_vlm_provider.py").read_text(encoding="utf-8")
    assert 'args.evidence_policy == "rank1"' in cli
    assert "engine.select_evidence" in cli
    assert "def select_evidence" in provider


def test_m63_answer_eval_only_claims_strict_exact_rank1_false_abstention() -> None:
    source = Path("scripts/m6_answer_eval.py").read_text(encoding="utf-8")
    assert "accept_rate_when_rank1_is_exact_target" in source
    assert "false_abstain_rate_when_rank1_is_exact_target" in source
    assert "diagnostic only" in source
