from pathlib import Path


def test_m64_production_vlm_is_generation_only() -> None:
    cli = Path("src/ui_rag/cli.py").read_text(encoding="utf-8")
    prompt = Path("src/ui_rag/rag/selector.py").read_text(encoding="utf-8")
    provider = Path("src/ui_rag/rag/providers/mlx_vlm_provider.py").read_text(encoding="utf-8")
    assert '"deterministic-rank1 -> single-image grounded VLM answer"' in cli
    assert "Your job is NOT to accept/reject or rerank it" in prompt
    assert "INSUFFICIENT_EVIDENCE" not in prompt
    assert '"insufficient": False' in provider


def test_m64_rank1_answer_keeps_deterministic_citation() -> None:
    cli = Path("src/ui_rag/cli.py").read_text(encoding="utf-8")
    assert '"selected_evidence_id": selected_id' in cli
    assert '"evidence_ids": [selected_id]' in cli
    assert '"abstain": False' in cli


def test_vlm_generate_does_not_receive_evidence_policy() -> None:
    cli = Path("src/ui_rag/cli.py").read_text(encoding="utf-8")
    section = cli.split("def _vlm_generate_command", 1)[1].split("def _vlm_ground_command", 1)[0]
    assert "evidence_policy=" not in section
