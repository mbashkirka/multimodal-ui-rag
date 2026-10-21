from ui_rag.rag.prompt import build_grounded_ui_prompt
from ui_rag.rag.schema import EvidenceItem


def test_prompt_enforces_grounding_and_abstention():
    evidence = [EvidenceItem("E1", 1, 7, "7.jpg", "/tmp/7.jpg", 0.1)]
    prompt = build_grounded_ui_prompt("find a login screen", evidence)
    assert "ONLY visible information" in prompt
    assert "abstain=true" in prompt
    assert "Every factual observation" in prompt
    assert "E1" in prompt
    assert "selected_evidence_id" in prompt
