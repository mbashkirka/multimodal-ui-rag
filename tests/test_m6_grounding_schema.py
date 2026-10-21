import pytest

from ui_rag.rag.parse import parse_grounded_answer


def test_grounded_answer_accepts_valid_citations():
    text = '{"answer":"Login form is visible.","selected_evidence_id":"E1","evidence_ids":["E1"],"observations":[{"evidence_id":"E1","claim":"A login form is visible"}],"confidence":0.8,"abstain":false}'
    out = parse_grounded_answer(text, allowed_evidence_ids={"E1", "E2"})
    assert out.selected_evidence_id == "E1"
    assert out.confidence == pytest.approx(0.8)
    assert out.abstain is False


def test_grounded_answer_rejects_unknown_citation():
    text = '{"answer":"x","selected_evidence_id":"E9","evidence_ids":["E9"],"observations":[],"confidence":0.5,"abstain":false}'
    with pytest.raises(ValueError, match="unsupported selected_evidence_id"):
        parse_grounded_answer(text, allowed_evidence_ids={"E1"})


def test_grounded_answer_accepts_json_fence():
    text = '```json\n{"answer":"insufficient","selected_evidence_id":null,"evidence_ids":[],"observations":[],"confidence":0.2,"abstain":true}\n```'
    out = parse_grounded_answer(text, allowed_evidence_ids={"E1"})
    assert out.abstain is True
    assert out.selected_evidence_id is None


def test_abstain_cannot_select_evidence():
    text = '{"answer":"no","selected_evidence_id":"E1","evidence_ids":["E1"],"observations":[],"confidence":0.2,"abstain":true}'
    with pytest.raises(ValueError, match="cannot select evidence"):
        parse_grounded_answer(text, allowed_evidence_ids={"E1"})
