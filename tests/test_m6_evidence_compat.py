from ui_rag.rag.schema import EvidenceItem


def _base_record() -> dict:
    return {
        "evidence_id": "E1",
        "rank": 1,
        "faiss_id": 42,
        "filename": "42.jpg",
        "path": "/tmp/42.jpg",
    }


def test_evidence_item_reads_legacy_fused_score_and_ignores_extra_fields() -> None:
    record = _base_record() | {"fused_score": 0.0123, "legacy_extra": "ignored"}
    item = EvidenceItem.from_mapping(record)
    assert item.retrieval_score == 0.0123
    assert item.evidence_id == "E1"


def test_evidence_item_prefers_canonical_retrieval_score() -> None:
    record = _base_record() | {"retrieval_score": 0.9, "fused_score": 0.1}
    item = EvidenceItem.from_mapping(record)
    assert item.retrieval_score == 0.9
    assert item.as_dict()["retrieval_score"] == 0.9
    assert "fused_score" not in item.as_dict()


def test_evidence_item_requires_a_score_field() -> None:
    try:
        EvidenceItem.from_mapping(_base_record())
    except ValueError as exc:
        assert "retrieval_score or legacy fused_score" in str(exc)
    else:
        raise AssertionError("expected missing-score validation failure")
