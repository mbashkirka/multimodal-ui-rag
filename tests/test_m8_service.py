from __future__ import annotations

from ui_rag.api.service import _sanitize_evidence, _sanitize_hit


def test_sanitize_hit_removes_path():
    row = {"rank": 1, "faiss_id": 1, "filename": "1.jpg", "path": "/secret/1.jpg",
           "score": 0.4, "base_rank": 2, "base_score": 0.3, "fine_score": 0.5}
    clean = _sanitize_hit(row)
    assert "path" not in clean
    assert clean["filename"] == "1.jpg"


def test_sanitize_evidence_removes_path():
    row = {"evidence_id": "E1", "rank": 1, "faiss_id": 1, "filename": "1.jpg",
           "path": "/secret/1.jpg", "retrieval_score": 0.4}
    clean = _sanitize_evidence(row)
    assert "path" not in clean
