from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from ui_rag.rag.schema import EvidenceItem
from ui_rag.rag.selector import (
    build_evidence_contact_sheet,
    build_selector_prompt,
    parse_evidence_choice,
)


def _evidence(tmp_path: Path, count: int = 4) -> list[EvidenceItem]:
    rows = []
    for idx in range(1, count + 1):
        path = tmp_path / f"{idx}.jpg"
        Image.new("RGB", (108, 192), "white").save(path)
        rows.append(
            EvidenceItem(
                evidence_id=f"E{idx}",
                rank=idx,
                faiss_id=idx,
                filename=path.name,
                path=str(path),
                retrieval_score=1.0 / idx,
            )
        )
    return rows


def test_parse_evidence_choice_accepts_short_label_and_none() -> None:
    allowed = {"E1", "E2", "E3", "E4"}
    assert parse_evidence_choice("E2", allowed_evidence_ids=allowed) == "E2"
    assert parse_evidence_choice("E3.", allowed_evidence_ids=allowed) == "E3"
    assert parse_evidence_choice("NONE", allowed_evidence_ids=allowed) == "NONE"


def test_parse_evidence_choice_rejects_ambiguous_or_unknown() -> None:
    allowed = {"E1", "E2", "E3", "E4"}
    with pytest.raises(ValueError):
        parse_evidence_choice("E1 or E2", allowed_evidence_ids=allowed)
    with pytest.raises(ValueError):
        parse_evidence_choice("E9", allowed_evidence_ids=allowed)


def test_selector_prompt_is_categorical(tmp_path: Path) -> None:
    evidence = _evidence(tmp_path)
    prompt = build_selector_prompt("bank login", evidence)
    assert "E1, E2, E3, E4, NONE" in prompt
    assert "Do not output JSON" in prompt
    assert "contact-sheet" in prompt


def test_contact_sheet_stays_within_visual_budget(tmp_path: Path) -> None:
    evidence = _evidence(tmp_path)
    out = tmp_path / "sheet.jpg"
    meta = build_evidence_contact_sheet(evidence, output_path=out, max_pixels=300_000)
    assert out.exists()
    assert meta["pixels"] <= 300_000
    assert [x["evidence_id"] for x in meta["items"]] == ["E1", "E2", "E3", "E4"]
    assert meta["columns"] == 2
