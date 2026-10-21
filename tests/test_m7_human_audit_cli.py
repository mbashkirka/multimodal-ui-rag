from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "m7_human_audit_cli.py"
    spec = importlib.util.spec_from_file_location("m7_human_audit_cli", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_m722_parse_scores_accepts_space_and_comma():
    m = _load_module()
    assert m._parse_scores("2 2 0 2") == ("2", "2", "0", "2")
    assert m._parse_scores("2,2,0,2") == ("2", "2", "0", "2")


def test_m722_parse_scores_rejects_invalid_values():
    m = _load_module()
    assert m._parse_scores("2 3 0 2") is None
    assert m._parse_scores("2 2 0") is None


def test_m722_atomic_write_is_parseable(tmp_path: Path):
    m = _load_module()
    out = tmp_path / "scores.csv"
    rows = [{
        "audit_id": "A01",
        "human_faithfulness_0_2": "2",
        "human_relevance_0_2": "1",
        "human_unsupported_0_2": "0",
        "human_completeness_0_2": "2",
        "human_notes": "ok",
    }]
    m._atomic_write(rows, out)
    loaded = m.parse_human_scores_csv(out)
    assert loaded[0]["audit_id"] == "A01"
    assert loaded[0]["human_relevance_0_2"] == 1
