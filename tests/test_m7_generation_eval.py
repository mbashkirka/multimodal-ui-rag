import csv
import tempfile
from pathlib import Path

from ui_rag.generation_eval.calibration import (
    make_variants,
    summarize_calibration,
    variant_detected,
)
from ui_rag.generation_eval.human_audit import (
    parse_human_scores_csv,
    select_blinded_audit_cases,
    summarize_human_scores,
)
from ui_rag.generation_eval.judge import JudgeScores, parse_judge_code
from ui_rag.generation_eval.metrics import deterministic_answer_checks


def test_parse_compact_judge_code():
    assert parse_judge_code('F2 R1 U0 C2') == JudgeScores(2, 1, 0, 2)


def test_parse_judge_code_rejects_ambiguous_multiple_codes():
    assert parse_judge_code('F2 R2 U0 C2 then F1 R1 U1 C1') is None


def test_parse_judge_code_tolerates_equals_and_case():
    assert parse_judge_code('f=2 r=2 u=0 c=1') == JudgeScores(2, 2, 0, 1)


def test_deterministic_answer_checks_valid_rank1_contract():
    record = {
        'evidence': [{'evidence_id': 'E1'}],
        'answer': {
            'answer': 'A dark banking login screen.',
            'selected_evidence_id': 'E1',
            'evidence_ids': ['E1'],
            'abstain': False,
        },
    }
    out = deterministic_answer_checks(record)
    assert out['citation_valid'] is True
    assert out['nonempty_answer'] is True
    assert out['ends_with_sentence_boundary'] is True


def test_deterministic_answer_checks_detects_truncated_ending():
    record = {
        'evidence': [{'evidence_id': 'E1'}],
        'answer': {
            'answer': 'The screen contains a username field and the',
            'selected_evidence_id': 'E1',
            'evidence_ids': ['E1'],
            'abstain': False,
        },
    }
    assert deterministic_answer_checks(record)['ends_with_sentence_boundary'] is False


def test_m71_calibration_variants_include_known_corruptions():
    variants = make_variants(
        answer="The screen shows a login form with username and password fields.",
        irrelevant_answer="A weather forecast lists rain and wind for tomorrow.",
    )
    assert [v.variant for v in variants] == [
        "original", "irrelevant", "unsupported_sentinel", "truncated"
    ]
    assert "CALIBRATION_SENTINEL_7Q9X" in variants[2].answer


def test_m71_variant_detection_contract():
    assert variant_detected("irrelevant", JudgeScores(2, 1, 0, 2))
    assert variant_detected("unsupported_sentinel", JudgeScores(1, 2, 0, 2))
    assert variant_detected("unsupported_sentinel", JudgeScores(2, 2, 1, 2))
    assert variant_detected("truncated", JudgeScores(2, 2, 0, 1))


def test_m71_calibration_fails_saturated_perfect_judge():
    rows = []
    perfect = {"faithfulness": 2, "relevance": 2, "unsupported": 0, "completeness": 2}
    for variant in ["original", "irrelevant", "unsupported_sentinel", "truncated"]:
        rows.append({"variant": variant, "scores": perfect})
    out = summarize_calibration(rows)
    assert out["calibration_pass"] is False
    assert out["corrupt_perfect_score_rate"] == 1.0


def test_m71_calibration_passes_sensitive_judge():
    rows = [
        {"variant": "original", "scores": {"faithfulness": 2, "relevance": 2, "unsupported": 0, "completeness": 2}},
        {"variant": "irrelevant", "scores": {"faithfulness": 0, "relevance": 0, "unsupported": 2, "completeness": 0}},
        {"variant": "unsupported_sentinel", "scores": {"faithfulness": 1, "relevance": 2, "unsupported": 2, "completeness": 2}},
        {"variant": "truncated", "scores": {"faithfulness": 2, "relevance": 1, "unsupported": 0, "completeness": 1}},
    ]
    out = summarize_calibration(rows)
    assert out["calibration_pass"] is True


def _audit_rows():
    rows = []
    for slice_name in ["rank1_exact", "top4_nonrank1", "miss"]:
        for i in range(4):
            rows.append({
                "case_id": f"{slice_name}-{i}",
                "retrieval_slice": slice_name,
                "query": f"query {i}",
                "answer": {"answer": f"answer {i}"},
                "evidence": [{"filename": f"{i}.jpg", "path": f"/tmp/{i}.jpg"}],
                "target_filename": f"target-{i}.jpg",
            })
    return rows


def test_m72_audit_selection_is_stratified_and_blinded_ids_are_unique():
    cases = select_blinded_audit_cases(_audit_rows(), total=12)
    assert len(cases) == 12
    assert len({c.audit_id for c in cases}) == 12
    counts = {s: sum(c.retrieval_slice == s for c in cases) for s in ["rank1_exact", "top4_nonrank1", "miss"]}
    assert counts == {"rank1_exact": 4, "top4_nonrank1": 4, "miss": 4}


def test_m72_human_score_parser_and_summary():
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "scores.csv"
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["audit_id","human_faithfulness_0_2","human_relevance_0_2","human_unsupported_0_2","human_completeness_0_2","human_notes"])
            w.writeheader()
            w.writerow({"audit_id":"A01","human_faithfulness_0_2":"2","human_relevance_0_2":"2","human_unsupported_0_2":"0","human_completeness_0_2":"1","human_notes":""})
        rows = parse_human_scores_csv(path)
        summary = summarize_human_scores(rows, {"A01":{"audit_id":"A01","retrieval_slice":"rank1_exact"}})
        assert summary["overall"]["faithfulness_normalized"] == 1.0
        assert summary["overall"]["completeness_normalized"] == 0.5
        assert summary["overall"]["unsupported_claim_case_rate"] == 0.0
