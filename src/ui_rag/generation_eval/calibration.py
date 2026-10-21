from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .judge import JudgeScores

_SENTINEL = "CALIBRATION_SENTINEL_7Q9X"


@dataclass(frozen=True)
class CalibrationVariant:
    variant: str
    answer: str
    expected_failure: str | None


def make_variants(*, answer: str, irrelevant_answer: str) -> list[CalibrationVariant]:
    """Build deterministic judge-calibration counterfactuals.

    The variants keep the original query and screenshot fixed and change only the
    candidate answer.  This isolates whether the judge is sensitive to known answer
    defects instead of merely emitting a preferred score code.
    """
    words = [w for w in answer.split() if w]
    if len(words) <= 6:
        truncated = " ".join(words[: max(1, len(words) // 2)]) + "."
    else:
        truncated = " ".join(words[:6]) + "."
    unsupported = (
        answer.rstrip()
        + f" The screenshot also visibly contains a large purple banner reading '{_SENTINEL}'."
    )
    return [
        CalibrationVariant("original", answer, None),
        CalibrationVariant("irrelevant", irrelevant_answer, "relevance_or_faithfulness"),
        CalibrationVariant("unsupported_sentinel", unsupported, "unsupported_or_faithfulness"),
        CalibrationVariant("truncated", truncated, "completeness"),
    ]


def variant_detected(variant: str, scores: JudgeScores) -> bool:
    if variant == "original":
        return True
    if variant == "irrelevant":
        return scores.relevance < 2 or scores.faithfulness < 2
    if variant == "unsupported_sentinel":
        return scores.unsupported > 0 or scores.faithfulness < 2
    if variant == "truncated":
        return scores.completeness < 2
    raise ValueError(f"unknown calibration variant: {variant}")


def summarize_calibration(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    valid = [r for r in rows if r.get("scores") is not None]

    def subset(name: str) -> list[dict[str, Any]]:
        return [r for r in valid if r.get("variant") == name]

    def detection_rate(name: str) -> float:
        items = subset(name)
        if not items:
            return 0.0
        detected = 0
        for r in items:
            s = JudgeScores(**r["scores"])
            detected += int(variant_detected(name, s))
        return detected / len(items)

    corrupt = [r for r in valid if r.get("variant") != "original"]
    perfect_corrupt = sum(
        r["scores"] == {"faithfulness": 2, "relevance": 2, "unsupported": 0, "completeness": 2}
        for r in corrupt
    )
    unique_codes = {
        (r["scores"]["faithfulness"], r["scores"]["relevance"], r["scores"]["unsupported"], r["scores"]["completeness"])
        for r in valid
    }
    metrics = {
        "contract_valid_rate": len(valid) / total if total else 0.0,
        "irrelevant_detection_rate": detection_rate("irrelevant"),
        "unsupported_detection_rate": detection_rate("unsupported_sentinel"),
        "truncation_detection_rate": detection_rate("truncated"),
        "corrupt_perfect_score_rate": perfect_corrupt / len(corrupt) if corrupt else 0.0,
        "unique_score_codes": len(unique_codes),
    }
    gates = {
        "contract_valid_rate_gte_0_95": metrics["contract_valid_rate"] >= 0.95,
        "irrelevant_detection_rate_gte_0_80": metrics["irrelevant_detection_rate"] >= 0.80,
        "unsupported_detection_rate_gte_0_80": metrics["unsupported_detection_rate"] >= 0.80,
        "truncation_detection_rate_gte_0_50": metrics["truncation_detection_rate"] >= 0.50,
        "corrupt_perfect_score_rate_lte_0_20": metrics["corrupt_perfect_score_rate"] <= 0.20,
        "unique_score_codes_gte_3": metrics["unique_score_codes"] >= 3,
    }
    return {
        **metrics,
        "gates": gates,
        "calibration_pass": all(gates.values()),
    }
