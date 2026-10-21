from __future__ import annotations

import json
import re
from typing import Any

from ui_rag.rag.schema import GroundedAnswer

_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)


def _extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    candidates = [stripped]
    match = _FENCE_RE.search(stripped)
    if match:
        candidates.insert(0, match.group(1))
    first = stripped.find("{")
    last = stripped.rfind("}")
    if first >= 0 and last > first:
        candidates.append(stripped[first : last + 1])
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("VLM output does not contain a valid JSON object")


def parse_grounded_answer(text: str, *, allowed_evidence_ids: set[str]) -> GroundedAnswer:
    payload = _extract_json_object(text)
    answer = str(payload.get("answer", "")).strip()
    selected = payload.get("selected_evidence_id")
    if selected is not None:
        selected = str(selected)
        if selected not in allowed_evidence_ids:
            raise ValueError(f"unsupported selected_evidence_id: {selected}")

    evidence_ids_raw = payload.get("evidence_ids", [])
    if not isinstance(evidence_ids_raw, list):
        raise TypeError("evidence_ids must be a list")
    evidence_ids = tuple(str(item) for item in evidence_ids_raw)
    unknown = set(evidence_ids) - allowed_evidence_ids
    if unknown:
        raise ValueError(f"unsupported evidence_ids: {sorted(unknown)}")

    observations_raw = payload.get("observations", [])
    if not isinstance(observations_raw, list):
        raise TypeError("observations must be a list")
    observations: list[dict[str, str]] = []
    for item in observations_raw:
        if not isinstance(item, dict):
            raise TypeError("each observation must be an object")
        evidence_id = str(item.get("evidence_id", ""))
        claim = str(item.get("claim", "")).strip()
        if evidence_id not in allowed_evidence_ids:
            raise ValueError(f"observation cites unsupported evidence: {evidence_id}")
        if not claim:
            raise ValueError("observation claim cannot be empty")
        observations.append({"evidence_id": evidence_id, "claim": claim})

    confidence = float(payload.get("confidence", 0.0))
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence must be in [0, 1]")
    abstain = bool(payload.get("abstain", False))
    if abstain and selected is not None:
        raise ValueError("abstaining answers cannot select evidence")
    if not abstain and selected is None:
        raise ValueError("non-abstaining answers must select evidence")
    if selected is not None and selected not in evidence_ids:
        evidence_ids = (selected, *evidence_ids)

    return GroundedAnswer(
        answer=answer,
        selected_evidence_id=selected,
        evidence_ids=evidence_ids,
        observations=tuple(observations),
        confidence=confidence,
        abstain=abstain,
        raw_text=text,
    )
