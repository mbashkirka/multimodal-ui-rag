from __future__ import annotations

import re
from typing import Any

_SENTENCE_END = re.compile(r"[.!?][\"')\]]?$", re.UNICODE)


def deterministic_answer_checks(record: dict[str, Any]) -> dict[str, Any]:
    answer = record.get("answer", {})
    text = str(answer.get("answer", "")).strip()
    selected = answer.get("selected_evidence_id")
    evidence_ids = list(answer.get("evidence_ids") or [])
    evidence = list(record.get("evidence") or [])
    allowed = {str(x.get("evidence_id")) for x in evidence}
    citation_valid = (
        selected == "E1"
        and evidence_ids == ["E1"]
        and selected in allowed
        and not bool(answer.get("abstain"))
    )
    words = [w for w in text.split() if w]
    return {
        "nonempty_answer": bool(text),
        "citation_valid": citation_valid,
        "selected_rank1": selected == "E1",
        "word_count": len(words),
        "ends_with_sentence_boundary": bool(_SENTENCE_END.search(text)),
        "contains_structural_abstention": bool(answer.get("abstain")),
    }
