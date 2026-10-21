from __future__ import annotations

import re
from dataclasses import dataclass

_JUDGE_RE = re.compile(r"\bF\s*=?\s*([0-2])\s+R\s*=?\s*([0-2])\s+U\s*=?\s*([0-2])\s+C\s*=?\s*([0-2])\b", re.IGNORECASE)


@dataclass(frozen=True)
class JudgeScores:
    faithfulness: int
    relevance: int
    unsupported: int
    completeness: int

    def as_dict(self) -> dict[str, int]:
        return {
            "faithfulness": self.faithfulness,
            "relevance": self.relevance,
            "unsupported": self.unsupported,
            "completeness": self.completeness,
        }


def parse_judge_code(text: str) -> JudgeScores | None:
    """Parse compact M7 judge output such as ``F2 R2 U0 C1``.

    The compact categorical contract deliberately keeps serialization work out of
    the VLM. Any prose around a single unambiguous code is tolerated; malformed or
    ambiguous outputs are rejected and may be retried by the caller.
    """
    matches = list(_JUDGE_RE.finditer(text.strip()))
    if len(matches) != 1:
        return None
    f, r, u, c = (int(x) for x in matches[0].groups())
    return JudgeScores(f, r, u, c)


def build_judge_prompt(*, query: str, answer: str) -> str:
    return f"""You are auditing one generated answer against ONE supplied mobile-UI screenshot.

User request:
{query}

Generated answer:
{answer}

Score only what can be verified from the screenshot and request.
F (faithfulness): 2=all material claims visually supported; 1=mostly supported with minor uncertainty; 0=material contradiction or unsupported claim.
R (relevance): 2=directly addresses the request; 1=partially addresses it; 0=mostly irrelevant.
U (unsupported-claim severity): 0=no material unsupported claim; 1=minor unsupported detail; 2=major hallucination/contradiction.
C (completeness): 2=covers the important visible aspects relevant to the request; 1=partially complete; 0=misses the request.

Return exactly one compact code and nothing else:
F2 R2 U0 C2
"""
