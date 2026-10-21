from __future__ import annotations

from ui_rag.rag.schema import EvidenceItem


def build_grounded_ui_prompt(query: str, evidence: list[EvidenceItem]) -> str:
    inventory = "\n".join(
        f"- {item.evidence_id}: retrieved rank {item.rank}, file {item.filename}"
        for item in evidence
    )
    return f"""You are a grounded UI evidence analyst.

User request:
{query}

You are given {len(evidence)} retrieved mobile UI screenshots in exactly the same order as this evidence inventory:
{inventory}

Your job is to answer the user's request using ONLY visible information from these screenshots.
Do not invent labels, controls, states, colors, navigation paths, or functionality that is not visibly supported.
If the evidence is insufficient, set abstain=true and explain the limitation briefly.
When not abstaining, select the single screenshot that best supports the answer.
Every factual observation about a screenshot must cite its evidence_id.

Return ONLY one JSON object with exactly this shape:
{{
  "answer": "concise grounded answer",
  "selected_evidence_id": "E1 or null",
  "evidence_ids": ["E1"],
  "observations": [
    {{"evidence_id": "E1", "claim": "visible fact from that screenshot"}}
  ],
  "confidence": 0.0,
  "abstain": false
}}
"""


def build_selection_eval_prompt(query: str, evidence: list[EvidenceItem]) -> str:
    inventory = "\n".join(f"- {x.evidence_id}: file {x.filename}" for x in evidence)
    return f"""You are evaluating retrieved UI evidence.
The natural-language description is:
{query}

Candidate screenshots are supplied in this order:
{inventory}

Select the ONE screenshot that most directly matches the description. If none of the supplied screenshots sufficiently supports the description, abstain.
Use only visible evidence. Never cite an ID outside the inventory.
Return ONLY JSON:
{{"answer":"short reason","selected_evidence_id":"E1 or null","evidence_ids":["E1"],"observations":[{{"evidence_id":"E1","claim":"visible supporting fact"}}],"confidence":0.0,"abstain":false}}
"""
