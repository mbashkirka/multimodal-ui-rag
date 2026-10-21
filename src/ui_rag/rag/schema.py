from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class EvidenceItem:
    evidence_id: str
    rank: int
    faiss_id: int
    filename: str
    path: str
    retrieval_score: float

    @classmethod
    def from_mapping(cls, item: Mapping[str, Any]) -> EvidenceItem:
        """Deserialize canonical or legacy M6 evidence records.

        M6.2 evaluation cases stored the fused RRF score under ``fused_score`` while
        runtime evidence uses ``retrieval_score``.  Accept both at the schema boundary
        and ignore unrelated metadata so saved evaluation cases remain reusable.
        """
        score = item.get("retrieval_score", item.get("fused_score"))
        if score is None:
            raise ValueError("evidence record requires retrieval_score or legacy fused_score")
        return cls(
            evidence_id=str(item["evidence_id"]),
            rank=int(item["rank"]),
            faiss_id=int(item["faiss_id"]),
            filename=str(item["filename"]),
            path=str(item["path"]),
            retrieval_score=float(score),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "rank": self.rank,
            "faiss_id": self.faiss_id,
            "filename": self.filename,
            "path": self.path,
            "retrieval_score": self.retrieval_score,
        }


@dataclass(frozen=True)
class GroundedAnswer:
    answer: str
    selected_evidence_id: str | None
    evidence_ids: tuple[str, ...]
    observations: tuple[dict[str, str], ...]
    confidence: float
    abstain: bool
    raw_text: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "selected_evidence_id": self.selected_evidence_id,
            "evidence_ids": list(self.evidence_ids),
            "observations": list(self.observations),
            "confidence": self.confidence,
            "abstain": self.abstain,
            "raw_text": self.raw_text,
        }
