from __future__ import annotations

from pathlib import Path

from ui_rag.rag.schema import EvidenceItem


def build_evidence_items(results: list[dict], *, evidence_k: int) -> list[EvidenceItem]:
    if evidence_k <= 0:
        raise ValueError("evidence_k must be positive")
    items: list[EvidenceItem] = []
    for rank, row in enumerate(results[:evidence_k], start=1):
        path = Path(row["path"])
        if not path.exists():
            raise FileNotFoundError(path)
        items.append(
            EvidenceItem(
                evidence_id=f"E{rank}",
                rank=rank,
                faiss_id=int(row["faiss_id"]),
                filename=str(row["filename"]),
                path=str(path),
                retrieval_score=float(row["score"]),
            )
        )
    return items
