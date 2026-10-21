from __future__ import annotations

import csv
import hashlib
import json
import random
import statistics
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SLICES = ("rank1_exact", "top4_nonrank1", "miss")
RUBRIC = {
    "faithfulness": {
        0: "Material claims conflict with or are unsupported by the screenshot.",
        1: "Mostly supported, but at least one material claim is uncertain or unsupported.",
        2: "All material claims are directly supported by the screenshot.",
    },
    "relevance": {
        0: "Does not address the user's request.",
        1: "Partially addresses the request or focuses on secondary details.",
        2: "Directly addresses the request using the visible evidence.",
    },
    "unsupported": {
        0: "No material unsupported claim.",
        1: "One minor unsupported/speculative claim.",
        2: "One or more major unsupported/speculative claims.",
    },
    "completeness": {
        0: "Misses the main visible information needed for the request.",
        1: "Covers the main point but omits useful visible details.",
        2: "Covers the important visible information relevant to the request.",
    },
}


@dataclass(frozen=True)
class AuditCase:
    audit_id: str
    case_id: str
    retrieval_slice: str
    query: str
    answer: str
    rank1_filename: str
    rank1_path: str
    target_filename: str


def _stable_seed(case_ids: Iterable[str]) -> int:
    payload = "\n".join(sorted(case_ids)).encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest()[:16], 16)


def select_blinded_audit_cases(rows: list[dict[str, Any]], *, total: int = 12) -> list[AuditCase]:
    if total < 3:
        raise ValueError("human audit size must be at least 3")
    per_slice = total // len(SLICES)
    remainder = total % len(SLICES)
    selected: list[dict[str, Any]] = []
    for i, name in enumerate(SLICES):
        need = per_slice + (1 if i < remainder else 0)
        pool = [r for r in rows if r.get("retrieval_slice") == name]
        if len(pool) < need:
            raise ValueError(f"not enough rows in slice {name}: need {need}, have {len(pool)}")
        selected.extend(pool[:need])

    seed = _stable_seed([r["case_id"] for r in selected])
    rng = random.Random(seed)
    rng.shuffle(selected)

    out: list[AuditCase] = []
    for idx, r in enumerate(selected, start=1):
        ev = r["evidence"][0]
        out.append(
            AuditCase(
                audit_id=f"A{idx:02d}",
                case_id=r["case_id"],
                retrieval_slice=r["retrieval_slice"],
                query=r["query"],
                answer=r["answer"]["answer"],
                rank1_filename=ev["filename"],
                rank1_path=ev["path"],
                target_filename=r.get("target_filename", ""),
            )
        )
    return out


def write_blinded_manifest(cases: list[AuditCase], path: Path) -> None:
    rows = []
    for c in cases:
        rows.append({
            "audit_id": c.audit_id,
            "case_id": c.case_id,
            "query": c.query,
            "answer": c.answer,
            "rank1_filename": c.rank1_filename,
        })
    path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")


def write_private_key(cases: list[AuditCase], path: Path) -> None:
    rows = []
    for c in cases:
        rows.append({
            "audit_id": c.audit_id,
            "case_id": c.case_id,
            "retrieval_slice": c.retrieval_slice,
            "rank1_filename": c.rank1_filename,
            "target_filename": c.target_filename,
        })
    path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")


def parse_human_scores_csv(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {
            "audit_id",
            "human_faithfulness_0_2",
            "human_relevance_0_2",
            "human_unsupported_0_2",
            "human_completeness_0_2",
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"human audit CSV missing columns: {sorted(missing)}")
        for r in reader:
            parsed = dict(r)
            for key in [
                "human_faithfulness_0_2",
                "human_relevance_0_2",
                "human_unsupported_0_2",
                "human_completeness_0_2",
            ]:
                raw = (r.get(key) or "").strip()
                if raw not in {"0", "1", "2"}:
                    raise ValueError(f"{r.get('audit_id')}: {key} must be 0, 1, or 2; got {raw!r}")
                parsed[key] = int(raw)
            rows.append(parsed)
    if not rows:
        raise ValueError("human audit CSV contains no scored rows")
    return rows


def summarize_human_scores(rows: list[dict[str, Any]], key_by_audit_id: dict[str, dict[str, Any]]) -> dict[str, Any]:
    enriched = []
    seen = set()
    for r in rows:
        aid = r["audit_id"]
        if aid in seen:
            raise ValueError(f"duplicate audit_id in scored CSV: {aid}")
        seen.add(aid)
        if aid not in key_by_audit_id:
            raise ValueError(f"unknown audit_id in scored CSV: {aid}")
        enriched.append({**r, **key_by_audit_id[aid]})

    def summary(items: list[dict[str, Any]]) -> dict[str, Any]:
        if not items:
            return {"n": 0}
        f = [x["human_faithfulness_0_2"] for x in items]
        r = [x["human_relevance_0_2"] for x in items]
        u = [x["human_unsupported_0_2"] for x in items]
        c = [x["human_completeness_0_2"] for x in items]
        return {
            "n": len(items),
            "faithfulness_normalized": statistics.mean(f) / 2.0,
            "relevance_normalized": statistics.mean(r) / 2.0,
            "completeness_normalized": statistics.mean(c) / 2.0,
            "unsupported_claim_case_rate": sum(x > 0 for x in u) / len(u),
            "major_unsupported_claim_case_rate": sum(x == 2 for x in u) / len(u),
            "mean_unsupported_severity": statistics.mean(u),
        }

    return {
        "overall": summary(enriched),
        "by_retrieval_slice": {
            name: summary([x for x in enriched if x["retrieval_slice"] == name]) for name in SLICES
        },
        "scored_audit_ids": [x["audit_id"] for x in enriched],
    }
