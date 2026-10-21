from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from ui_rag.rag.evidence import build_evidence_items
from ui_rag.rag.retrieve import retrieve_m5_text


def run_grounded_rag(
    *,
    query: str,
    base_manifest_path: Path,
    rerank_stage_path: Path,
    evidence_k: int = 4,
    candidate_k: int = 20,
    model_name: str,
    max_tokens: int = 96,
    temperature: float = 0.0,
    device: str | None = None,
    max_image_pixels: int = 1280 * 28 * 28,
    evidence_policy: str = "rank1",
    verify_hashes: bool = True,
) -> dict:
    """Run frozen M5 retrieval, then grounded answering in one isolated MLX child.

    M6.4 defaults to deterministic rank-1 evidence because the M6.2 dev experiment
    showed the categorical VLM selector underperformed the frozen retrieval rank-1
    baseline (50% vs 55% exact selection on retrieval-hit cases). M6.3 then showed that
    using the VLM as a verifier caused 90.9% false abstention when the exact target was
    already rank-1. Therefore the production VLM is generation-only: it sees the selected
    screenshot and produces a grounded explanation, but cannot rerank or reject evidence.
    The former selector remains available as an explicit diagnostic ablation.
    """
    del temperature  # two-stage path is intentionally deterministic
    total_started = time.perf_counter()
    retrieval = retrieve_m5_text(
        query=query,
        base_manifest_path=base_manifest_path,
        rerank_stage_path=rerank_stage_path,
        candidate_k=candidate_k,
        base_weight=0.5,
        rrf_constant=60,
        top_k=evidence_k,
        device=device,
        verify_hashes=verify_hashes,
    )
    evidence = build_evidence_items(retrieval["results"], evidence_k=evidence_k)

    with tempfile.TemporaryDirectory(prefix="ui-rag-m62-vlm-") as tmp:
        evidence_path = Path(tmp) / "evidence.json"
        evidence_path.write_text(
            json.dumps([x.as_dict() for x in evidence], ensure_ascii=False),
            encoding="utf-8",
        )
        cmd = [
            sys.executable,
            "-m",
            "ui_rag",
            "vlm-ground",
            "--query",
            query,
            "--evidence-file",
            str(evidence_path),
            "--model",
            model_name,
            "--evidence-policy",
            evidence_policy,
            "--selector-max-tokens",
            "8",
            "--selector-max-image-pixels",
            str(max_image_pixels),
            "--answer-max-tokens",
            str(max_tokens),
            "--answer-max-image-pixels",
            str(max_image_pixels),
        ]
        generation_started = time.perf_counter()
        proc = subprocess.run(cmd, check=True, text=True, capture_output=True)
        generation_ms = (time.perf_counter() - generation_started) * 1000.0
        try:
            grounded = json.loads(proc.stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"M6.4 VLM child returned invalid wrapper JSON: {proc.stdout}\n{proc.stderr}"
            ) from exc

    answer = grounded["answer"]
    selected = answer["selected_evidence_id"]
    allowed = {x.evidence_id for x in evidence}
    citations_valid = selected is None or selected in allowed
    if not citations_valid:
        raise RuntimeError(f"M6.4 child selected unsupported evidence ID: {selected}")

    retrieval_timings = retrieval.get("timings_ms", {})
    total_ms = (time.perf_counter() - total_started) * 1000.0
    return {
        "milestone": "M6.4",
        "query": query,
        "retrieval": retrieval["retrieval"],
        "evidence": [x.as_dict() for x in evidence],
        "grounding": {
            "architecture": grounded["grounding_architecture"],
            "evidence_policy": evidence_policy,
            "selector_parse_valid": grounded.get("selector_parse_valid"),
            "citations_valid": citations_valid,
            "allowed_evidence_ids": sorted(allowed),
            "selector": grounded["selector"],
        },
        "answer_generation": grounded.get("answer_generation"),
        "answer": answer,
        "timings_ms": {
            "encode_ms": retrieval_timings.get("encode_ms"),
            "search_ms": retrieval_timings.get("search_ms"),
            "rerank_ms": retrieval_timings.get("rerank_ms"),
            "generation_ms": generation_ms,
            "total_ms": total_ms,
        },
    }
