from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from ui_rag.rag.providers.mlx_vlm_provider import MLXVLMEngine
from ui_rag.rag.schema import EvidenceItem


def _pct(n: int, d: int) -> float:
    return n / d if d else 0.0


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cases", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--answer-max-tokens", type=int, default=64)
    p.add_argument("--max-image-pixels", type=int, default=1003520)
    args = p.parse_args()

    cases = [json.loads(line) for line in Path(args.cases).read_text(encoding="utf-8").splitlines() if line.strip()]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    engine = MLXVLMEngine(args.model)

    rows: list[dict] = []
    seconds: list[float] = []
    exact_rank1_hits = 0
    accepted_exact_rank1_hits = 0
    nonexact_rank1_hit_cases = 0
    abstained_nonexact_rank1_hits = 0
    miss_cases = 0
    abstained_misses = 0

    for case in cases:
        evidence = [EvidenceItem.from_mapping(item) for item in case["evidence"]]
        if not evidence:
            raise RuntimeError(f"case has no evidence: {case['case_id']}")
        e1 = evidence[0]
        start = time.perf_counter()
        answer = engine.answer_selected_evidence(
            query=case["query"],
            evidence_item=e1,
            max_tokens=args.answer_max_tokens,
            max_image_pixels=args.max_image_pixels,
        )
        elapsed = time.perf_counter() - start
        seconds.append(elapsed)
        abstain = bool(answer["insufficient"])
        rank1_exact = case.get("target_evidence_id") == "E1"
        retrieval_hit = bool(case.get("retrieval_hit"))

        if retrieval_hit and rank1_exact:
            exact_rank1_hits += 1
            if not abstain:
                accepted_exact_rank1_hits += 1
        elif retrieval_hit:
            nonexact_rank1_hit_cases += 1
            if abstain:
                abstained_nonexact_rank1_hits += 1
        else:
            miss_cases += 1
            if abstain:
                abstained_misses += 1

        row = {
            "case_id": case["case_id"],
            "query": case["query"],
            "retrieval_hit": retrieval_hit,
            "target_evidence_id": case.get("target_evidence_id"),
            "rank1_filename": e1.filename,
            "rank1_exact_target": rank1_exact,
            "abstain": abstain,
            "raw_text": answer["text"],
            "seconds": elapsed,
        }
        rows.append(row)
        print(
            f"case={case['case_id']} hit={retrieval_hit} rank1_exact={rank1_exact} "
            f"abstain={abstain} seconds={elapsed:.2f}"
        )

    summary = {
        "milestone": "M6.3",
        "protocol": "deterministic rank-1 evidence with single-image VLM verifier/answer",
        "model": args.model,
        "num_cases": len(cases),
        "exact_rank1_hit_cases": exact_rank1_hits,
        "accept_rate_when_rank1_is_exact_target": _pct(accepted_exact_rank1_hits, exact_rank1_hits),
        "false_abstain_rate_when_rank1_is_exact_target": 1.0 - _pct(accepted_exact_rank1_hits, exact_rank1_hits) if exact_rank1_hits else 0.0,
        "nonexact_rank1_hit_cases": nonexact_rank1_hit_cases,
        "abstain_rate_when_target_is_elsewhere_in_top4_diagnostic": _pct(abstained_nonexact_rank1_hits, nonexact_rank1_hit_cases),
        "retrieval_miss_cases": miss_cases,
        "abstain_rate_on_retrieval_misses_diagnostic": _pct(abstained_misses, miss_cases),
        "answer_total_seconds": sum(seconds),
        "answer_mean_seconds": statistics.mean(seconds) if seconds else 0.0,
        "answer_median_seconds": statistics.median(seconds) if seconds else 0.0,
        "answer_p95_seconds": sorted(seconds)[max(0, int(0.95 * len(seconds)) - 1)] if seconds else 0.0,
        "interpretation": (
            "The only strict semantic check here is whether the VLM abstains when the exact Screen2Words target is already rank-1. "
            "Abstention on non-exact or retrieval-miss cases is diagnostic only because a semantically similar non-target screen may still support the natural-language request."
        ),
    }
    (out_dir / "results.jsonl").write_text(
        "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows), encoding="utf-8"
    )
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
