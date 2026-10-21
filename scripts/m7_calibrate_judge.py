from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from ui_rag.generation_eval.calibration import make_variants, summarize_calibration
from ui_rag.generation_eval.judge import build_judge_prompt, parse_judge_code
from ui_rag.rag.providers.mlx_vlm_provider import MLXVLMEngine


def _p95(values: list[float]) -> float:
    return sorted(values)[max(0, int(0.95 * len(values)) - 1)] if values else 0.0


def _select(rows: list[dict], per_slice: int) -> list[dict]:
    out: list[dict] = []
    for name in ["rank1_exact", "top4_nonrank1", "miss"]:
        out.extend([r for r in rows if r["retrieval_slice"] == name][:per_slice])
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--answers", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--per-slice", type=int, default=2)
    p.add_argument("--max-tokens", type=int, default=20)
    p.add_argument("--max-image-pixels", type=int, default=1003520)
    args = p.parse_args()

    rows = [json.loads(x) for x in Path(args.answers).read_text(encoding="utf-8").splitlines() if x.strip()]
    selected = _select(rows, args.per_slice)
    if len(selected) < 2:
        raise SystemExit("Need at least two source answers for calibration")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    engine = MLXVLMEngine(args.model)

    results: list[dict] = []
    timings: list[float] = []
    for i, row in enumerate(selected):
        irrelevant = selected[(i + 1) % len(selected)]["answer"]["answer"]
        variants = make_variants(answer=row["answer"]["answer"], irrelevant_answer=irrelevant)
        for v in variants:
            prompt = build_judge_prompt(query=row["query"], answer=v.answer)
            start = time.perf_counter()
            gen = engine.generate(
                prompt=prompt,
                image_paths=[row["evidence"][0]["path"]],
                max_tokens=args.max_tokens,
                temperature=0.0,
                max_image_pixels=args.max_image_pixels,
            )
            elapsed = time.perf_counter() - start
            timings.append(elapsed)
            scores = parse_judge_code(gen["text"])
            result = {
                "case_id": row["case_id"],
                "retrieval_slice": row["retrieval_slice"],
                "variant": v.variant,
                "query": row["query"],
                "image": row["evidence"][0]["path"],
                "answer": v.answer,
                "expected_failure": v.expected_failure,
                "raw_text": gen["text"].strip(),
                "scores": scores.as_dict() if scores else None,
                "seconds": elapsed,
            }
            results.append(result)
            print(
                f"case={row['case_id']} variant={v.variant} valid={scores is not None} "
                f"code={gen['text'].strip()!r} seconds={elapsed:.2f}"
            )

    calibration = summarize_calibration(results)
    summary = {
        "milestone": "M7.1",
        "stage": "automated_judge_calibration",
        "judge_model": args.model,
        "source_cases": len(selected),
        "num_judgements": len(results),
        **calibration,
        "judge_mean_seconds": statistics.mean(timings) if timings else 0.0,
        "judge_median_seconds": statistics.median(timings) if timings else 0.0,
        "judge_p95_seconds": _p95(timings),
        "interpretation": (
            "PASS means the judge demonstrated minimum sensitivity to deterministic corruptions; "
            "it does not make same-family judging an independent gold standard. FAIL means automated "
            "F/R/U/C scores must be treated as unreliable and human audit becomes the reporting source."
        ),
    }
    (out_dir / "calibration.jsonl").write_text(
        "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in results), encoding="utf-8"
    )
    (out_dir / "calibration_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
