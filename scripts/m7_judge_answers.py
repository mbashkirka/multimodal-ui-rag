from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from ui_rag.generation_eval.judge import build_judge_prompt, parse_judge_code
from ui_rag.rag.providers.mlx_vlm_provider import MLXVLMEngine


def _p95(values: list[float]) -> float:
    return sorted(values)[max(0, int(0.95 * len(values)) - 1)] if values else 0.0


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--answers", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--max-tokens", type=int, default=20)
    p.add_argument("--max-image-pixels", type=int, default=1003520)
    args = p.parse_args()

    rows = [json.loads(x) for x in Path(args.answers).read_text(encoding="utf-8").splitlines() if x.strip()]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    engine = MLXVLMEngine(args.model)
    judged = []
    timings = []
    retries = 0
    for row in rows:
        prompt = build_judge_prompt(query=row["query"], answer=row["answer"]["answer"])
        start = time.perf_counter()
        gen = engine.generate(prompt=prompt, image_paths=[row["evidence"][0]["path"]], max_tokens=args.max_tokens, temperature=0.0, max_image_pixels=args.max_image_pixels)
        scores = parse_judge_code(gen["text"])
        retried = False
        if scores is None:
            retries += 1
            retried = True
            retry_prompt = prompt + "\nYour previous format was invalid. Return ONLY: F# R# U# C# where each # is 0, 1, or 2."
            gen = engine.generate(prompt=retry_prompt, image_paths=[row["evidence"][0]["path"]], max_tokens=12, temperature=0.0, max_image_pixels=args.max_image_pixels)
            scores = parse_judge_code(gen["text"])
        elapsed = time.perf_counter() - start
        timings.append(elapsed)
        judged_row = {
            **row,
            "judge": {
                "valid": scores is not None,
                "retried": retried,
                "raw_text": gen["text"].strip(),
                "scores": scores.as_dict() if scores else None,
                "seconds": elapsed,
            },
        }
        judged.append(judged_row)
        print(f"case={row['case_id']} slice={row['retrieval_slice']} valid={scores is not None} code={gen['text'].strip()!r} seconds={elapsed:.2f}")

    valid = [x for x in judged if x["judge"]["valid"]]
    def avg(name: str) -> float:
        return statistics.mean(x["judge"]["scores"][name] for x in valid) / 2.0 if valid else 0.0
    summary = {
        "milestone": "M7",
        "stage": "automated_visual_judge",
        "judge_model": args.model,
        "num_cases": len(judged),
        "judge_contract_valid_rate": len(valid) / len(judged) if judged else 0.0,
        "judge_retry_rate": retries / len(judged) if judged else 0.0,
        "faithfulness_normalized": avg("faithfulness"),
        "relevance_normalized": avg("relevance"),
        "completeness_normalized": avg("completeness"),
        "unsupported_claim_case_rate": sum(x["judge"]["scores"]["unsupported"] > 0 for x in valid) / len(valid) if valid else 0.0,
        "major_unsupported_claim_case_rate": sum(x["judge"]["scores"]["unsupported"] == 2 for x in valid) / len(valid) if valid else 0.0,
        "judge_mean_seconds": statistics.mean(timings) if timings else 0.0,
        "judge_median_seconds": statistics.median(timings) if timings else 0.0,
        "judge_p95_seconds": _p95(timings),
        "methodological_note": "The automated judge uses the same local Qwen2.5-VL model family as generation and is therefore a diagnostic, not an independent gold-standard evaluator. Human audit is required for calibration.",
    }
    (out_dir / "judged.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in judged), encoding="utf-8")
    (out_dir / "judge_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
