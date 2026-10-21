from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from ui_rag.generation_eval.metrics import deterministic_answer_checks
from ui_rag.rag.providers.mlx_vlm_provider import MLXVLMEngine
from ui_rag.rag.schema import EvidenceItem


def _p95(values: list[float]) -> float:
    return sorted(values)[max(0, int(0.95 * len(values)) - 1)] if values else 0.0


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cases", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--max-image-pixels", type=int, default=1003520)
    args = p.parse_args()

    cases = [json.loads(x) for x in Path(args.cases).read_text(encoding="utf-8").splitlines() if x.strip()]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    engine = MLXVLMEngine(args.model)
    rows = []
    timings = []
    for case in cases:
        e1 = EvidenceItem.from_mapping(case["evidence"][0])
        start = time.perf_counter()
        generated = engine.answer_selected_evidence(
            query=case["query"], evidence_item=e1, max_tokens=args.max_tokens,
            max_image_pixels=args.max_image_pixels,
        )
        elapsed = time.perf_counter() - start
        timings.append(elapsed)
        answer = {
            "answer": generated["text"].strip(),
            "selected_evidence_id": "E1",
            "evidence_ids": ["E1"],
            "observations": [{"evidence_id": "E1", "claim": generated["text"].strip()}],
            "confidence": None,
            "abstain": False,
            "raw_text": generated["text"].strip(),
        }
        row = {**case, "answer": answer, "generation_seconds": elapsed}
        row["deterministic_checks"] = deterministic_answer_checks(row)
        rows.append(row)
        print(f"case={case['case_id']} slice={case['retrieval_slice']} words={row['deterministic_checks']['word_count']} clean_end={row['deterministic_checks']['ends_with_sentence_boundary']} seconds={elapsed:.2f}")

    path = out_dir / "answers.jsonl"
    path.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows), encoding="utf-8")
    summary = {
        "milestone": "M7",
        "stage": "generation",
        "model": args.model,
        "num_cases": len(rows),
        "citation_valid_rate": sum(x["deterministic_checks"]["citation_valid"] for x in rows) / len(rows) if rows else 0.0,
        "nonempty_answer_rate": sum(x["deterministic_checks"]["nonempty_answer"] for x in rows) / len(rows) if rows else 0.0,
        "clean_sentence_end_rate": sum(x["deterministic_checks"]["ends_with_sentence_boundary"] for x in rows) / len(rows) if rows else 0.0,
        "generation_mean_seconds": statistics.mean(timings) if timings else 0.0,
        "generation_median_seconds": statistics.median(timings) if timings else 0.0,
        "generation_p95_seconds": _p95(timings),
    }
    (out_dir / "generation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"answers={path}")


if __name__ == "__main__":
    main()
