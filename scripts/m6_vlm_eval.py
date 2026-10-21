from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from ui_rag.rag.parse import parse_grounded_answer
from ui_rag.rag.prompt import build_selection_eval_prompt
from ui_rag.rag.providers.mlx_vlm_provider import MLXVLMEngine
from ui_rag.rag.schema import EvidenceItem


def _read_cases(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cases", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--model", default="mlx-community/Qwen2.5-VL-3B-Instruct-4bit")
    p.add_argument("--max-tokens", type=int, default=180)
    p.add_argument("--max-image-pixels", type=int, default=1280 * 28 * 28)
    args = p.parse_args()

    cases = _read_cases(Path(args.cases))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    engine = MLXVLMEngine(args.model)
    records: list[dict] = []
    valid = 0
    hit_total = 0
    hit_selected = 0
    miss_total = 0
    miss_abstain = 0
    total_seconds = 0.0

    for case in cases:
        evidence = [EvidenceItem.from_mapping(x) for x in case["evidence"]]
        prompt = build_selection_eval_prompt(case["query"], evidence)
        started = time.perf_counter()
        generation = engine.generate(
            prompt=prompt,
            image_paths=[x.path for x in evidence],
            max_tokens=args.max_tokens,
            temperature=0.0,
            max_image_pixels=args.max_image_pixels,
        )
        elapsed = time.perf_counter() - started
        total_seconds += elapsed
        record = {"case": case, "generation_seconds": elapsed, "raw_text": generation["text"]}
        try:
            answer = parse_grounded_answer(
                generation["text"], allowed_evidence_ids={x.evidence_id for x in evidence}
            )
            valid += 1
            record["valid"] = True
            record["answer"] = answer.as_dict()
            if case["retrieval_hit"]:
                hit_total += 1
                if answer.selected_evidence_id == case["target_evidence_id"]:
                    hit_selected += 1
            else:
                miss_total += 1
                if answer.abstain:
                    miss_abstain += 1
        except Exception as exc:  # noqa: BLE001 - evaluation records per-case model failures
            record["valid"] = False
            record["error"] = str(exc)
            if case["retrieval_hit"]:
                hit_total += 1
            else:
                miss_total += 1
        records.append(record)
        print(
            f"case={case['case_id']} hit={case['retrieval_hit']} valid={record['valid']} "
            f"seconds={elapsed:.2f}"
        )

    metrics = {
        "milestone": "M6",
        "model": args.model,
        "max_image_pixels": args.max_image_pixels,
        "num_cases": len(cases),
        "structured_json_and_citation_valid_rate": valid / len(cases) if cases else 0.0,
        "retrieval_hit_cases": hit_total,
        "evidence_selection_accuracy_on_hits": hit_selected / hit_total if hit_total else None,
        "end_to_end_exact_selection_rate": hit_selected / len(cases) if cases else 0.0,
        "retrieval_miss_cases": miss_total,
        "abstain_rate_on_misses_diagnostic": miss_abstain / miss_total if miss_total else None,
        "generation_total_seconds": total_seconds,
        "generation_mean_seconds": total_seconds / len(cases) if cases else None,
        "interpretation": "Exact target selection is scored only when retrieval supplied the target. Miss abstention is diagnostic because semantically similar non-target screens may still support the natural-language request.",
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    with (out_dir / "records.jsonl").open("w", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
