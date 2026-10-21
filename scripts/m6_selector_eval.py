from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from ui_rag.rag.providers.mlx_vlm_provider import MLXVLMEngine
from ui_rag.rag.schema import EvidenceItem


def _read_cases(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cases", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--model", default="mlx-community/Qwen2.5-VL-3B-Instruct-4bit")
    p.add_argument("--selector-max-tokens", type=int, default=8)
    p.add_argument("--max-image-pixels", type=int, default=1280 * 28 * 28)
    args = p.parse_args()

    cases = _read_cases(Path(args.cases))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    engine = MLXVLMEngine(args.model)

    records: list[dict] = []
    parse_valid = 0
    hit_total = 0
    hit_selected = 0
    hit_top1_baseline = 0
    miss_total = 0
    miss_abstain = 0
    latencies: list[float] = []
    hit_target_ranks: list[int] = []

    for case in cases:
        evidence = [EvidenceItem.from_mapping(x) for x in case["evidence"]]
        started = time.perf_counter()
        record = {"case": case}
        try:
            selection = engine.select_evidence(
                query=case["query"],
                evidence=evidence,
                max_tokens=args.selector_max_tokens,
                max_image_pixels=args.max_image_pixels,
            )
            elapsed = time.perf_counter() - started
            latencies.append(elapsed)
            parse_valid += 1
            record.update(
                {
                    "parse_valid": True,
                    "selection_seconds": elapsed,
                    "selected_evidence_id": selection["selected_evidence_id"],
                    "abstain": selection["abstain"],
                    "raw_text": selection["raw_text"],
                    "contact_sheet": selection["contact_sheet"],
                }
            )
            if case["retrieval_hit"]:
                hit_total += 1
                target_id = case["target_evidence_id"]
                target_rank = int(target_id[1:])
                hit_target_ranks.append(target_rank)
                if target_id == "E1":
                    hit_top1_baseline += 1
                if selection["selected_evidence_id"] == target_id:
                    hit_selected += 1
            else:
                miss_total += 1
                if selection["abstain"]:
                    miss_abstain += 1
        except Exception as exc:  # noqa: BLE001 - evaluation must record per-case failures
            elapsed = time.perf_counter() - started
            latencies.append(elapsed)
            record.update(
                {
                    "parse_valid": False,
                    "selection_seconds": elapsed,
                    "error": str(exc),
                }
            )
            if case["retrieval_hit"]:
                hit_total += 1
                target_id = case["target_evidence_id"]
                target_rank = int(target_id[1:])
                hit_target_ranks.append(target_rank)
                if target_id == "E1":
                    hit_top1_baseline += 1
            else:
                miss_total += 1
        records.append(record)
        print(
            f"case={case['case_id']} hit={case['retrieval_hit']} "
            f"valid={record['parse_valid']} selected={record.get('selected_evidence_id')} "
            f"seconds={elapsed:.2f}"
        )

    num_cases = len(cases)
    metrics = {
        "milestone": "M6.2",
        "protocol": "categorical evidence selector on labeled contact sheet",
        "model": args.model,
        "num_cases": num_cases,
        "selector_parse_valid_rate": parse_valid / num_cases if num_cases else 0.0,
        "retrieval_hit_cases": hit_total,
        "evidence_selection_accuracy_on_hits": hit_selected / hit_total if hit_total else None,
        "retrieval_rank1_baseline_accuracy_on_hits": hit_top1_baseline / hit_total if hit_total else None,
        "selector_absolute_gain_over_rank1_on_hits": (
            (hit_selected - hit_top1_baseline) / hit_total if hit_total else None
        ),
        "end_to_end_exact_selection_rate": hit_selected / num_cases if num_cases else 0.0,
        "retrieval_miss_cases": miss_total,
        "abstain_rate_on_misses_diagnostic": miss_abstain / miss_total if miss_total else None,
        "hit_target_rank_median": statistics.median(hit_target_ranks) if hit_target_ranks else None,
        "selection_total_seconds": sum(latencies),
        "selection_mean_seconds": statistics.mean(latencies) if latencies else None,
        "selection_median_seconds": statistics.median(latencies) if latencies else None,
        "selection_p95_seconds": (
            sorted(latencies)[min(len(latencies) - 1, int(0.95 * len(latencies)))] if latencies else None
        ),
        "interpretation": (
            "Structural validity measures only categorical selector parsing. Exact target selection is scored "
            "only on retrieval-hit cases. Miss abstention remains diagnostic because semantically similar "
            "non-target screens can still satisfy a natural-language request."
        ),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    with (out_dir / "records.jsonl").open("w", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
