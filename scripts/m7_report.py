from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


def _mean(rows: list[dict], key: str) -> float:
    vals = [r["judge"]["scores"][key] / 2.0 for r in rows if r["judge"]["valid"]]
    return statistics.mean(vals) if vals else 0.0


def _slice_summary(rows: list[dict]) -> dict:
    valid = [r for r in rows if r["judge"]["valid"]]
    return {
        "n": len(rows),
        "judge_valid_rate": len(valid) / len(rows) if rows else 0.0,
        "faithfulness": _mean(rows, "faithfulness"),
        "relevance": _mean(rows, "relevance"),
        "completeness": _mean(rows, "completeness"),
        "unsupported_case_rate": sum(r["judge"]["scores"]["unsupported"] > 0 for r in valid) / len(valid) if valid else 0.0,
        "clean_sentence_end_rate": sum(bool(r["deterministic_checks"]["ends_with_sentence_boundary"]) for r in rows) / len(rows) if rows else 0.0,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--judged", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--human-audit-size", type=int, default=12)
    p.add_argument("--calibration-summary")
    args = p.parse_args()
    rows = [json.loads(x) for x in Path(args.judged).read_text(encoding="utf-8").splitlines() if x.strip()]
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    slices = {name: _slice_summary([r for r in rows if r["retrieval_slice"] == name]) for name in ["rank1_exact", "top4_nonrank1", "miss"]}
    overall = _slice_summary(rows)
    calibration = None
    if args.calibration_summary and Path(args.calibration_summary).exists():
        calibration = json.loads(Path(args.calibration_summary).read_text(encoding="utf-8"))
    judge_reliable = bool(calibration and calibration.get("calibration_pass"))
    report = {
        "milestone": "M7.1",
        "protocol": "deterministic contracts + same-family visual judge + human audit",
        "overall": overall,
        "by_retrieval_slice": slices,
        "automated_judge_calibration": calibration,
        "automated_judge_reliable": judge_reliable,
        "interpretation_guardrail": (
            "Automated judge passed corruption calibration but remains a same-family diagnostic; human audit is still required."
            if judge_reliable else
            "Automated judge did NOT pass calibration (or calibration was not supplied). F/R/U/C scores are UNRELIABLE and must not be used as quality claims; use human audit."
        ),
    }
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    audit = []
    per_slice = max(1, args.human_audit_size // 3)
    for name in ["rank1_exact", "top4_nonrank1", "miss"]:
        audit.extend([r for r in rows if r["retrieval_slice"] == name][:per_slice])
    audit = audit[: args.human_audit_size]
    with (out / "human_audit.csv").open("w", encoding="utf-8", newline="") as f:
        fields = ["case_id","retrieval_slice","query","rank1_filename","target_filename","answer","judge_code","human_faithfulness_0_2","human_relevance_0_2","human_unsupported_0_2","human_completeness_0_2","human_notes"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in audit:
            w.writerow({
                "case_id": r["case_id"],
                "retrieval_slice": r["retrieval_slice"],
                "query": r["query"],
                "rank1_filename": r["evidence"][0]["filename"],
                "target_filename": r["target_filename"],
                "answer": r["answer"]["answer"],
                "judge_code": r["judge"]["raw_text"],
                "human_faithfulness_0_2": "",
                "human_relevance_0_2": "",
                "human_unsupported_0_2": "",
                "human_completeness_0_2": "",
                "human_notes": "",
            })

    print("M7.1 generation evaluation report")
    print(f"automated_judge_reliable={judge_reliable}")
    if calibration:
        print(f"calibration_pass={calibration.get('calibration_pass')}")
    print("automated score table" + (" (CALIBRATED DIAGNOSTIC)" if judge_reliable else " (UNRELIABLE - DO NOT USE AS QUALITY CLAIM)"))
    print("slice             n  faithful relevance complete unsupported clean_end")
    for name, s in [("overall", overall), *slices.items()]:
        print(f"{name:<17} {s['n']:>2}  {s['faithfulness']*100:7.2f}% {s['relevance']*100:8.2f}% {s['completeness']*100:7.2f}% {s['unsupported_case_rate']*100:9.2f}% {s['clean_sentence_end_rate']*100:8.2f}%")
    print(f"json={out / 'report.json'}")
    print(f"human_audit={out / 'human_audit.csv'}")


if __name__ == "__main__":
    main()
