from __future__ import annotations

import argparse
import json
from pathlib import Path

from ui_rag.generation_eval.human_audit import parse_human_scores_csv, summarize_human_scores


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--scores", required=True)
    p.add_argument("--audit-key", required=True)
    p.add_argument("--deterministic-report", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    scored = parse_human_scores_csv(Path(args.scores))
    key_rows = json.loads(Path(args.audit_key).read_text(encoding="utf-8"))
    key = {x["audit_id"]: x for x in key_rows}
    human = summarize_human_scores(scored, key)
    base = json.loads(Path(args.deterministic_report).read_text(encoding="utf-8"))
    calibration = base.get("automated_judge_calibration")
    report = {
        "milestone": "M7.2",
        "protocol": "deterministic generation contracts + blinded human audit; same-family automated judge rejected by calibration",
        "automated_judge_reliable": False,
        "automated_judge_calibration": calibration,
        "deterministic_generation_metrics": {
            "overall": base.get("overall", {}),
            "by_retrieval_slice": base.get("by_retrieval_slice", {}),
        },
        "human_audit": human,
        "quality_claim_source": "blinded_human_audit",
        "interpretation_guardrail": "Automated Qwen2.5-VL F/R/U/C scores failed deterministic corruption calibration and are excluded from quality claims. Human audit is the reporting source for semantic generation quality.",
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("M7.2 human-audit report")
    print("automated_judge_reliable=False")
    overall = human["overall"]
    print(f"human_cases={overall['n']}")
    print(f"faithfulness={overall['faithfulness_normalized']*100:.2f}%")
    print(f"relevance={overall['relevance_normalized']*100:.2f}%")
    print(f"completeness={overall['completeness_normalized']*100:.2f}%")
    print(f"unsupported_case_rate={overall['unsupported_claim_case_rate']*100:.2f}%")
    print(f"major_unsupported_claim_case_rate={overall['major_unsupported_claim_case_rate']*100:.2f}%")
    print(f"json={out}")


if __name__ == "__main__":
    main()
