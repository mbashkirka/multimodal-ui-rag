from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "artifacts" / "evaluation" / "screen2words-test-primary-siglip2-naflex" / "metrics.json"
RERANK = ROOT / "artifacts" / "evaluation" / "m5-rerank-test" / "comparison.json"
FUSION = ROOT / "artifacts" / "evaluation" / "m5-fusion-test" / "comparison.json"
OUT = ROOT / "artifacts" / "evaluation" / "m5-comparison"


def _load(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"missing M5 result: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    baseline = _load(BASELINE)
    rerank = _load(RERANK)
    fusion = _load(FUSION)
    OUT.mkdir(parents=True, exist_ok=True)

    b = baseline["metrics"]
    r = rerank["selected"]["metrics"]
    f = fusion["selected"]["metrics"]
    rows = [
        {
            "system": "NaFlex p256 text-only baseline",
            "benchmark": "Screen2Words test",
            "recall_at_1": b["recall_at_1"],
            "recall_at_5": b["recall_at_5"],
            "recall_at_10": b["recall_at_10"],
            "mrr_at_10": b["mrr_at_10"],
            "ndcg_at_10": b["ndcg_at_10"],
        },
        {
            "system": "M5 p512 coarse-to-fine reranker",
            "benchmark": "Screen2Words test",
            "recall_at_1": r["recall_at_1"],
            "recall_at_5": r["recall_at_5"],
            "recall_at_10": r["recall_at_10"],
            "mrr_at_10": r["mrr_at_10"],
            "ndcg_at_10": r["ndcg_at_10"],
        },
        {
            "system": "M5 text+visual-exemplar fusion",
            "benchmark": "controlled visual-exemplar test",
            "recall_at_1": f["recall_at_1"],
            "recall_at_5": f["recall_at_5"],
            "recall_at_10": f["recall_at_10"],
            "mrr_at_10": f["mrr_at_10"],
            "ndcg_at_10": f["ndcg_at_10"],
        },
    ]
    payload = {
        "milestone": "M5",
        "text_only_primary": {
            "baseline_recall_at_10": b["recall_at_10"],
            "reranked_recall_at_10": r["recall_at_10"],
            "absolute_delta": r["recall_at_10"] - b["recall_at_10"],
            "relative_lift": (r["recall_at_10"] / b["recall_at_10"] - 1.0) if b["recall_at_10"] else None,
            "selected_config": {
                k: rerank["selected"][k] for k in ("candidate_k", "base_weight", "fine_weight", "rrf_constant")
            },
        },
        "multimodal_secondary": {
            "selected_text_weight": fusion["selected"]["text_weight"],
            "selected_image_weight": fusion["selected"]["image_weight"],
            "recall_at_10": f["recall_at_10"],
            "caveat": fusion["protocol"]["caveat"],
        },
        "rows": rows,
    }
    (OUT / "comparison.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (OUT / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print("M5 final comparison")
    print("system                                  R@1    R@5    R@10   MRR@10  nDCG@10")
    for row in rows:
        print(
            f"{row['system']:<39} {row['recall_at_1']*100:>6.2f}% {row['recall_at_5']*100:>6.2f}% "
            f"{row['recall_at_10']*100:>6.2f}% {row['mrr_at_10']:>7.4f} {row['ndcg_at_10']:>8.4f}"
        )
    print(f"json={OUT / 'comparison.json'}")
    print(f"csv={OUT / 'comparison.csv'}")


if __name__ == "__main__":
    main()
