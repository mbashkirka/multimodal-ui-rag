from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from ui_rag.generation_eval.human_audit import RUBRIC, parse_human_scores_csv

FIELDS = [
    "audit_id",
    "human_faithfulness_0_2",
    "human_relevance_0_2",
    "human_unsupported_0_2",
    "human_completeness_0_2",
    "human_notes",
]
SCORE_FIELDS = FIELDS[1:5]


def _print_rubric() -> None:
    titles = {
        "faithfulness": "F = Faithfulness",
        "relevance": "R = Relevance",
        "unsupported": "U = Unsupported-claim severity",
        "completeness": "C = Completeness",
    }
    print("\nScoring rubric (0-2):")
    for key in ["faithfulness", "relevance", "unsupported", "completeness"]:
        print(f"  {titles[key]}")
        for score in [0, 1, 2]:
            print(f"    {score}: {RUBRIC[key][score]}")


def _atomic_write(rows: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix="human-audit-", suffix=".csv", dir=path.parent)
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        with tmp.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            for row in rows:
                w.writerow({k: row.get(k, "") for k in FIELDS})
        # Validate the exact file we are about to commit.
        parse_human_scores_csv(tmp)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def _load_progress(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    rows = parse_human_scores_csv(path)
    return {str(r["audit_id"]): {k: str(r.get(k, "")) for k in FIELDS} for r in rows}


def _asset_path(audit_dir: Path, case: dict) -> Path:
    return audit_dir / "assets" / f"{case['audit_id']}_{case['rank1_filename']}"


def _open_image(path: Path) -> None:
    if sys.platform == "darwin":
        subprocess.run(["open", str(path)], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _parse_scores(raw: str) -> tuple[str, str, str, str] | None:
    parts = raw.replace(",", " ").split()
    if len(parts) != 4 or any(x not in {"0", "1", "2"} for x in parts):
        return None
    return parts[0], parts[1], parts[2], parts[3]


def main() -> None:
    p = argparse.ArgumentParser(description="Terminal-based M7.2.2 blinded human audit; no browser required.")
    p.add_argument("--audit-dir", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--no-open-images", action="store_true", help="Do not open screenshots in macOS Preview.")
    p.add_argument("--restart", action="store_true", help="Ignore and overwrite existing audit progress.")
    args = p.parse_args()

    audit_dir = Path(args.audit_dir).resolve()
    manifest_path = audit_dir / "audit_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing {manifest_path}; run make m7-human-audit first")
    cases = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(cases, list) or not cases:
        raise ValueError("audit manifest is empty or invalid")

    out = Path(args.output).resolve()
    progress = {} if args.restart else _load_progress(out)
    case_ids = {str(c["audit_id"]) for c in cases}
    unknown = set(progress) - case_ids
    if unknown:
        raise ValueError(f"existing audit contains unknown IDs: {sorted(unknown)}")

    print("M7.2.2 terminal human audit")
    print("No browser or export button is used.")
    print(f"cases={len(cases)}")
    print(f"progress={len(progress)}/{len(cases)}")
    print(f"autosave={out}")
    print("Enter one line per case: F R U C, e.g. 2 2 0 2")
    print("Commands: q = save and quit; r = print rubric again")
    _print_rubric()

    for idx, case in enumerate(cases, start=1):
        aid = str(case["audit_id"])
        if aid in progress:
            print(f"\n[{idx}/{len(cases)}] {aid} already scored; skipping.")
            continue

        image = _asset_path(audit_dir, case)
        if not image.exists():
            raise FileNotFoundError(f"missing audit screenshot: {image}")
        if not args.no_open_images:
            _open_image(image)

        print("\n" + "=" * 88)
        print(f"[{idx}/{len(cases)}] {aid}")
        print(f"Evidence image: {image}")
        print("\nQUERY:\n" + str(case["query"]))
        print("\nGENERATED ANSWER:\n" + str(case["answer"]))

        while True:
            try:
                raw = input("\nScores F R U C [0-2 each] > ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print("\nStopped. Existing completed cases remain saved.")
                return
            if raw == "q":
                print(f"Saved progress {len(progress)}/{len(cases)} to {out}")
                return
            if raw == "r":
                _print_rubric()
                continue
            scores = _parse_scores(raw)
            if scores is None:
                print("Invalid input. Enter exactly four numbers 0, 1, or 2; example: 2 2 0 2")
                continue
            break

        try:
            notes = input("Notes [optional, Enter to skip] > ").strip()
        except (EOFError, KeyboardInterrupt):
            notes = ""

        progress[aid] = {
            "audit_id": aid,
            "human_faithfulness_0_2": scores[0],
            "human_relevance_0_2": scores[1],
            "human_unsupported_0_2": scores[2],
            "human_completeness_0_2": scores[3],
            "human_notes": notes,
        }
        ordered = [progress[str(c["audit_id"])] for c in cases if str(c["audit_id"]) in progress]
        _atomic_write(ordered, out)
        print(f"Autosaved: {len(progress)}/{len(cases)}")

    rows = parse_human_scores_csv(out)
    if len(rows) != len(cases):
        raise ValueError(f"audit incomplete: expected {len(cases)} rows, got {len(rows)}")
    print("\nM7.2.2 human audit COMPLETE")
    print(f"rows={len(rows)}")
    print(f"csv={out}")
    print("Next: make m7-human-report")


if __name__ == "__main__":
    main()
