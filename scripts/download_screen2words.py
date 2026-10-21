from __future__ import annotations

import csv
import hashlib
import json
import urllib.request
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "data_storage" / "benchmarks" / "screen2words" / "source"
FILES = {
    "screen_summaries.csv": "https://github.com/google-research-datasets/screen2words/raw/refs/heads/main/screen_summaries.csv",
    "dev_screens.txt": "https://github.com/google-research-datasets/screen2words/raw/refs/heads/main/split/dev_screens.txt",
    "test_screens.txt": "https://github.com/google-research-datasets/screen2words/raw/refs/heads/main/split/test_screens.txt",
}
EXPECTED_SUMMARY_ROWS = 112_085
EXPECTED_SCREENS = 22_417
EXPECTED_DEV_SCREENS = 2_364
EXPECTED_TEST_SCREENS = 4_310
EXPECTED_SUMMARIES_PER_SCREEN = 5


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _download(filename: str, url: str) -> Path:
    target = SOURCE_DIR / filename
    if target.exists() and target.stat().st_size > 0:
        print(f"Present: {target} ({target.stat().st_size} bytes, sha256={sha256(target)})")
        return target
    print(f"Downloading {filename} from official Screen2Words repository...")
    urllib.request.urlretrieve(url, target)
    print(f"Saved: {target} ({target.stat().st_size} bytes, sha256={sha256(target)})")
    return target


def _load_split(path: Path, expected: int, label: str) -> list[str]:
    ids = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(ids) != expected or len(set(ids)) != expected:
        raise RuntimeError(f"Screen2Words {label} split must contain {expected:,} unique screen ids")
    return ids


def validate(summaries_path: Path, dev_path: Path, test_path: Path) -> dict:
    counts: Counter[str] = Counter()
    rows = 0
    with summaries_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["screenId", "summary"]:
            raise RuntimeError(f"unexpected Screen2Words schema: {reader.fieldnames}")
        for row in reader:
            counts[row["screenId"].strip()] += 1
            rows += 1

    dev_ids = _load_split(dev_path, EXPECTED_DEV_SCREENS, "dev")
    test_ids = _load_split(test_path, EXPECTED_TEST_SCREENS, "test")
    if rows != EXPECTED_SUMMARY_ROWS:
        raise RuntimeError(f"summary row count mismatch: {rows} != {EXPECTED_SUMMARY_ROWS}")
    if len(counts) != EXPECTED_SCREENS:
        raise RuntimeError(f"unique screen count mismatch: {len(counts)} != {EXPECTED_SCREENS}")
    if set(counts.values()) != {EXPECTED_SUMMARIES_PER_SCREEN}:
        raise RuntimeError("expected exactly five summaries for every Screen2Words screen")
    for label, ids in (("dev", dev_ids), ("test", test_ids)):
        missing = [screen_id for screen_id in ids if screen_id not in counts]
        if missing:
            raise RuntimeError(f"{len(missing)} {label} screen ids have no summaries")
    overlap = set(dev_ids) & set(test_ids)
    if overlap:
        raise RuntimeError(f"Screen2Words dev/test overlap detected: {len(overlap)} ids")

    return {
        "dataset": "Screen2Words",
        "summary_rows": rows,
        "unique_screens": len(counts),
        "summaries_per_screen": EXPECTED_SUMMARIES_PER_SCREEN,
        "dev_screens": len(dev_ids),
        "test_screens": len(test_ids),
        "dev_test_disjoint": True,
        "files": {
            name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
            for name, path in {
                "screen_summaries.csv": summaries_path,
                "dev_screens.txt": dev_path,
                "test_screens.txt": test_path,
            }.items()
        },
    }


def main() -> None:
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    downloaded = {name: _download(name, url) for name, url in FILES.items()}
    manifest = validate(
        downloaded["screen_summaries.csv"],
        downloaded["dev_screens.txt"],
        downloaded["test_screens.txt"],
    )
    manifest_path = SOURCE_DIR / "source_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        "Screen2Words validation passed: "
        f"{manifest['summary_rows']} summaries, {manifest['unique_screens']} screens, "
        f"{manifest['dev_screens']} dev screens, {manifest['test_screens']} test screens."
    )
    print(f"Source manifest: {manifest_path}")


if __name__ == "__main__":
    main()
