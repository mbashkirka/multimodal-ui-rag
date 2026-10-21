from __future__ import annotations

import csv
import json
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

from ui_rag.dataset.mapping import load_canonical_mapping


@dataclass(frozen=True)
class BenchmarkQuery:
    query_id: str
    screen_id: str
    query: str
    target_filename: str
    summary_index: int


def _screen_id_from_filename(filename: str) -> str:
    return Path(filename).stem


def load_screen2words_summaries(path: Path) -> dict[str, list[str]]:
    summaries: dict[str, list[str]] = defaultdict(list)
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["screenId", "summary"]:
            raise ValueError(f"unexpected Screen2Words schema: {reader.fieldnames}")
        for row in reader:
            screen_id = row["screenId"].strip()
            summary = " ".join(row["summary"].split())
            if screen_id and summary:
                summaries[screen_id].append(summary)
    return dict(summaries)


def load_split_screen_ids(path: Path, *, split_name: str = "split") -> list[str]:
    screen_ids = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    screen_ids = [screen_id for screen_id in screen_ids if screen_id]
    if len(screen_ids) != len(set(screen_ids)):
        raise ValueError(f"Screen2Words {split_name} split contains duplicate screen ids")
    return screen_ids


def load_test_screen_ids(path: Path) -> list[str]:
    """Backward-compatible alias retained for M2/M3 callers and tests."""
    return load_split_screen_ids(path, split_name="test")


def prepare_screen2words_queries(
    *,
    summaries_path: Path,
    test_screens_path: Path,
    legacy_mapping_path: Path,
    variant: str,
    limit_screens: int | None = None,
    split_name: str = "test",
) -> tuple[list[BenchmarkQuery], dict]:
    if variant not in {"primary", "all"}:
        raise ValueError("variant must be 'primary' or 'all'")
    if limit_screens is not None and limit_screens <= 0:
        raise ValueError("limit_screens must be positive")

    summaries = load_screen2words_summaries(summaries_path)
    split_screen_ids = load_split_screen_ids(test_screens_path, split_name=split_name)
    if limit_screens is not None:
        split_screen_ids = split_screen_ids[:limit_screens]

    canonical_mapping, aliases = load_canonical_mapping(legacy_mapping_path)
    filename_by_screen_id = {
        _screen_id_from_filename(filename): filename for _, filename in canonical_mapping
    }

    missing_corpus = [sid for sid in split_screen_ids if sid not in filename_by_screen_id]
    missing_summaries = [sid for sid in split_screen_ids if sid not in summaries]
    if missing_corpus:
        raise RuntimeError(f"{len(missing_corpus)} {split_name} screens are absent from RICO corpus")
    if missing_summaries:
        raise RuntimeError(f"{len(missing_summaries)} {split_name} screens have no summaries")

    queries: list[BenchmarkQuery] = []
    for screen_id in split_screen_ids:
        screen_summaries = summaries[screen_id]
        if not screen_summaries:
            raise RuntimeError(f"screen {screen_id} has zero summaries")
        chosen = screen_summaries[:1] if variant == "primary" else screen_summaries
        for summary_index, query in enumerate(chosen):
            queries.append(
                BenchmarkQuery(
                    query_id=f"s2w-{split_name}-{screen_id}-{summary_index}",
                    screen_id=screen_id,
                    query=query,
                    target_filename=filename_by_screen_id[screen_id],
                    summary_index=summary_index,
                )
            )

    metadata = {
        "dataset": "Screen2Words",
        "split": split_name,
        "variant": variant,
        "num_screens": len(split_screen_ids),
        "num_queries": len(queries),
        "summaries_per_screen_expected": 1 if variant == "primary" else 5,
        "legacy_alias_records_in_rico_mapping": len(aliases),
        "all_targets_present_in_canonical_rico": True,
    }
    return queries, metadata


def write_queries_jsonl(path: Path, queries: Iterable[BenchmarkQuery]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for query in queries:
            handle.write(json.dumps(asdict(query), ensure_ascii=False, sort_keys=True) + "\n")


def read_queries_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            required = {"query_id", "screen_id", "query", "target_filename", "summary_index"}
            if not required.issubset(record):
                raise ValueError(f"missing fields in {path}:{line_number}")
            records.append(record)
    return records
