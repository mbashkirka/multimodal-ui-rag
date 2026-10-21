from __future__ import annotations

import json
import re
from pathlib import Path

_DUPLICATE_SUFFIX_RE = re.compile(r"^(?P<stem>.+) \(1\)(?P<suffix>\.[^.]+)$")


def load_legacy_mapping(mapping_path: Path | str) -> list[tuple[int, str]]:
    """Load the original FAISS-id -> filename mapping in deterministic id order."""
    path = Path(mapping_path)
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)

    ordered = sorted(((int(key), value) for key, value in raw.items()), key=lambda item: item[0])
    ids = [idx for idx, _ in ordered]
    if ids != list(range(len(ordered))):
        raise ValueError("image_index_map keys must be contiguous integers starting at 0")
    return ordered


def canonical_filename(filename: str, known_filenames: set[str]) -> str:
    """Resolve a legacy Finder-style duplicate name when the canonical RICO file exists.

    The supplied historical mapping contains ten names such as ``68471 (1).jpg`` while
    also containing ``68471.jpg``. The official RICO archive contains only the canonical
    filename. We collapse only aliases whose canonical counterpart is present in the
    same mapping, so unrelated filenames are never rewritten heuristically.
    """
    match = _DUPLICATE_SUFFIX_RE.match(filename)
    if match is None:
        return filename

    candidate = f"{match.group('stem')}{match.group('suffix')}"
    return candidate if candidate in known_filenames else filename


def canonicalize_legacy_mapping(
    ordered_mapping: list[tuple[int, str]],
) -> tuple[list[tuple[int, str]], list[dict[str, int | str]]]:
    """Collapse legacy Finder-style duplicate aliases.

    The official archive contains the canonical filename (for example ``68471.jpg``),
    while the historical mapping also contains an extra ``68471 (1).jpg`` row.
    We drop only that alias row and keep the canonical row/filename.
    """
    known_filenames = {filename for _, filename in ordered_mapping}
    id_by_filename = {filename: legacy_id for legacy_id, filename in ordered_mapping}
    canonical: list[tuple[int, str]] = []
    aliases: list[dict[str, int | str]] = []
    seen: set[str] = set()

    for legacy_id, filename in ordered_mapping:
        resolved = canonical_filename(filename, known_filenames)
        if resolved != filename:
            aliases.append(
                {
                    "legacy_id": legacy_id,
                    "legacy_filename": filename,
                    "canonical_filename": resolved,
                    "kept_legacy_id": id_by_filename[resolved],
                }
            )
            continue

        if filename in seen:
            # Defensive exact-name deduplication; the supplied mapping does not
            # currently contain such rows, but an index must never contain
            # accidental duplicate corpus entries.
            aliases.append(
                {
                    "legacy_id": legacy_id,
                    "legacy_filename": filename,
                    "canonical_filename": filename,
                    "kept_legacy_id": id_by_filename[filename],
                }
            )
            continue

        seen.add(filename)
        canonical.append((legacy_id, filename))

    return canonical, aliases


def load_canonical_mapping(
    mapping_path: Path | str,
) -> tuple[list[tuple[int, str]], list[dict[str, int | str]]]:
    return canonicalize_legacy_mapping(load_legacy_mapping(mapping_path))
