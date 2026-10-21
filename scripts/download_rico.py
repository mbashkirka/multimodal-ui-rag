from __future__ import annotations

import argparse
import shutil
import tarfile
import urllib.request
from pathlib import Path

from tqdm import tqdm

from ui_rag.dataset.mapping import load_canonical_mapping, load_legacy_mapping

RICO_URL = (
    "https://storage.googleapis.com/crowdstf-rico-uiuc-4540/"
    "rico_dataset_v0.1/unique_uis.tar.gz"
)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data_storage"
DEFAULT_ARCHIVE = DATA_DIR / "unique_uis.tar.gz"
DEFAULT_TARGET = DATA_DIR / "RICO_dataset"
DEFAULT_MAPPING = DATA_DIR / "image_index_map.json"


class TqdmUpTo(tqdm):
    def update_to(
        self,
        blocks: int = 1,
        block_size: int = 1,
        total_size: int | None = None,
    ) -> None:
        if total_size is not None:
            self.total = total_size
        self.update(blocks * block_size - self.n)


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and destination.stat().st_size > 0:
        print(f"Archive already exists: {destination}")
        return

    tmp_path = destination.with_suffix(destination.suffix + ".part")
    with TqdmUpTo(
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        miniters=1,
        desc="Downloading RICO",
    ) as bar:
        urllib.request.urlretrieve(url, filename=tmp_path, reporthook=bar.update_to)
    tmp_path.replace(destination)


def extract_jpegs(archive: Path, target: Path) -> int:
    target.mkdir(parents=True, exist_ok=True)
    extracted = 0

    # Stream through the gzip archive once. This avoids expensive random seeks
    # across a large compressed tarball and works well on macOS.
    with tarfile.open(archive, mode="r|gz") as tar:
        for member in tqdm(tar, desc="Extracting screenshots", unit="files"):
            if not member.isfile() or not member.name.lower().endswith((".jpg", ".jpeg")):
                continue

            out_path = target / Path(member.name).name
            if out_path.is_file() and out_path.stat().st_size > 0:
                continue

            source = tar.extractfile(member)
            if source is None:
                continue

            with source, out_path.open("wb") as out:
                shutil.copyfileobj(source, out)
            extracted += 1

    return extracted


def validate(target: Path, mapping_path: Path) -> tuple[int, int, list[str], int]:
    legacy = load_legacy_mapping(mapping_path)
    canonical, aliases = load_canonical_mapping(mapping_path)
    expected_paths = [target / filename for _, filename in canonical]
    missing = [path.name for path in expected_paths if not path.is_file()]
    return len(legacy), len(canonical), missing, len(aliases)


def print_validation(
    *,
    legacy_count: int,
    canonical_count: int,
    missing: list[str],
    alias_count: int,
) -> None:
    present = canonical_count - len(missing)
    print(f"Legacy mapping rows: {legacy_count}")
    print(f"Canonical unique screenshots: {canonical_count}")
    print(f"Legacy duplicate/alias records collapsed: {alias_count}")
    print(f"Canonical screenshots present: {present}/{canonical_count}")

    if missing:
        print("First missing canonical files:")
        for filename in missing[:20]:
            print(f"  - {filename}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=RICO_URL)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--keep-archive", action="store_true")
    args = parser.parse_args()

    legacy_count, canonical_count, missing, alias_count = validate(args.target, args.mapping)

    # Important for the macOS workflow: if the 6 GB archive was already extracted
    # successfully, do not download or rescan it just because the historical mapping
    # contains duplicate Finder-style filenames.
    if not missing:
        print_validation(
            legacy_count=legacy_count,
            canonical_count=canonical_count,
            missing=missing,
            alias_count=alias_count,
        )
        print("RICO validation passed; download/extraction not needed.")
        if not args.validate_only and not args.keep_archive:
            args.archive.unlink(missing_ok=True)
        return

    if args.validate_only:
        print_validation(
            legacy_count=legacy_count,
            canonical_count=canonical_count,
            missing=missing,
            alias_count=alias_count,
        )
        raise SystemExit(2)

    download(args.url, args.archive)
    print(f"New JPEG files extracted: {extract_jpegs(args.archive, args.target)}")

    legacy_count, canonical_count, missing, alias_count = validate(args.target, args.mapping)
    print_validation(
        legacy_count=legacy_count,
        canonical_count=canonical_count,
        missing=missing,
        alias_count=alias_count,
    )

    if missing:
        raise SystemExit(2)

    if not args.keep_archive:
        args.archive.unlink(missing_ok=True)
    print("RICO validation passed.")


if __name__ == "__main__":
    main()
