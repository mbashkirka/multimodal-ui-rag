import json

from ui_rag.config import LEGACY_IMAGE_MAP
from ui_rag.dataset.mapping import load_canonical_mapping


def test_legacy_mapping_is_contiguous() -> None:
    with LEGACY_IMAGE_MAP.open("r", encoding="utf-8") as handle:
        mapping = json.load(handle)
    keys = sorted(int(key) for key in mapping)
    assert keys == list(range(len(keys)))
    assert len(mapping) == 66_271


def test_rico_mapping_canonicalizes_ten_legacy_aliases() -> None:
    canonical, aliases = load_canonical_mapping(LEGACY_IMAGE_MAP)
    assert len(canonical) == 66_261
    assert len(aliases) == 10
    assert len({filename for _, filename in canonical}) == 66_261
    assert all(" (1)." in alias["legacy_filename"] for alias in aliases)
    assert all(" (1)." not in alias["canonical_filename"] for alias in aliases)
