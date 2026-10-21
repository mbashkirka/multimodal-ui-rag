from ui_rag.dataset.mapping import canonicalize_legacy_mapping


def test_only_known_finder_aliases_are_collapsed() -> None:
    raw = [
        (0, "10 (1).jpg"),
        (1, "10.jpg"),
        (2, "screen (1).jpg"),
        (3, "11.jpg"),
    ]
    canonical, aliases = canonicalize_legacy_mapping(raw)
    assert canonical == [
        (1, "10.jpg"),
        (2, "screen (1).jpg"),
        (3, "11.jpg"),
    ]
    assert aliases == [
        {
            "legacy_id": 0,
            "legacy_filename": "10 (1).jpg",
            "canonical_filename": "10.jpg",
            "kept_legacy_id": 1,
        }
    ]
