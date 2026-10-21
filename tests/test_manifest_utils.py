from pathlib import Path

from ui_rag.indexing.manifest import read_json, sha256_file, write_json


def test_manifest_helpers_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    payload = {"model": "ViT-B-32-quickgelu", "normalized": True}
    write_json(path, payload)
    assert read_json(path) == payload
    assert len(sha256_file(path)) == 64
