"""Subprocess-isolated index builder compatibility wrapper."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from ui_rag.config import (
    DEFAULT_ARTIFACT_NAME,
    DEFAULT_MODEL_NAME,
    DEFAULT_PRETRAINED,
    LEGACY_IMAGE_MAP,
    METADATA_DIR,
    RICO_DIR,
)
from ui_rag.indexing.manifest import read_json


def build_index(
    *,
    image_dir: Path = RICO_DIR,
    mapping_path: Path = LEGACY_IMAGE_MAP,
    artifact_name: str = DEFAULT_ARTIFACT_NAME,
    model_name: str = DEFAULT_MODEL_NAME,
    pretrained: str = DEFAULT_PRETRAINED,
    encoder_backend: str | None = None,
    max_num_patches: int | None = None,
    batch_size: int = 16,
    device: str | None = None,
    limit: int | None = None,
    restart: bool = False,
    **_: object,
) -> dict:
    command = [
        sys.executable,
        "-m",
        "ui_rag",
        "index",
        "--image-dir",
        str(image_dir),
        "--mapping",
        str(mapping_path),
        "--artifact-name",
        artifact_name,
        "--model-name",
        model_name,
        "--pretrained",
        pretrained,
        "--batch-size",
        str(batch_size),
    ]
    if encoder_backend:
        command += ["--encoder-backend", encoder_backend]
    if max_num_patches is not None:
        command += ["--max-num-patches", str(max_num_patches)]
    if limit is not None:
        command += ["--limit", str(limit)]
    if device:
        command += ["--device", device]
    if restart:
        command += ["--restart"]
    subprocess.run(command, check=True)
    return read_json(METADATA_DIR / f"{artifact_name}-manifest.json")
