from __future__ import annotations

from pathlib import Path


def test_mlx_provider_uses_official_load_config_and_bounded_images() -> None:
    source = Path("src/ui_rag/rag/providers/mlx_vlm_provider.py").read_text(encoding="utf-8")
    assert "from mlx_vlm.utils import load_config" in source
    assert "self.config = load_config(model_name)" in source
    assert "prepare_vlm_images" in source
    assert "max_image_pixels" in source
    assert "self.model.config" not in source


def test_vlm_smoke_surfaces_child_stderr_on_native_failure() -> None:
    source = Path("scripts/m6_vlm_smoke.py").read_text(encoding="utf-8")
    assert "except subprocess.CalledProcessError" in source
    assert "child stderr" in source
    assert "returncode=" in source
