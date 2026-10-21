from __future__ import annotations

from pathlib import Path

from PIL import Image

from ui_rag.rag.image_prep import (
    QWEN25_VL_RECOMMENDED_MAX_PIXELS,
    _bounded_size,
    prepare_vlm_images,
)


def test_bounded_size_caps_large_mobile_screenshot() -> None:
    width, height = _bounded_size(
        1440,
        2560,
        max_pixels=QWEN25_VL_RECOMMENDED_MAX_PIXELS,
    )
    assert width * height <= QWEN25_VL_RECOMMENDED_MAX_PIXELS
    assert width % 28 == 0
    assert height % 28 == 0
    assert width < 1440
    assert height < 2560
    original_ratio = 1440 / 2560
    resized_ratio = width / height
    assert abs(original_ratio - resized_ratio) < 0.03


def test_prepare_vlm_images_resizes_oversized_image(tmp_path: Path) -> None:
    source = tmp_path / "screen.jpg"
    Image.new("RGB", (1440, 2560), "white").save(source)
    out = tmp_path / "prepared"
    paths, metadata = prepare_vlm_images([source], output_dir=out)
    assert len(paths) == 1
    assert metadata[0]["resized"] is True
    assert metadata[0]["prepared_pixels"] <= QWEN25_VL_RECOMMENDED_MAX_PIXELS
    assert Path(paths[0]).exists()


def test_prepare_vlm_images_does_not_upscale_small_image(tmp_path: Path) -> None:
    source = tmp_path / "small.jpg"
    Image.new("RGB", (280, 420), "white").save(source)
    paths, metadata = prepare_vlm_images([source], output_dir=tmp_path / "prepared")
    assert paths == [str(source)]
    assert metadata[0]["resized"] is False
    assert metadata[0]["prepared_width"] == 280
    assert metadata[0]["prepared_height"] == 420
