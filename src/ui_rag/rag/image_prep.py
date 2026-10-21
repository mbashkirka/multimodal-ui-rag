from __future__ import annotations

import math
from collections.abc import Iterable
from pathlib import Path

from PIL import Image, ImageOps

# Qwen2.5-VL's own model card recommends a typical 256-1280 visual-token
# range. One merged visual token corresponds to 28x28 input pixels.
QWEN25_VL_RECOMMENDED_MAX_PIXELS = 1280 * 28 * 28  # 1,003,520
QWEN25_VL_SIZE_MULTIPLE = 28


def _bounded_size(
    width: int,
    height: int,
    *,
    max_pixels: int,
    multiple: int = QWEN25_VL_SIZE_MULTIPLE,
) -> tuple[int, int]:
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    if max_pixels <= 0:
        raise ValueError("max_pixels must be positive")
    if multiple <= 0:
        raise ValueError("multiple must be positive")
    if width * height <= max_pixels:
        return width, height

    scale = math.sqrt(max_pixels / float(width * height))
    scaled_w = max(multiple, int(width * scale))
    scaled_h = max(multiple, int(height * scale))
    # Qwen's vision stack operates naturally on 28-pixel merged patches. Round
    # down so the cap is never exceeded and aspect-ratio distortion stays small.
    new_w = max(multiple, (scaled_w // multiple) * multiple)
    new_h = max(multiple, (scaled_h // multiple) * multiple)

    while new_w * new_h > max_pixels:
        if new_w / width >= new_h / height and new_w > multiple:
            new_w -= multiple
        elif new_h > multiple:
            new_h -= multiple
        else:
            break
    return new_w, new_h


def prepare_vlm_images(
    image_paths: Iterable[str | Path],
    *,
    output_dir: Path,
    max_pixels: int = QWEN25_VL_RECOMMENDED_MAX_PIXELS,
) -> tuple[list[str], list[dict]]:
    """Bound VLM image resolution while preserving the source evidence.

    Small images are passed through unchanged. Oversized images are EXIF-corrected,
    resized with LANCZOS while preserving aspect ratio, and written into a temporary
    directory. The returned metadata makes the preprocessing explicit/auditable.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    prepared: list[str] = []
    metadata: list[dict] = []

    for index, raw_path in enumerate(image_paths, start=1):
        source = Path(raw_path)
        if not source.exists():
            raise FileNotFoundError(source)
        with Image.open(source) as raw:
            image = ImageOps.exif_transpose(raw)
            width, height = image.size
            new_w, new_h = _bounded_size(width, height, max_pixels=max_pixels)
            resized = (new_w, new_h) != (width, height)
            if resized:
                image = image.convert("RGB").resize((new_w, new_h), Image.Resampling.LANCZOS)
                destination = output_dir / f"evidence_{index:02d}_{new_w}x{new_h}.jpg"
                image.save(destination, format="JPEG", quality=92, optimize=True)
                prepared_path = destination
            else:
                prepared_path = source

        prepared.append(str(prepared_path))
        metadata.append(
            {
                "source": str(source),
                "prepared": str(prepared_path),
                "original_width": width,
                "original_height": height,
                "prepared_width": new_w,
                "prepared_height": new_h,
                "original_pixels": width * height,
                "prepared_pixels": new_w * new_h,
                "max_pixels": max_pixels,
                "resized": resized,
            }
        )
    return prepared, metadata
