from __future__ import annotations

import math
import re
from collections.abc import Iterable
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

from ui_rag.rag.schema import EvidenceItem

DEFAULT_SELECTOR_MAX_PIXELS = 1280 * 28 * 28
_SELECTOR_RE = re.compile(r"\b(E[1-9][0-9]*|NONE)\b", re.IGNORECASE)


def build_selector_prompt(query: str, evidence: list[EvidenceItem]) -> str:
    allowed = ", ".join([x.evidence_id for x in evidence] + ["NONE"])
    return f"""You are a visual UI retrieval verifier.

User description:
{query}

The single contact-sheet image contains {len(evidence)} candidate mobile UI screenshots. Each candidate has a visible label matching its evidence ID.
Choose the ONE screenshot that most directly matches the user description.
If none of the candidates sufficiently matches, choose NONE.

Reply with exactly one label from: {allowed}
Do not explain. Do not add punctuation. Do not output JSON.
"""


def build_selected_answer_prompt(query: str, evidence_id: str) -> str:
    return f"""You are a grounded UI analyst.

User request:
{query}

You are shown exactly one retrieved screenshot, evidence {evidence_id}.
The retrieval/reranking system has already selected this screenshot as the evidence.
Your job is NOT to accept/reject or rerank it. Your job is to describe what is visibly
supported by the screenshot and relate those visible details to the user request.

Be concise and concrete. Do not invent controls, labels, states, colors, navigation paths,
or functionality that are not visible. If the requested property is not clearly visible,
say that explicitly while still describing the relevant visible evidence.
Do not emit special refusal/sentinel tokens. Do not mention internal retrieval scores or filenames.
"""


def parse_evidence_choice(text: str, *, allowed_evidence_ids: set[str]) -> str:
    raw = text.strip().upper()
    if raw == "NONE":
        return "NONE"
    if raw in allowed_evidence_ids:
        return raw

    matches = [m.upper() for m in _SELECTOR_RE.findall(raw)]
    valid = [m for m in matches if m == "NONE" or m in allowed_evidence_ids]
    unique = []
    for item in valid:
        if item not in unique:
            unique.append(item)
    if len(unique) == 1:
        return unique[0]
    if not unique:
        raise ValueError(f"selector output contains no allowed evidence label: {text!r}")
    raise ValueError(f"selector output is ambiguous: {unique}")


def _fit_size(width: int, height: int, box_w: int, box_h: int) -> tuple[int, int]:
    scale = min(box_w / width, box_h / height)
    return max(1, int(width * scale)), max(1, int(height * scale))


def _sheet_geometry(num_items: int, max_pixels: int) -> tuple[int, int, int, int, int]:
    if num_items <= 0:
        raise ValueError("num_items must be positive")
    columns = 2 if num_items > 1 else 1
    rows = math.ceil(num_items / columns)
    gap = 12
    # Phone screenshots are roughly 9:16. A compact label band sits above each cell.
    low, high = 96, 900
    best = low
    while low <= high:
        cell_w = (low + high) // 2
        label_h = max(36, int(cell_w * 0.12))
        content_h = int(cell_w * 16 / 9)
        sheet_w = columns * cell_w + (columns + 1) * gap
        sheet_h = rows * (label_h + content_h) + (rows + 1) * gap
        if sheet_w * sheet_h <= max_pixels:
            best = cell_w
            low = cell_w + 1
        else:
            high = cell_w - 1
    cell_w = best
    label_h = max(36, int(cell_w * 0.12))
    content_h = int(cell_w * 16 / 9)
    sheet_w = columns * cell_w + (columns + 1) * gap
    sheet_h = rows * (label_h + content_h) + (rows + 1) * gap
    return columns, rows, cell_w, label_h, content_h


def build_evidence_contact_sheet(
    evidence: Iterable[EvidenceItem],
    *,
    output_path: Path,
    max_pixels: int = DEFAULT_SELECTOR_MAX_PIXELS,
) -> dict:
    items = list(evidence)
    if not items:
        raise ValueError("at least one evidence item is required")
    columns, rows, cell_w, label_h, content_h = _sheet_geometry(len(items), max_pixels)
    gap = 12
    sheet_w = columns * cell_w + (columns + 1) * gap
    sheet_h = rows * (label_h + content_h) + (rows + 1) * gap
    sheet = Image.new("RGB", (sheet_w, sheet_h), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=max(24, int(label_h * 0.62)))
    item_meta: list[dict] = []

    for idx, item in enumerate(items):
        row, col = divmod(idx, columns)
        x0 = gap + col * (cell_w + gap)
        y0 = gap + row * (label_h + content_h + gap)
        draw.rectangle((x0, y0, x0 + cell_w, y0 + label_h), fill="black")
        text_box = draw.textbbox((0, 0), item.evidence_id, font=font)
        tw = text_box[2] - text_box[0]
        th = text_box[3] - text_box[1]
        draw.text(
            (x0 + (cell_w - tw) / 2, y0 + (label_h - th) / 2 - text_box[1]),
            item.evidence_id,
            fill="white",
            font=font,
        )
        source = Path(item.path)
        with Image.open(source) as raw:
            image = ImageOps.exif_transpose(raw).convert("RGB")
            original_w, original_h = image.size
            fit_w, fit_h = _fit_size(original_w, original_h, cell_w, content_h)
            image = image.resize((fit_w, fit_h), Image.Resampling.LANCZOS)
        paste_x = x0 + (cell_w - fit_w) // 2
        paste_y = y0 + label_h + (content_h - fit_h) // 2
        sheet.paste(image, (paste_x, paste_y))
        item_meta.append(
            {
                "evidence_id": item.evidence_id,
                "source": str(source),
                "original_width": original_w,
                "original_height": original_h,
                "rendered_width": fit_w,
                "rendered_height": fit_h,
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, format="JPEG", quality=92, optimize=True)
    return {
        "path": str(output_path),
        "width": sheet_w,
        "height": sheet_h,
        "pixels": sheet_w * sheet_h,
        "max_pixels": max_pixels,
        "columns": columns,
        "rows": rows,
        "items": item_meta,
    }
