from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import Any

from ui_rag.rag.image_prep import (
    QWEN25_VL_RECOMMENDED_MAX_PIXELS,
    prepare_vlm_images,
)

DEFAULT_MLX_VLM_MODEL = "mlx-community/Qwen2.5-VL-3B-Instruct-4bit"


def _stage(message: str) -> None:
    # Native Metal crashes do not produce Python exceptions. Flushed stage markers
    # make the last successful boundary visible to the parent process.
    print(f"[m6-vlm] {message}", file=sys.stderr, flush=True)


class MLXVLMEngine:
    """Reusable MLX VLM engine for one isolated process."""

    def __init__(self, model_name: str = DEFAULT_MLX_VLM_MODEL) -> None:
        _stage("stage=import_mlx_vlm start")
        try:
            from mlx_vlm import generate, load
            from mlx_vlm.prompt_utils import apply_chat_template
            from mlx_vlm.utils import load_config
        except ImportError as exc:
            raise RuntimeError("mlx-vlm is not installed; run `make m6-setup`") from exc
        _stage("stage=import_mlx_vlm done")

        self._generate = generate
        self._apply_chat_template = apply_chat_template
        self.model_name = model_name

        _stage(f"stage=load_config start model={model_name}")
        self.config = load_config(model_name)
        _stage("stage=load_config done")

        _stage(f"stage=load_model start model={model_name}")
        self.model, self.processor = load(model_name)
        _stage("stage=load_model done")

    def generate(
        self,
        *,
        prompt: str,
        image_paths: list[str],
        max_tokens: int = 300,
        temperature: float = 0.0,
        max_image_pixels: int = QWEN25_VL_RECOMMENDED_MAX_PIXELS,
    ) -> dict[str, Any]:
        with tempfile.TemporaryDirectory(prefix="ui-rag-m6-images-") as tmp:
            _stage(
                f"stage=prepare_images start count={len(image_paths)} "
                f"max_pixels={max_image_pixels}"
            )
            prepared_paths, prep_metadata = prepare_vlm_images(
                image_paths,
                output_dir=Path(tmp),
                max_pixels=max_image_pixels,
            )
            summary = ",".join(
                f"{x['prepared_width']}x{x['prepared_height']}"
                for x in prep_metadata
            )
            _stage(f"stage=prepare_images done sizes={summary}")

            _stage("stage=apply_chat_template start")
            formatted = self._apply_chat_template(
                self.processor,
                self.config,
                prompt,
                num_images=len(prepared_paths),
            )
            _stage("stage=apply_chat_template done")

            _stage(
                f"stage=generate start max_tokens={max_tokens} temperature={temperature}"
            )
            output = self._generate(
                model=self.model,
                processor=self.processor,
                prompt=formatted,
                image=prepared_paths,
                max_tokens=max_tokens,
                temperature=temperature,
                verbose=False,
            )
            _stage("stage=generate done")

        text = getattr(output, "text", None)
        if text is None:
            text = str(output)
        return {
            "provider": "mlx_vlm",
            "model": self.model_name,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "image_preprocessing": {
                "max_pixels": max_image_pixels,
                "qwen_visual_token_budget": max_image_pixels // (28 * 28),
                "items": prep_metadata,
            },
            "text": text,
        }


    def select_evidence(
        self,
        *,
        query: str,
        evidence,
        max_tokens: int = 8,
        max_image_pixels: int = QWEN25_VL_RECOMMENDED_MAX_PIXELS,
    ) -> dict[str, Any]:
        from ui_rag.rag.selector import (
            build_evidence_contact_sheet,
            build_selector_prompt,
            parse_evidence_choice,
        )

        with tempfile.TemporaryDirectory(prefix="ui-rag-m62-selector-") as tmp:
            sheet_path = Path(tmp) / "evidence_contact_sheet.jpg"
            sheet_meta = build_evidence_contact_sheet(
                evidence, output_path=sheet_path, max_pixels=max_image_pixels
            )
            prompt = build_selector_prompt(query, evidence)
            generation = self.generate(
                prompt=prompt,
                image_paths=[str(sheet_path)],
                max_tokens=max_tokens,
                temperature=0.0,
                max_image_pixels=max_image_pixels,
            )
        selected = parse_evidence_choice(
            generation["text"],
            allowed_evidence_ids={x.evidence_id for x in evidence},
        )
        return {
            "selected_evidence_id": None if selected == "NONE" else selected,
            "abstain": selected == "NONE",
            "raw_text": generation["text"],
            "contact_sheet": sheet_meta,
            "generation": {k: v for k, v in generation.items() if k != "text"},
        }

    def answer_selected_evidence(
        self,
        *,
        query: str,
        evidence_item,
        max_tokens: int = 96,
        max_image_pixels: int = QWEN25_VL_RECOMMENDED_MAX_PIXELS,
    ) -> dict[str, Any]:
        from ui_rag.rag.selector import build_selected_answer_prompt

        prompt = build_selected_answer_prompt(query, evidence_item.evidence_id)
        generation = self.generate(
            prompt=prompt,
            image_paths=[evidence_item.path],
            max_tokens=max_tokens,
            temperature=0.0,
            max_image_pixels=max_image_pixels,
        )
        text = generation["text"].strip()
        if not text:
            raise RuntimeError("VLM returned an empty grounded answer")
        return {
            "text": text,
            # Kept for compatibility with historical M6.3 tooling. M6.4 production
            # deliberately does not use the VLM as an evidence acceptance gate.
            "insufficient": False,
            "generation": {k: v for k, v in generation.items() if k != "text"},
        }


def generate_mlx_vlm(
    *,
    prompt: str,
    image_paths: list[str],
    model_name: str = DEFAULT_MLX_VLM_MODEL,
    max_tokens: int = 300,
    temperature: float = 0.0,
    max_image_pixels: int = QWEN25_VL_RECOMMENDED_MAX_PIXELS,
) -> dict[str, Any]:
    """One-shot helper used by the CLI child process."""
    return MLXVLMEngine(model_name).generate(
        prompt=prompt,
        image_paths=image_paths,
        max_tokens=max_tokens,
        temperature=temperature,
        max_image_pixels=max_image_pixels,
    )
