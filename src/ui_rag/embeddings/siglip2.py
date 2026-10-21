from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from ui_rag.embeddings.base import fuse_normalized


class SigLIP2Encoder:
    """Hugging Face Transformers SigLIP2 dual encoder.

    Supports both fixed-resolution and NaFlex checkpoints. Fixed-resolution
    SigLIP2 checkpoints currently load through the Transformers ``SiglipModel``
    implementation, while NaFlex checkpoints load through ``Siglip2Model``.
    In Transformers 5.15.1 both ``get_text_features`` and
    ``get_image_features`` return a ``BaseModelOutputWithPooling``-like object;
    the retrieval vector is therefore taken from ``pooler_output``.

    Text is explicitly lowercased and padded/truncated to 64 tokens to match
    the SigLIP2 training recipe even when checkpoint tokenizer metadata resolves
    to a plain Gemma tokenizer. NaFlex image preprocessing can be bounded with
    ``max_num_patches``; 256 is the controlled M3 setting.
    """

    backend = "transformers_siglip2"
    text_max_length = 64
    explicit_lowercase = True

    def __init__(
        self,
        model_name: str,
        pretrained: str = "huggingface",
        device: str | None = None,
        max_num_patches: int | None = None,
    ) -> None:
        # Lazy import keeps Transformers an M3-only dependency and avoids affecting
        # the frozen M1/M2 OpenCLIP baseline.
        from transformers import AutoModel, AutoProcessor

        self.device = device or self.resolve_device()
        self.model_name = model_name
        self.pretrained = pretrained
        self.max_num_patches = max_num_patches
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model = self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def resolve_device() -> str:
        if torch.cuda.is_available():
            return "cuda"
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
        return "cpu"

    @staticmethod
    def _normalize(embeddings: torch.Tensor) -> torch.Tensor:
        return embeddings / embeddings.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    @staticmethod
    def _move_inputs(inputs: object, device: str):
        # BatchFeature supports .to(); keeping this helper small also makes the
        # encoder robust to future processor return types that expose tensors in a
        # mapping interface.
        if hasattr(inputs, "to"):
            return inputs.to(device)
        return {key: value.to(device) for key, value in inputs.items()}

    @staticmethod
    def _pooled_tensor(output: object, *, modality: str) -> torch.Tensor:
        """Extract a 2D retrieval embedding from Transformers feature outputs.

        Transformers 5.15.1 returns ``BaseModelOutputWithPooling`` from the
        SigLIP/SigLIP2 ``get_*_features`` methods. Older/future releases may
        return tensors directly, so both contracts are accepted deliberately.
        """
        if isinstance(output, torch.Tensor):
            pooled = output
        else:
            pooled = getattr(output, "pooler_output", None)
            if (
                pooled is None
                and isinstance(output, (tuple, list))
                and len(output) >= 2
                and isinstance(output[1], torch.Tensor)
            ):
                # With return_dict=False the pooled feature is conventionally the
                # second entry: (last_hidden_state, pooler_output, ...).
                pooled = output[1]

        if not isinstance(pooled, torch.Tensor):
            raise TypeError(
                f"{modality} feature API returned {type(output).__name__} without "
                "a tensor pooler_output"
            )
        if pooled.ndim != 2:
            raise RuntimeError(
                f"{modality} pooled features must be rank-2 [batch, dim], "
                f"got shape={tuple(pooled.shape)}"
            )
        return pooled

    @torch.inference_mode()
    def encode_text(self, texts: str | Sequence[str]) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]

        # SigLIP2 was trained on lowercased text with fixed length 64. We make
        # this explicit instead of relying on checkpoint tokenizer_class metadata.
        normalized_texts = [text.lower() for text in texts]
        inputs = self.processor(
            text=normalized_texts,
            padding="max_length",
            truncation=True,
            max_length=self.text_max_length,
            return_tensors="pt",
        )
        inputs = self._move_inputs(inputs, self.device)
        output = self.model.get_text_features(**inputs)
        embeddings = self._pooled_tensor(output, modality="text")
        embeddings = self._normalize(embeddings.float())
        return np.ascontiguousarray(embeddings.cpu().numpy().astype(np.float32))

    @torch.inference_mode()
    def encode_images(
        self, image_paths: str | Path | Sequence[str | Path]
    ) -> np.ndarray:
        if isinstance(image_paths, (str, Path)):
            image_paths = [image_paths]
        images: list[Image.Image] = []
        try:
            for raw_path in image_paths:
                with Image.open(Path(raw_path)) as image:
                    images.append(image.convert("RGB").copy())

            kwargs: dict = {"images": images, "return_tensors": "pt"}
            if self.max_num_patches is not None:
                kwargs["max_num_patches"] = self.max_num_patches
            inputs = self.processor(**kwargs)
            inputs = self._move_inputs(inputs, self.device)
            output = self.model.get_image_features(**inputs)
            embeddings = self._pooled_tensor(output, modality="image")
            embeddings = self._normalize(embeddings.float())
            return np.ascontiguousarray(embeddings.cpu().numpy().astype(np.float32))
        finally:
            for image in images:
                image.close()

    def fuse(
        self,
        text_embedding: np.ndarray,
        image_embedding: np.ndarray,
        text_weight: float = 0.5,
    ) -> np.ndarray:
        return fuse_normalized(text_embedding, image_embedding, text_weight)

    def metadata(self) -> dict:
        import transformers

        return {
            "backend": self.backend,
            "model_name": self.model_name,
            "pretrained": self.pretrained,
            "max_num_patches": self.max_num_patches,
            "text_preprocessing": {
                "lowercase": self.explicit_lowercase,
                "padding": "max_length",
                "truncation": True,
                "max_length": self.text_max_length,
            },
            "feature_output": "pooler_output",
            "library": "transformers",
            "library_version": transformers.__version__,
        }
