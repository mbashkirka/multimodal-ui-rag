from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import open_clip
import torch
from PIL import Image

from ui_rag.embeddings.base import fuse_normalized


class CLIPEncoder:
    backend = "open_clip"
    """OpenCLIP ViT encoder for normalized text/image retrieval embeddings."""

    def __init__(
        self,
        model_name: str = "ViT-B-32-quickgelu",
        pretrained: str = "openai",
        device: str | None = None,
    ) -> None:
        self.device = device or self.resolve_device()
        self.model_name = model_name
        self.pretrained = pretrained
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
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
    def normalize(embeddings: torch.Tensor) -> torch.Tensor:
        return embeddings / embeddings.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    @torch.inference_mode()
    def encode_text(self, texts: str | Sequence[str]) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        tokens = self.tokenizer(list(texts)).to(self.device)
        embeddings = self.normalize(self.model.encode_text(tokens).float())
        return np.ascontiguousarray(embeddings.cpu().numpy().astype(np.float32))

    @torch.inference_mode()
    def encode_images(self, image_paths: str | Path | Sequence[str | Path]) -> np.ndarray:
        if isinstance(image_paths, (str, Path)):
            image_paths = [image_paths]
        tensors: list[torch.Tensor] = []
        for raw_path in image_paths:
            with Image.open(Path(raw_path)) as image:
                tensors.append(self.preprocess(image.convert("RGB")))
        batch = torch.stack(tensors).to(self.device)
        embeddings = self.normalize(self.model.encode_image(batch).float())
        return np.ascontiguousarray(embeddings.cpu().numpy().astype(np.float32))

    def fuse(
        self,
        text_embedding: np.ndarray,
        image_embedding: np.ndarray,
        text_weight: float = 0.5,
    ) -> np.ndarray:
        return fuse_normalized(text_embedding, image_embedding, text_weight)

    def metadata(self) -> dict:
        return {
            "backend": self.backend,
            "model_name": self.model_name,
            "pretrained": self.pretrained,
            "max_num_patches": None,
            "library": "open_clip",
            "library_version": getattr(open_clip, "__version__", "unknown"),
        }
