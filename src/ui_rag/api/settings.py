from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from ui_rag.config import PROJECT_ROOT


def _env_path(name: str, default: str) -> Path:
    raw = os.environ.get(name, default)
    path = Path(raw).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    value = int(raw)
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    value = default if raw is None else float(raw)
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


@dataclass(frozen=True)
class APISettings:
    base_manifest: Path
    rerank_stage: Path
    model_name: str
    max_image_pixels: int
    candidate_k: int
    evidence_k: int
    max_concurrent_inference: int
    acquire_timeout_seconds: float
    encoder_timeout_seconds: float
    faiss_timeout_seconds: float
    vlm_timeout_seconds: float
    worker_startup_timeout_seconds: float

    @classmethod
    def from_env(cls) -> APISettings:
        return cls(
            base_manifest=_env_path(
                "UI_RAG_BASE_MANIFEST",
                "artifacts/metadata/siglip2-base-p16-naflex-p256-manifest.json",
            ),
            rerank_stage=_env_path(
                "UI_RAG_RERANK_STAGE",
                "artifacts/metadata/siglip2-base-p16-naflex-p512-rerank-embedding-stage.json",
            ),
            model_name=os.environ.get(
                "UI_RAG_VLM_MODEL", "mlx-community/Qwen2.5-VL-3B-Instruct-4bit"
            ),
            max_image_pixels=_env_int("UI_RAG_MAX_IMAGE_PIXELS", 1_003_520),
            candidate_k=_env_int("UI_RAG_CANDIDATE_K", 20),
            evidence_k=_env_int("UI_RAG_EVIDENCE_K", 4),
            max_concurrent_inference=_env_int("UI_RAG_MAX_CONCURRENT_INFERENCE", 1),
            acquire_timeout_seconds=_env_float("UI_RAG_ACQUIRE_TIMEOUT_SECONDS", 0.05),
            encoder_timeout_seconds=_env_float("UI_RAG_ENCODER_TIMEOUT_SECONDS", 30.0),
            faiss_timeout_seconds=_env_float("UI_RAG_FAISS_TIMEOUT_SECONDS", 10.0),
            vlm_timeout_seconds=_env_float("UI_RAG_VLM_TIMEOUT_SECONDS", 60.0),
            worker_startup_timeout_seconds=_env_float(
                "UI_RAG_WORKER_STARTUP_TIMEOUT_SECONDS", 300.0
            ),
        )
