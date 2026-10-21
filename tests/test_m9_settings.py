from __future__ import annotations

import pytest

from ui_rag.api.settings import APISettings


def test_worker_timeout_defaults_are_positive(monkeypatch):
    for name in (
        "UI_RAG_ENCODER_TIMEOUT_SECONDS",
        "UI_RAG_FAISS_TIMEOUT_SECONDS",
        "UI_RAG_VLM_TIMEOUT_SECONDS",
        "UI_RAG_WORKER_STARTUP_TIMEOUT_SECONDS",
    ):
        monkeypatch.delenv(name, raising=False)
    settings = APISettings.from_env()
    assert settings.encoder_timeout_seconds == 30.0
    assert settings.faiss_timeout_seconds == 10.0
    assert settings.vlm_timeout_seconds == 60.0
    assert settings.worker_startup_timeout_seconds == 300.0


def test_nonpositive_worker_timeout_is_rejected(monkeypatch):
    monkeypatch.setenv("UI_RAG_VLM_TIMEOUT_SECONDS", "0")
    with pytest.raises(ValueError, match="UI_RAG_VLM_TIMEOUT_SECONDS must be positive"):
        APISettings.from_env()
