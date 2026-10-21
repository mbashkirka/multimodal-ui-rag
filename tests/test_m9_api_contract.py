from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from fastapi.testclient import TestClient

from ui_rag.api.app import create_app


class FakeOperationalService:
    def readiness(self):
        return {
            "status": "ready",
            "api_version": "v1",
            "artifacts": {"base_manifest": {"exists": True}},
            "workers": {"encoder": {"alive": True, "restarts": 1, "timeouts": 0}},
        }

    def model_info(self):
        return {
            "service_version": "0.9.0",
            "api_version": "v1",
            "retrieval_encoder": "siglip",
            "retrieval_index": "flatip",
            "reranker": "rrf",
            "vlm_model": "qwen",
            "artifact_fingerprint": "abc",
            "native_isolation": True,
            "serving_mode": "persistent-isolated-workers",
            "evidence_policy": "rank1",
            "max_concurrent_inference": 1,
            "worker_timeouts_seconds": {"encoder": 30.0, "faiss": 10.0, "vlm": 60.0},
        }

    def metrics_text(self):
        return "# TYPE ui_rag_inference_active gauge\nui_rag_inference_active 0\n"

    def record_http_request(self, **_kwargs):
        return None


def test_metrics_endpoint_is_prometheus_text():
    client = TestClient(create_app(service=FakeOperationalService()))
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "ui_rag_inference_active 0" in response.text
    assert response.headers["content-type"].startswith("text/plain")


def test_readiness_exposes_recovery_counters():
    client = TestClient(create_app(service=FakeOperationalService()))
    worker = client.get("/ready").json()["workers"]["encoder"]
    assert worker == {"alive": True, "restarts": 1, "timeouts": 0}


def test_m9_api_host_import_remains_native_stack_free():
    code = (
        "import sys; import ui_rag.api.app; import ui_rag.api.metrics; "
        "print(','.join(x for x in ['torch','faiss','mlx','mlx_vlm'] if x in sys.modules))"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    out = subprocess.check_output([sys.executable, "-c", code], text=True, env=env).strip()
    assert out == ""


def test_oversized_request_id_is_replaced():
    client = TestClient(create_app(service=FakeOperationalService()))
    response = client.get("/health", headers={"X-Request-ID": "x" * 500})
    assert response.status_code == 200
    assert response.headers["X-Request-ID"] != "x" * 500
    assert len(response.headers["X-Request-ID"]) <= 128
