from __future__ import annotations

import json

from fastapi.testclient import TestClient

from ui_rag.api.app import create_app
from ui_rag.api.service import ProductionService
from ui_rag.api.settings import APISettings


class ControlPlaneOnlyService:
    """Expose metadata/readiness without starting heavy persistent workers."""

    def __init__(self) -> None:
        self._service = ProductionService(APISettings.from_env())

    def readiness(self):
        return self._service.readiness()

    def model_info(self):
        return self._service.model_info()


def main() -> None:
    app = create_app(service=ControlPlaneOnlyService())
    with TestClient(app, raise_server_exceptions=False) as client:
        health = client.get("/health")
        ready = client.get("/ready")
        info = client.get("/v1/model-info")
    print(json.dumps({
        "health_status": health.status_code,
        "health": health.json(),
        "ready_status": ready.status_code,
        "ready": ready.json(),
        "model_info_status": info.status_code,
        "model_info": info.json(),
    }, indent=2))
    if health.status_code != 200:
        raise SystemExit("health smoke failed")
    if ready.status_code != 200:
        raise SystemExit("readiness smoke failed: required M5 artifacts are missing")
    if info.status_code != 200:
        raise SystemExit("model-info smoke failed")
    print("M8.1 API control-plane smoke PASSED")


if __name__ == "__main__":
    main()
