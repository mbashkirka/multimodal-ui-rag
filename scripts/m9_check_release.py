from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from ui_rag import __version__
from ui_rag.api.app import create_app


class _ContractService:
    def readiness(self):
        return {"status": "ready", "api_version": "v1", "artifacts": {}, "workers": {}}

    def model_info(self):
        return {
            "service_version": __version__, "api_version": "v1", "retrieval_encoder": "siglip",
            "retrieval_index": "flatip", "reranker": "rrf", "vlm_model": "qwen",
            "artifact_fingerprint": None, "native_isolation": True,
            "serving_mode": "persistent-isolated-workers", "evidence_policy": "rank1",
            "max_concurrent_inference": 1,
            "worker_timeouts_seconds": {"encoder": 30.0, "faiss": 10.0, "vlm": 60.0},
        }

    def metrics_text(self):
        return "ui_rag_inference_active 0\n"


root = Path(__file__).resolve().parents[1]
pyproject = (root / "pyproject.toml").read_text(encoding="utf-8")
if f'version = "{__version__}"' not in pyproject:
    raise SystemExit("version mismatch between pyproject.toml and ui_rag.__version__")

schema = create_app(service=_ContractService()).openapi()
for route in ("/health", "/ready", "/v1/model-info", "/v1/search", "/v1/rag"):
    if route not in schema["paths"]:
        raise SystemExit(f"missing OpenAPI route: {route}")
if "/metrics" in schema["paths"]:
    raise SystemExit("/metrics must stay out of public OpenAPI schema")

code = (
    "import sys; import ui_rag.api.app; import ui_rag.api.service; import ui_rag.api.workers; "
    "print(','.join(x for x in ['torch','faiss','mlx','mlx_vlm'] if x in sys.modules))"
)
env = dict(os.environ)
env["PYTHONPATH"] = str(root / "src")
out = subprocess.check_output([sys.executable, "-c", code], text=True, env=env).strip()
if out:
    raise SystemExit(f"native-stack import leak in API host: {out}")

print(json.dumps({
    "milestone": "M9",
    "version": __version__,
    "openapi_routes": sorted(schema["paths"]),
    "metrics_hidden_from_openapi": True,
    "native_import_isolation": True,
}, indent=2))
print("M9 release-contract smoke PASSED")
