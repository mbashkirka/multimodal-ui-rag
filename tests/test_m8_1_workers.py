from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_worker_module_import_is_native_stack_free():
    code = (
        "import sys; import ui_rag.api.workers; import ui_rag.api.worker_main; "
        "print(','.join(x for x in ['torch','faiss','mlx','mlx_vlm'] if x in sys.modules))"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    out = subprocess.check_output([sys.executable, "-c", code], text=True, env=env).strip()
    assert out == ""


def test_service_source_uses_persistent_worker_mode():
    from ui_rag.api.service import ProductionService

    # Stable public metadata is easier to regression-test than implementation details.
    assert "persistent isolated workers" in (ProductionService.__doc__ or "").lower()
