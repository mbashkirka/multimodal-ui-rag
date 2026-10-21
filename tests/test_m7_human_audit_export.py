from __future__ import annotations

import importlib.util
import json
import threading
import urllib.request
from pathlib import Path


def _load_server_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "m7_human_audit_server.py"
    spec = importlib.util.spec_from_file_location("m7_human_audit_server", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _csv_12() -> str:
    header = "audit_id,human_faithfulness_0_2,human_relevance_0_2,human_unsupported_0_2,human_completeness_0_2,human_notes"
    rows = [header]
    for i in range(1, 13):
        rows.append(f'A{i:02d},2,2,0,2,"ok"')
    return "\n".join(rows) + "\n"


def test_human_audit_server_saves_validated_csv(tmp_path: Path) -> None:
    mod = _load_server_module()
    (tmp_path / "audit.html").write_text("ok", encoding="utf-8")

    def handler(*args, **kwargs):
        return mod.AuditHandler(*args, directory=str(tmp_path), **kwargs)

    server = mod.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{server.server_address[1]}/save"
        payload = json.dumps({"csv": _csv_12()}).encode("utf-8")
        req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=5) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        assert body["ok"] is True
        out = tmp_path / "human_audit_completed.csv"
        assert out.exists()
        assert "A12,2,2,0,2" in out.read_text(encoding="utf-8")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_audit_builder_uses_http_save_endpoint() -> None:
    text = (Path(__file__).resolve().parents[1] / "scripts" / "m7_build_human_audit.py").read_text(encoding="utf-8")
    assert "fetch('/save'" in text
    assert "Copy CSV fallback" in text
    assert "make m7-human-audit-serve" in text
