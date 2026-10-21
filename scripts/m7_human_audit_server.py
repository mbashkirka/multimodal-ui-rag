from __future__ import annotations

import argparse
import json
import os
import tempfile
import threading
import webbrowser
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from ui_rag.generation_eval.human_audit import parse_human_scores_csv


class AuditHandler(SimpleHTTPRequestHandler):
    server_version = "UIRAGHumanAudit/0.7.2.1"

    def __init__(self, *args, directory: str | None = None, **kwargs):
        super().__init__(*args, directory=directory, **kwargs)

    @property
    def audit_dir(self) -> Path:
        return Path(self.directory).resolve()

    def _json(self, status: int, payload: dict) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self) -> None:
        if self.path != "/save":
            self._json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "not found"})
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": "invalid content length"})
            return
        if length <= 0 or length > 512_000:
            self._json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": "invalid payload size"})
            return

        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            csv_text = payload.get("csv", "")
            if not isinstance(csv_text, str) or not csv_text.strip():
                raise ValueError("missing CSV payload")

            out = self.audit_dir / "human_audit_completed.csv"
            fd, tmp_name = tempfile.mkstemp(prefix="human-audit-", suffix=".csv", dir=self.audit_dir)
            os.close(fd)
            tmp = Path(tmp_name)
            try:
                tmp.write_text(csv_text, encoding="utf-8")
                scored = parse_human_scores_csv(tmp)
                if len(scored) != 12:
                    raise ValueError(f"expected 12 scored rows, got {len(scored)}")
                os.replace(tmp, out)
            finally:
                if tmp.exists():
                    tmp.unlink()
        except (OSError, TypeError, ValueError) as exc:
            self._json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": str(exc)})
            return

        self._json(
            HTTPStatus.OK,
            {"ok": True, "path": str(out), "rows": 12},
        )


def main() -> None:
    p = argparse.ArgumentParser(description="Serve the M7.2.1 blinded human audit over localhost.")
    p.add_argument("--audit-dir", required=True)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--no-open", action="store_true")
    args = p.parse_args()

    audit_dir = Path(args.audit_dir).resolve()
    html = audit_dir / "audit.html"
    if not html.exists():
        raise FileNotFoundError(f"missing audit HTML: {html}; run make m7-human-audit first")

    def handler(*h_args, **h_kwargs):
        return AuditHandler(*h_args, directory=str(audit_dir), **h_kwargs)

    server = ThreadingHTTPServer((args.host, args.port), handler)
    host, port = server.server_address[:2]
    url = f"http://{host}:{port}/audit.html"
    print("M7.2.1 human-audit server")
    print(f"url={url}")
    print(f"save_path={audit_dir / 'human_audit_completed.csv'}")
    print("Complete all 12 cases and press 'Save completed CSV'. Press Ctrl+C when finished.")

    if not args.no_open:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nM7.2.1 human-audit server stopped")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
