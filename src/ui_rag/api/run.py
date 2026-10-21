from __future__ import annotations

import argparse
import os

from ui_rag.api.observability import configure_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the UI-RAG FastAPI service")
    parser.add_argument("--host", default=os.environ.get("UI_RAG_API_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("UI_RAG_API_PORT", "8000")))
    parser.add_argument("--log-level", default=os.environ.get("UI_RAG_LOG_LEVEL", "info"))
    args = parser.parse_args()

    configure_logging(args.log_level)

    import uvicorn

    uvicorn.run(
        "ui_rag.api.app:app",
        host=args.host,
        port=args.port,
        log_level=args.log_level,
        workers=1,
        log_config=None,
        access_log=False,
    )


if __name__ == "__main__":
    main()
