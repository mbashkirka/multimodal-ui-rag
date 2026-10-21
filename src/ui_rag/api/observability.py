from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import Any

_EXTRA_FIELDS = (
    "event",
    "request_id",
    "method",
    "path",
    "status_code",
    "elapsed_ms",
    "error_type",
    "worker",
    "restart_count",
    "timeout_count",
)


class JsonFormatter(logging.Formatter):
    """Compact JSON-lines formatter for service logs."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for field in _EXTRA_FIELDS:
            value = getattr(record, field, None)
            if value is not None:
                payload[field] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def configure_logging(level: str = "INFO") -> None:
    """Configure a single warning-clean JSON log stream for the API process."""

    root = logging.getLogger()
    root.handlers.clear()
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    root.addHandler(handler)
    root.setLevel(level.upper())

    # We emit one request-complete event ourselves, so disable duplicate Uvicorn access logs.
    logging.getLogger("uvicorn.access").disabled = True
