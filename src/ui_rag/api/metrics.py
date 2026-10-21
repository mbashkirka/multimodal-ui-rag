from __future__ import annotations

import threading
from collections import defaultdict
from typing import Any

_PROM_CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"


def _labels(**values: str | int) -> str:
    if not values:
        return ""
    escaped = []
    for key, value in sorted(values.items()):
        text = str(value).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
        escaped.append(f'{key}="{text}"')
    return "{" + ",".join(escaped) + "}"


class MetricsRegistry:
    """Small in-process Prometheus registry with bounded label cardinality.

    The API has only a fixed route set and fixed worker names, so a minimal registry keeps
    M9 dependency-light while still exposing standard Prometheus exposition text.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._http_total: dict[tuple[str, str, int], int] = defaultdict(int)
        self._http_seconds_sum: dict[tuple[str, str], float] = defaultdict(float)
        self._http_seconds_count: dict[tuple[str, str], int] = defaultdict(int)
        self._stage_seconds_sum: dict[str, float] = defaultdict(float)
        self._stage_seconds_count: dict[str, int] = defaultdict(int)
        self._busy_total = 0
        self._active_inference = 0

    def observe_http(self, *, method: str, path: str, status: int, seconds: float) -> None:
        with self._lock:
            self._http_total[(method, path, int(status))] += 1
            self._http_seconds_sum[(method, path)] += float(seconds)
            self._http_seconds_count[(method, path)] += 1

    def observe_stage(self, stage: str, milliseconds: float | None) -> None:
        if milliseconds is None:
            return
        with self._lock:
            self._stage_seconds_sum[stage] += float(milliseconds) / 1000.0
            self._stage_seconds_count[stage] += 1

    def inference_acquired(self) -> None:
        with self._lock:
            self._active_inference += 1

    def inference_released(self) -> None:
        with self._lock:
            self._active_inference = max(0, self._active_inference - 1)

    def inference_busy(self) -> None:
        with self._lock:
            self._busy_total += 1

    def render(self, worker_diagnostics: dict[str, dict[str, Any]] | None = None) -> str:
        worker_diagnostics = worker_diagnostics or {}
        with self._lock:
            http_total = dict(self._http_total)
            http_sum = dict(self._http_seconds_sum)
            http_count = dict(self._http_seconds_count)
            stage_sum = dict(self._stage_seconds_sum)
            stage_count = dict(self._stage_seconds_count)
            busy_total = self._busy_total
            active = self._active_inference

        lines = [
            "# HELP ui_rag_http_requests_total HTTP requests by route and status.",
            "# TYPE ui_rag_http_requests_total counter",
        ]
        for (method, path, status), value in sorted(http_total.items()):
            lines.append(
                f"ui_rag_http_requests_total{_labels(method=method, path=path, status=status)} {value}"
            )

        lines += [
            "# HELP ui_rag_http_request_duration_seconds_sum Cumulative HTTP request time.",
            "# TYPE ui_rag_http_request_duration_seconds_sum counter",
        ]
        for (method, path), value in sorted(http_sum.items()):
            lines.append(
                f"ui_rag_http_request_duration_seconds_sum{_labels(method=method, path=path)} {value:.9f}"
            )
        lines += [
            "# HELP ui_rag_http_request_duration_seconds_count Number of timed HTTP requests.",
            "# TYPE ui_rag_http_request_duration_seconds_count counter",
        ]
        for (method, path), value in sorted(http_count.items()):
            lines.append(
                f"ui_rag_http_request_duration_seconds_count{_labels(method=method, path=path)} {value}"
            )

        lines += [
            "# HELP ui_rag_stage_duration_seconds_sum Cumulative pipeline stage time.",
            "# TYPE ui_rag_stage_duration_seconds_sum counter",
        ]
        for stage, value in sorted(stage_sum.items()):
            lines.append(f"ui_rag_stage_duration_seconds_sum{_labels(stage=stage)} {value:.9f}")
        lines += [
            "# HELP ui_rag_stage_duration_seconds_count Number of pipeline stage observations.",
            "# TYPE ui_rag_stage_duration_seconds_count counter",
        ]
        for stage, value in sorted(stage_count.items()):
            lines.append(f"ui_rag_stage_duration_seconds_count{_labels(stage=stage)} {value}")

        lines += [
            "# HELP ui_rag_inference_active Current admitted heavy inference requests.",
            "# TYPE ui_rag_inference_active gauge",
            f"ui_rag_inference_active {active}",
            "# HELP ui_rag_inference_busy_total Requests rejected by the inference capacity gate.",
            "# TYPE ui_rag_inference_busy_total counter",
            f"ui_rag_inference_busy_total {busy_total}",
        ]

        for metric, field, help_text in (
            ("ui_rag_worker_alive", "alive", "Whether a persistent native worker is alive."),
            ("ui_rag_worker_restarts_total", "restarts", "Persistent worker restarts."),
            ("ui_rag_worker_timeouts_total", "timeouts", "Persistent worker request timeouts."),
            ("ui_rag_worker_failures_total", "failures", "Persistent worker transport failures."),
        ):
            lines += [f"# HELP {metric} {help_text}", f"# TYPE {metric} {'gauge' if field == 'alive' else 'counter'}"]
            for worker, diag in sorted(worker_diagnostics.items()):
                value = int(bool(diag.get(field))) if field == "alive" else int(diag.get(field, 0))
                lines.append(f"{metric}{_labels(worker=worker)} {value}")

        return "\n".join(lines) + "\n"


PROMETHEUS_CONTENT_TYPE = _PROM_CONTENT_TYPE
