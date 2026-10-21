from __future__ import annotations

import json
import logging

from ui_rag.api.metrics import MetricsRegistry
from ui_rag.api.observability import JsonFormatter


def test_json_formatter_emits_structured_fields():
    record = logging.LogRecord(
        name="ui_rag.api",
        level=logging.INFO,
        pathname=__file__,
        lineno=10,
        msg="request complete",
        args=(),
        exc_info=None,
    )
    record.event = "request_complete"
    record.request_id = "r-1"
    record.status_code = 200
    payload = json.loads(JsonFormatter().format(record))
    assert payload["event"] == "request_complete"
    assert payload["request_id"] == "r-1"
    assert payload["status_code"] == 200
    assert payload["level"] == "INFO"


def test_metrics_registry_exposes_http_pipeline_and_worker_metrics():
    metrics = MetricsRegistry()
    metrics.observe_http(method="POST", path="/v1/search", status=200, seconds=0.2)
    metrics.observe_stage("encode", 150.0)
    metrics.inference_acquired()
    metrics.inference_busy()
    text = metrics.render({
        "encoder": {"alive": True, "restarts": 2, "timeouts": 1, "failures": 3}
    })
    assert 'ui_rag_http_requests_total{method="POST",path="/v1/search",status="200"} 1' in text
    assert 'ui_rag_stage_duration_seconds_count{stage="encode"} 1' in text
    assert "ui_rag_inference_active 1" in text
    assert "ui_rag_inference_busy_total 1" in text
    assert 'ui_rag_worker_restarts_total{worker="encoder"} 2' in text
