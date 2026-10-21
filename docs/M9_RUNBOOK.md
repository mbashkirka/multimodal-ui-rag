# M9 operational runbook

## Health signals

- `/health`: process liveness only.
- `/ready`: artifact checks plus worker liveness/restart/timeout counters.
- `/metrics`: Prometheus text metrics (intentionally hidden from public OpenAPI docs).
- `/v1/model-info`: frozen model/artifact identity plus worker timeout configuration.

## Important metrics

- `ui_rag_http_requests_total`
- `ui_rag_http_request_duration_seconds_{sum,count}`
- `ui_rag_stage_duration_seconds_{sum,count}`
- `ui_rag_inference_active`
- `ui_rag_inference_busy_total`
- `ui_rag_worker_alive`
- `ui_rag_worker_restarts_total`
- `ui_rag_worker_timeouts_total`
- `ui_rag_worker_failures_total`

## Recovery behavior

A dead persistent worker is restarted before the next inference request. A transport
failure or request timeout kills the affected worker and performs at most one automatic
retry on a freshly started worker. A model/input error returned cleanly by the worker is
not retried because restarting would repeat the same semantic failure.

Default request timeouts:

- encoder: 30 s
- FAISS: 10 s
- VLM: 60 s
- worker startup: 300 s

All are configurable with `UI_RAG_*_TIMEOUT_SECONDS` environment variables.

## Backpressure

The validated MacBook profile keeps `UI_RAG_MAX_CONCURRENT_INFERENCE=1`. Concurrent
heavy requests that cannot acquire capacity within the configured acquire timeout are
returned as HTTP 429 instead of competing for unified memory.

## Shutdown

SIGINT/SIGTERM is handled by Uvicorn/FastAPI lifespan. The service closes VLM, encoder,
and FAISS workers and escalates from cooperative shutdown to terminate/kill only if a
worker does not exit within the shutdown window.
