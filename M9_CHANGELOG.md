# M9 changelog

- Version `0.9.0`.
- Added JSON-line structured API logging with request IDs and elapsed time.
- Added Prometheus `/metrics` endpoint hidden from the public OpenAPI schema.
- Added request/stage, backpressure, worker liveness/restart/timeout/failure metrics.
- Added encoder/FAISS/VLM request timeouts and worker startup timeout.
- Added one-shot worker transport recovery and dead-worker recovery.
- Extended `/ready` with worker restart/timeout counters.
- Extended `/v1/model-info` with explicit worker timeout configuration.
- Added backpressure/load smoke and real FAISS worker-recovery smoke.
- Migrated TestClient dependency to HTTPX2 and made deprecation/resource warnings fail tests.
- Added exact macOS proven direct-dependency constraints.
- Added GitHub Actions CI, Dependabot, CI Dockerfile, and container/deployment policy.
- Frozen M8.1 retrieval/RAG semantics are unchanged.
