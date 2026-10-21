# M9 Engineering & Operational Hardening Protocol

M9 does not change retrieval, reranking, evidence selection, or generation semantics.
It hardens the M8.1 production service around those frozen components.

## Gates

1. **Quality gate**: exact-version dev tooling, ruff, warning-clean pytest, compileall,
   synchronized package version, OpenAPI contract check, native-import isolation.
2. **Observability gate**: JSON request logs, Prometheus endpoint, request/stage metrics,
   worker restart/timeout/failure counters.
3. **Resilience gate**: bounded worker request/startup timeouts, one-shot transport
   recovery, dead-worker recovery, graceful shutdown.
4. **Backpressure gate**: concurrent search load must produce successful responses and
   controlled HTTP 429 responses with `ui_rag_inference_busy_total` incremented.
5. **CI gate**: Linux quality job + macOS ARM native-contract job using current official
   GitHub Actions majors.
6. **Container policy gate**: Docker is provided for code-quality reproducibility only;
   no unsupported claim of Metal-equivalent production inference in a Linux container.

## Non-goals

- changing M5 retrieval metrics;
- changing M6/M7 prompts or answer policy;
- increasing inference concurrency on the MacBook profile;
- claiming Linux/CUDA deployment parity without a separate benchmark.
