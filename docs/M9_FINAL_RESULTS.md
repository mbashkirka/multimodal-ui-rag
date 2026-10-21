# M9 FINAL Results — Engineering & Operational Hardening

## Status

**M9 COMPLETE.** Final acceptance was executed on the target macOS Apple Silicon environment against `ui-rag 0.9.1.1`. Retrieval, reranking, evidence selection, and VLM generation semantics remained frozen from M8.1/M6.4.

## Final quality gate

The target Mac completed the full local M9 quality gate successfully:

- Ruff: **All checks passed**
- Pytest: **109 passed**
- Release-contract smoke: **PASSED**
- `compileall`: **PASSED**
- OpenAPI public routes: `/health`, `/ready`, `/v1/model-info`, `/v1/search`, `/v1/rag`
- `/metrics` intentionally hidden from OpenAPI
- Native import isolation: **true**

The M9.1/M9.1.1 hotfixes repaired historical static-analysis debt exposed by the first strict repository-wide Ruff gate. The lint policy was not weakened.

## Resilience acceptance

A real FAISS persistent worker was started, successfully queried, intentionally killed, and recovered on the next request:

| Field | Observed value |
|---|---:|
| Worker | `faiss` |
| PID before | `32508` |
| PID after | `32511` |
| Restart count | **1** |
| Timeout count | **0** |
| Alive after recovery | **true** |

This validates dead-worker recovery while preserving native-process isolation.

## Observability acceptance

With the live M9 server, `/metrics` reported:

- `ui_rag_inference_active 0`
- `ui_rag_inference_busy_total 0` before load smoke
- encoder / FAISS / VLM `worker_alive = 1`
- worker restart / timeout / failure counters initially `0`

`/ready` after the operational tests reported all three workers alive with zero restart and timeout counters in the production server process.

## Backpressure / load acceptance

Six concurrent `/v1/search` requests were sent against a service configured for one concurrent heavy inference request.

Observed result:

| Status | Count |
|---|---:|
| HTTP 200 | **1** |
| HTTP 429 | **5** |
| HTTP 500/503 | **0** |

`ui_rag_inference_busy_total` increased to **5**, exactly matching the five controlled rejections. This validates deliberate backpressure rather than uncontrolled unified-memory contention.

The successful request in this particular load smoke measured 5,671 ms total (5,148 ms encode, 502 ms search, 16 ms rerank). This sample is not used as the steady-state latency reference because it was obtained during a concurrent/cold-ish operational smoke. The M8.1 warm latency reference remains the dedicated steady-state measurement: 171 ms total `/v1/search` with 165 ms encode, 5 ms FAISS search, and 0.5 ms rerank.

## Frozen operational contract

- Package version: `0.9.1.1`
- API version: `v1`
- Serving mode: `persistent-isolated-workers`
- Retrieval encoder: SigLIP2 Base NaFlex p256
- Search index: exact FAISS `IndexFlatIP`
- Reranker: SigLIP2 NaFlex p512 + weighted RRF (0.5/0.5, k=60)
- Evidence policy: deterministic rank-1
- VLM: `mlx-community/Qwen2.5-VL-3B-Instruct-4bit`
- Max concurrent heavy inference: `1`
- Worker timeouts: encoder 30 s, FAISS 10 s, VLM 60 s, startup 300 s
- Structured logging: JSON lines
- Metrics: Prometheus-compatible `/metrics`
- Artifact fingerprint: `ec4a3ac7411b2aaf07d14d36b17b8e51de991d274c2ef008c59e7a74f1d03d43`

## Deployment policy

The validated production data plane remains native macOS Apple Silicon for the frozen MLX configuration. The repository's Docker image is a CI/contract environment, not a claim of Linux container inference parity.

## Decision

M9 is complete. The service now has a green repository-wide quality gate, observable worker/request state, bounded worker timeouts, automatic worker recovery, graceful lifecycle handling, and controlled overload behavior. Further work belongs to M10 portfolio/release packaging rather than operational correctness of the current service.
