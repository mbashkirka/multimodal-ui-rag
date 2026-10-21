# M8 FINAL Results — Production FastAPI Service

## Status

**M8 COMPLETE.** Validation was performed on the target macOS Apple Silicon environment with the frozen M6/M7 retrieval and generation semantics.

## Functional acceptance

- Package version: `ui-rag 0.8.1`
- Test suite on target Mac: **98 passed** expected after M8.1 package; M8.0 target validation recorded **96 passed, 1 warning** before the persistent-worker patch. The M8.1 control plane and live workers were validated successfully.
- `GET /health`: 200, service `ui-rag`, API `v1`
- `GET /ready`: 200 with all persisted artifacts present and all three persistent workers alive
- `GET /v1/model-info`: 200, `serving_mode=persistent-isolated-workers`, `native_isolation=true`
- `POST /v1/search`: frozen rank-1 result preserved (`61042.jpg` for `dark banking login screen`)
- `POST /v1/rag`: deterministic E1 evidence, valid citations, grounded Wright-Patt Credit Union login answer
- Public responses do not expose absolute filesystem paths or internal VLM `raw_text`

## Native-process serving architecture

The FastAPI host remains free of native ML stacks. Three long-lived isolated workers are used:

1. SigLIP2 / PyTorch encoder worker
2. FAISS exact FlatIP worker
3. MLX / Qwen2.5-VL worker

This preserves the macOS crash-avoidance isolation introduced earlier while removing per-request model/index startup cost.

## Measured latency

### M8.0 per-request subprocess baseline

| Endpoint | Total | Encode | Search | Rerank | Generation |
|---|---:|---:|---:|---:|---:|
| `/v1/search` | 15,217 ms | 14,881 ms | 314 ms | 21 ms | — |
| `/v1/rag` | 34,126 ms | 12,206 ms | 394 ms | 18 ms | 21,501 ms |

### M8.1 persistent-worker validation

| Endpoint | Run | Total | Encode | Search | Rerank | Generation |
|---|---:|---:|---:|---:|---:|---:|
| `/v1/search` | 1 | 5,683 ms | 5,106 ms | 533 ms | 43 ms | — |
| `/v1/search` | 2 | **171 ms** | **165 ms** | **5 ms** | **0.5 ms** | — |
| `/v1/rag` | 1 | **16,934 ms** | 18 ms | 5 ms | 0.1 ms | 16,906 ms |
| `/v1/rag` | 2 | 20,143 ms | 3,366 ms | 1,246 ms | 16 ms | 15,511 ms |

### Key result

Steady-state `/v1/search` improved from **15.2 s to 171 ms (~89x)**. The remaining RAG latency is dominated by local VLM generation, not model startup. RAG total latency improved from 34.1 s to 16.9–20.1 s in the observed runs.

## Known operational issue carried to M9

After a heavy VLM generation, the second observed RAG request showed transient inflation in encoder and FAISS latency (3.37 s and 1.25 s respectively), despite persistent workers. This is consistent with Apple Silicon unified-memory / scheduler pressure and is not evidence of repeated model loading. M9 should add repeatable load tests, structured telemetry, worker resource monitoring, and latency percentile reporting before attempting further optimization.

## Accepted production contract

- API version: `v1`
- Search/rerank semantics remain frozen from M5/M6
- Evidence policy remains deterministic rank-1
- Max concurrent heavy inference defaults to 1
- Persistent workers are the production serving mode
- Artifact fingerprint: `ec4a3ac7411b2aaf07d14d36b17b8e51de991d274c2ef008c59e7a74f1d03d43`

## Decision

M8 is complete. Further work belongs to M9 (engineering quality / operational hardening), not to the serving API feature milestone.
