# M8.1 — Persistent isolated workers

M8.0 proved API correctness but exposed a serving bottleneck on the reference Mac: every HTTP request spawned fresh encoder/FAISS/VLM subprocesses. The observed `/v1/search` request spent ~14.9 s in the encoder stage and `/v1/rag` spent ~12.2 s in encoding plus ~21.5 s in generation, dominated by repeated model/process initialization.

M8.1 changes serving only; frozen M5/M6 retrieval and generation semantics remain unchanged.

## Changes

- Added three long-lived JSONL workers:
  - SigLIP2 encoder worker (torch/Transformers only)
  - FAISS worker (FAISS/NumPy only)
  - MLX VLM worker (MLX/MLX-VLM only)
- Workers are started during FastAPI lifespan and reused across requests.
- API host still does not import torch, FAISS, MLX, or mlx-vlm.
- p512 reranker embeddings are memory-mapped once in the API host.
- Artifact SHA-256 validation occurs once before worker startup.
- `/ready` reports worker liveness after startup.
- `/v1/model-info` exposes `serving_mode=persistent-isolated-workers`.
- Added a live `m8-latency-smoke` target for two search + two RAG requests.
- M8.0 per-request subprocess path remains available through the CLI; only API serving changes.

## Acceptance criterion

Functional output must remain identical to the frozen M8.0 path (same retrieval ranking and deterministic E1 policy), while repeated HTTP requests must no longer include model/index load time on every call.
