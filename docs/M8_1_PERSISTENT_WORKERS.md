# M8.1 Persistent Worker Serving Protocol

## Motivation

The first end-to-end M8.0 API validation passed functionally, but the reference Mac showed serving latency dominated by process/model startup:

- `/v1/search`: 15.217 s total; 14.881 s encode
- `/v1/rag`: 34.126 s total; 12.206 s encode; 21.501 s generation

The CLI subprocess architecture was intentionally designed for native-library isolation and reproducibility, not high-QPS serving. Reusing that one-shot architecture directly in HTTP serving therefore reloaded SigLIP2, FAISS, and the VLM on every request.

## Architecture

```text
FastAPI host (NumPy + orchestration only)
   |-- persistent encoder worker: torch + Transformers + SigLIP2
   |-- persistent FAISS worker: FAISS + NumPy + frozen FlatIP index
   `-- persistent VLM worker: MLX + mlx-vlm + Qwen2.5-VL
```

The three workers are separate OS processes. This preserves the macOS native-isolation contract while keeping heavy resources resident.

## Frozen semantics

M8.1 does not modify:

- SigLIP2 NaFlex p256 query embedding
- FlatIP candidate retrieval
- Top-20 candidate count
- p512 weighted RRF (`0.5/0.5`, constant `60`)
- deterministic rank-1 evidence policy
- Qwen2.5-VL generation prompt and image preprocessing

## Lifecycle

FastAPI lifespan:

1. validate artifact hashes once;
2. memory-map p512 reranker embeddings;
3. start encoder worker and load SigLIP2 once;
4. start FAISS worker and load index/mapping once;
5. start VLM worker and load Qwen2.5-VL once;
6. mark application startup complete.

On shutdown, workers are terminated cleanly.

## Validation

Run the server, then execute `make m8-latency-smoke`. The script issues two `/v1/search` calls and two `/v1/rag` calls. Compare run 1 and run 2 to distinguish first-inference kernel warmup from steady-state latency.
