# M8 — Production FastAPI Service

- Added versioned FastAPI serving layer (`ui-rag 0.8.0`).
- Added `GET /health`, `GET /ready`, `GET /v1/model-info`.
- Added `POST /v1/search` for frozen M5 retrieval + p512 RRF reranking.
- Added `POST /v1/rag` for frozen M6.4 deterministic-E1 grounded generation.
- Added request IDs and `X-Process-Time-Ms` response header.
- Added structured validation/runtime errors and a bounded inference concurrency gate.
- Added stage latency breakdown (encode/search/rerank/generation).
- Full artifact SHA verification is performed once on first data-plane use and cached; per-request child searches skip redundant large-file hashing.
- Public API responses intentionally omit absolute filesystem paths and VLM `raw_text`.
- Preserved native process isolation: API startup does not import torch, FAISS, MLX, or mlx-vlm.
- Added API regression tests and control-plane smoke command.
