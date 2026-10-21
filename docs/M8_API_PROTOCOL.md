# M8 API protocol

M8 exposes the frozen M5/M6 production path; it does not retune retrieval or generation.

## Endpoints

- `GET /health`: process liveness only.
- `GET /ready`: checks the persisted retrieval/reranking artifacts and RICO dataset needed for requests.
- `GET /v1/model-info`: service/API versions, frozen model stack, artifact fingerprint, evidence policy.
- `POST /v1/search`: text retrieval using SigLIP2 NaFlex p256 → FlatIP → p512 weighted RRF.
- `POST /v1/rag`: frozen search path → deterministic rank-1 evidence → local Qwen2.5-VL grounded answer.

## Operational contract

The macOS native-isolation boundary from M1–M6 is preserved. The FastAPI host does not load torch, FAISS, or MLX at import/startup. Existing child-process boundaries perform encoder, FAISS and MLX work. A bounded semaphore defaults to one in-flight inference request; excess requests receive HTTP 429 rather than competing for memory.

Public responses do not expose absolute local image paths. Each response carries `X-Request-ID`; callers may supply their own request ID. Latency is reported by stage where available. Large FAISS/mapping/p512 artifact hashes are fully verified once before the first data-plane request and then cached, avoiding repeated multi-hundred-megabyte hashing on every request.
