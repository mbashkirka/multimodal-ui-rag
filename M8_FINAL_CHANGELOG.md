# M8 FINAL Changelog

- Marked M8 Production FastAPI Service as COMPLETE after target-Mac validation.
- Retained `ui-rag 0.8.1` and the M8.1 persistent isolated worker architecture.
- Recorded live `/health`, `/ready`, `/v1/model-info`, `/v1/search`, and `/v1/rag` acceptance results.
- Recorded M8.0 vs M8.1 latency results.
- Documented ~89x steady-state search speedup (15.2 s -> 171 ms).
- Documented RAG latency reduction from 34.1 s to 16.9–20.1 s and the remaining VLM-generation bottleneck.
- Documented post-VLM Apple Silicon latency variability as an M9 operational-hardening item.
- No change to frozen retrieval, reranking, evidence policy, or generation semantics.
