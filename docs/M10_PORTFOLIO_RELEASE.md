# M10 — Portfolio Release Candidate

## Goal

M10 does not change retrieval, reranking, generation, or serving behavior. It packages the validated M9 system so that a reviewer can understand the project in minutes and drill into protocols/results without reconstructing the development conversation.

## Public-facing deliverables

- concise recruiter-oriented `README.md`;
- consolidated architecture diagrams;
- consolidated benchmark/evaluation table with caveats;
- end-to-end engineering case study;
- five-minute demo guide;
- interview narrative and technical Q&A;
- reproducibility guide;
- public release checklist;
- automated M10 documentation/release hygiene check.

## Frozen technical state

M10 inherits the validated M9 contract:

- SigLIP2 Base NaFlex p256 global retrieval;
- exact FAISS IndexFlatIP;
- p512 Top-20 weighted RRF reranking;
- deterministic rank-1 evidence selection;
- Qwen2.5-VL-3B 4-bit generation on MLX;
- persistent isolated encoder / FAISS / VLM workers;
- FastAPI v1;
- structured logs, metrics, timeouts, restart/retry, backpressure;
- target-Mac M9 acceptance: 109 tests, Ruff/release/compile passes.

## Versioning

The M10 release candidate uses project version `0.10.0.1`. After the public-repository acceptance checks and GitHub presentation are confirmed, it is appropriate to tag the repository as a stable `v1.0.0` without changing ML semantics.
