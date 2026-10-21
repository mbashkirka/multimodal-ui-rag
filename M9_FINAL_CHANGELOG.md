# M9 FINAL Changelog — Engineering & Operational Hardening Complete

- Final package version: `0.9.1.1`.
- M9 accepted on the target Apple-Silicon Mac with **109 passed** tests and a clean Ruff gate.
- Release-contract and `compileall` gates passed.
- Real FAISS worker recovery validated: process killed, restarted once, and healthy afterward.
- Prometheus-compatible worker/backpressure metrics validated on the live server.
- Six-way concurrent search smoke produced **1 HTTP 200 + 5 controlled HTTP 429 + 0 5xx**, with `ui_rag_inference_busy_total=5`.
- `/ready` confirmed encoder, FAISS, and VLM workers alive after operational tests.
- M9.1 and M9.1.1 are retained as the documented quality-gate repair history; lint policy was not weakened.
- Added `docs/M9_FINAL_RESULTS.md`, `docs/m9_reference_run.json`, and `docs/m9_reference_summary.csv`.
- Retrieval, reranking, evidence selection, VLM model, and API semantics remain frozen from M8.1/M6.4.
- M9 is COMPLETE; next milestone is M10 portfolio/release packaging.
