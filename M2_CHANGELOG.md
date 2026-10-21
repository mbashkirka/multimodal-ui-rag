# M2 Changelog — Retrieval Benchmark & Evaluation

## M2.1 benchmark harness

- Added the official Screen2Words test split as an external text-to-RICO retrieval benchmark.
- Added `make benchmark-data` to download and validate the official source files.
- Added a primary protocol: 4,310 screens × one human summary = 4,310 queries.
- Added a paraphrase protocol: 4,310 screens × five human summaries = 21,550 queries.
- Added Recall@1/5/10/50/100, MRR@10, and nDCG@10.
- Added query-embedding throughput and batch latency metrics.
- Added FAISS exact-search throughput and batch latency metrics.
- Added machine-readable `metrics.json`, per-query `predictions.jsonl`, and `failures.csv`.
- Preserved macOS ARM native-process isolation: OpenCLIP/PyTorch query embedding and FAISS evaluation run in different interpreters.
- Added synthetic tests for benchmark construction, metrics, and native import boundaries.
- M1 model/index artifacts remain frozen; M2 performs evaluation only.

## M2 final reference run (macOS Apple Silicon)

- `15 passed` in the M2 test suite.
- Screen2Words sources validated: 112,085 summaries, 22,417 screens, 4,310 test screens.
- Primary 4,310-query run completed on MPS + exact FAISS:
  - Recall@1: 1.137%
  - Recall@5: 3.248%
  - Recall@10: 4.826%
  - Recall@50: 10.696%
  - Recall@100: 13.875%
  - MRR@10: 0.02088
  - nDCG@10: 0.02726
- All-five 21,550-query paraphrase run completed:
  - Recall@1: 1.267%
  - Recall@5: 3.350%
  - Recall@10: 4.831%
  - Recall@50: 10.278%
  - Recall@100: 13.740%
  - MRR@10: 0.02198
  - nDCG@10: 0.02813
- Exact FAISS search measured at ~0.12 ms/query; CLIP text encoding at ~4.39 ms/query.
- M2 conclusion: encoder quality is the dominant next optimization target; ANN indexing is deferred.
- Reference metrics and hashes recorded in `docs/M2_FINAL_RESULTS.md` and `docs/m2_reference_run.json`.
