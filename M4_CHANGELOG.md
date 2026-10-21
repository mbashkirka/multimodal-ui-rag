# M4 Changelog

## 0.4.0 — Exact vs ANN/HNSW experiment

- Froze the M3 winner: SigLIP2 Base NaFlex with `max_num_patches=256`.
- Reuses the existing 66,261-vector M3 embedding artifact; no corpus re-encoding in M4.
- Added FAISS-only `build-hnsw` CLI command.
- Added `IndexHNSWFlat` grid for M=16/32/48 with fixed efConstruction=200.
- Added runtime efSearch sweep over 16/32/64/128/256.
- Added exact-neighbor Recall@K and Top-1 agreement against the FlatIP oracle.
- Added Screen2Words task-metric retention, single-query latency, batch throughput, index size,
  and HNSW build-time measurement.
- Added an explicit selection gate: >=95% exact-neighbor R@10, >=98% task R@10 retention,
  and >=1.20x online p50 speedup; otherwise retain FlatIP.
- Added M4 HNSW smoke test and pure-Python regression tests.
- Preserved macOS native-process isolation: M4 scripts import FAISS/NumPy only, never Torch.
