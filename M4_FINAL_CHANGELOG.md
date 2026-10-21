# M4 Final Changelog

## 0.4.0 — M4 completed

- Completed 35/35 regression tests on the target macOS environment.
- Verified HNSW smoke retrieval over 1,000 NaFlex vectors with exact self-retrieval at rank 1.
- Built full 66,261-vector HNSW graphs for M=16/32/48 at efConstruction=200.
- Swept efSearch=16/32/64/128/256 across all three graphs under one FAISS CPU thread.
- Measured exact-neighbor Recall@10, downstream Screen2Words Recall@10 retention, single-query p50 latency, batch throughput, index size, and build cost.
- No HNSW configuration satisfied all pre-registered production gates.
- Selected exact `IndexFlatIP` as the production index for the current 66k-screen corpus.
- Recorded `HNSW M48 / efSearch=256` as the closest future scalability candidate (92.88% exact-neighbor R@10, 99.0% task retention, 6.03x single-query speedup).
- Added final M4 results and machine-readable reference snapshots under `docs/`.
