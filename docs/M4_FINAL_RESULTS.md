# M4 Final Results — Exact FlatIP vs HNSW

## Status

**COMPLETE. Production decision: keep `IndexFlatIP` for the current 66,261-screen corpus.**

M4 froze the M3 winner (`google/siglip2-base-patch16-naflex`, `max_num_patches=256`) and changed only the FAISS index. The exact `IndexFlatIP` ranking served as the oracle. Three `IndexHNSWFlat` graphs (`M=16/32/48`, `efConstruction=200`) were each evaluated at `efSearch=16/32/64/128/256`.

## Pre-registered selection gate

An HNSW configuration was eligible only if all three conditions held:

1. exact-neighbor Recall@10 >= 95%;
2. Screen2Words Recall@10 retention >= 98% of FlatIP;
3. single-query p50 speedup >= 1.20x.

No HNSW configuration passed all three conditions.

## Main comparison

| Index | Exact-neighbor R@10 | Task R@10 | Task retention | Single p50 (ms) | Speedup | Batch qps | Size (MiB) |
|---|---:|---:|---:|---:|---:|---:|---:|
| **FlatIP exact** | **100.00%** | **16.22%** | **100.0%** | 3.8590 | 1.00x | **7,726.4** | **194.12** |
| HNSW M16 efS16 | 58.02% | 12.85% | 79.3% | **0.0628** | **61.42x** | 16,227.1 | 203.24 |
| HNSW M16 efS64 | 78.64% | 14.85% | 91.6% | 0.1699 | 22.71x | 6,000.4 | 203.24 |
| HNSW M16 efS256 | 91.29% | 15.80% | 97.4% | 0.5630 | 6.85x | 1,797.3 | 203.24 |
| HNSW M32 efS128 | 87.73% | 15.59% | 96.1% | 0.3403 | 11.34x | 2,992.8 | 211.32 |
| HNSW M32 efS256 | 92.47% | 15.94% | 98.3% | 0.6276 | 6.15x | 1,619.0 | 211.32 |
| HNSW M48 efS128 | 87.99% | 15.64% | 96.4% | 0.3499 | 11.03x | 2,892.3 | 219.41 |
| **HNSW M48 efS256** | **92.88%** | **16.06%** | **99.0%** | **0.6403** | **6.03x** | 1,572.1 | 219.41 |

The full 16-row grid is preserved in `docs/m4_reference_comparison.csv` and `docs/m4_reference_run.json`.

## Decision analysis

### Why FlatIP remains the production index

`HNSW M48 / efSearch=256` is the closest ANN candidate. It retains 99.0% of downstream Recall@10 and makes a single Top-10 lookup about 6x faster, but its exact-neighbor Recall@10 is only 92.88%, missing the 95% fidelity gate by 2.12 percentage points. This means roughly 7.12% of the exact Top-10 neighborhood is lost on average.

The higher-fidelity HNSW settings also reverse the throughput advantage for batched work: `M48/efS256` reaches 1,572 qps versus 7,726 qps for FlatIP at batch size 256. Its serialized index is 219.41 MiB versus 194.12 MiB for FlatIP, about 13.0% larger. Therefore, at 66k vectors FlatIP provides the best overall quality/simplicity/batch-throughput trade-off.

### What HNSW still proves

The experiment demonstrates a real online-latency frontier. At low `efSearch`, HNSW can reduce single-query latency by tens of times, but the loss in nearest-neighbor fidelity is substantial. If the corpus grows by an order of magnitude or the serving workload becomes dominated by latency-sensitive single-query traffic, `M48/efSearch=256` is the first configuration worth re-testing, followed by a broader `efSearch` sweep above 256.

## Build artifacts

| HNSW graph | Build time | Size | SHA-256 |
|---|---:|---:|---|
| M16 / efC200 | 12.788 s | 203.24 MiB | `b9c256e05387ec7e7b20b3acea3ab4be04bc6e523ffc528992bc7daddc6f4b46` |
| M32 / efC200 | 14.771 s | 211.32 MiB | `0d17d8a58bda8dc27e03251ee7d7cc9111e17f91f75fa60028531e389328dfc8` |
| M48 / efC200 | 15.830 s | 219.41 MiB | `25ea954b5e6790391b9821b9f6c93367084da29f1e4e63dd451313e9b8474350` |

Frozen M3 artifacts:

- NaFlex embeddings SHA-256: `509cd1bc04e7a7a578fc6f120e0d58f5fea24be9f1781e573377359c76521fdf`
- exact FlatIP index SHA-256: `a6f0cf7613f1e829252bf38f74750b8bb56cd518e8056aaac2def6bb59e43477`
- exact source manifest SHA-256: `eb24469b5ed0338cc3197504ac266aa605d45582c02a467a0064df44f6bd29ee`

## Portfolio/interview takeaway

M4 is intentionally a negative-selection result: ANN was not adopted merely because it was faster. A pre-defined quality gate showed that the current 66k-vector corpus is small enough for exact search to remain preferable. The project therefore demonstrates both ANN implementation skill and evidence-based restraint in production index selection.
