# Evaluation & Benchmarks

This page consolidates the portfolio-facing results. Detailed protocols and reference JSON/CSV remain in the milestone documents.

## Retrieval benchmark

Benchmark: Screen2Words official test split, 4,310 screens, one human summary per target screen, exact target RICO identity as the relevant item.

| System | R@1 | R@5 | R@10 | R@100 | MRR@10 | nDCG@10 |
|---|---:|---:|---:|---:|---:|---:|
| Legacy CLIP ViT-B/32 | 1.14% | 3.25% | 4.83% | 13.87% | 0.0209 | 0.0273 |
| SigLIP2 Base FixRes | 4.55% | 11.46% | 14.94% | 32.92% | 0.0740 | 0.0917 |
| SigLIP2 Base NaFlex p256 | 5.20% | 12.06% | 16.22% | 37.31% | 0.0826 | 0.1013 |
| **NaFlex p256 + p512 Top-20 RRF** | **5.22%** | **12.55%** | **16.73%** | — | **0.0846** | **0.1041** |

Final R@10 is **3.47×** the legacy baseline (`16.73 / 4.83`).

### Why the absolute recall is still low

This is exact-instance retrieval. A visually or functionally equivalent screenshot with a different RICO ID is counted as wrong. The metric is intentionally strict so that encoder/index changes can be compared deterministically.

## Exact vs HNSW study

Selection gate:

- exact-neighbor Recall@10 ≥ 95%;
- task R@10 retention ≥ 98%;
- single-query p50 speedup ≥ 1.2×.

Best tested HNSW quality/speed compromise (`M=48`, `efSearch=256`):

- exact-neighbor Recall@10: **92.88%**;
- task R@10: **16.06%**;
- task retention: **99%**;
- p50: **0.640 ms** vs exact **3.859 ms**;
- speedup: **6.03×**.

It failed the 95% neighbor-recall gate, so exact FlatIP stayed in production.

## M5 multimodal diagnostic

The text+visual-exemplar diagnostic reached **77.05% R@10**. It must not be compared directly to normal text retrieval because the exemplar is selected using target knowledge. It is retained only as an upper-bound diagnostic showing how informative a related visual observation can be.

## VLM role ablations

| Variant | Primary measured result | Decision |
|---|---|---|
| Multi-image JSON grounding | 12.5% valid JSON+citations; 10% selection accuracy on retrieval hits | Reject |
| Contact-sheet categorical selector | 50% exact selection on hits vs 55% always-E1 baseline | Reject |
| Single-image verifier / gate | 90.91% false abstention when the exact target was already E1 | Reject |
| **Generation-only VLM** | deterministic E1 provenance + grounded single-image answer | **Select** |

## Generation evaluation

30-case deterministic holdout:

- citation validity: **100%**;
- non-empty answers: **100%**;
- clean sentence ending: **93.33%**;
- generation mean: **10.14 s**;
- generation p95: **13.47 s**.

12-case blinded human audit:

| Metric | Result |
|---|---:|
| Faithfulness | **91.67%** |
| Relevance | **75.00%** |
| Completeness | **75.00%** |
| Any unsupported claim | **58.33% (7/12)** |
| Major unsupported claim | **8.33% (1/12)** |

These percentages are compact release-audit results from one evaluator and 12 cases; they are not statistically precise population estimates.

## Automated judge calibration failure

The same-family Qwen2.5-VL automated judge was stress-tested with deterministic corruptions:

| Calibration metric | Result |
|---|---:|
| Irrelevant detection | **0%** |
| Unsupported-sentinel detection | **0%** |
| Truncation detection | **0%** |
| Perfect score on corrupted answers | **100%** |
| Unique score codes | **1** |
| Gate | **FAIL** |

Therefore its semantic F/R/U/C scores are excluded from portfolio quality claims.

## Serving latency

### M8.0: per-request model-loading baseline

| Endpoint | Total | Encode | Search | Rerank | Generation |
|---|---:|---:|---:|---:|---:|
| Search | 15,217 ms | 14,881 ms | 314 ms | 21 ms | — |
| RAG | 34,126 ms | 12,206 ms | 394 ms | 18 ms | 21,501 ms |

### M8.1: persistent isolated workers

| Endpoint | Run | Total | Encode | Search | Rerank | Generation |
|---|---:|---:|---:|---:|---:|---:|
| Search | warm run #2 | **171 ms** | **165 ms** | **5 ms** | **0.5 ms** | — |
| RAG | run #1 | **16,934 ms** | 18 ms | 5 ms | 0.1 ms | **16,906 ms** |
| RAG | run #2 | 20,143 ms | 3,366 ms | 1,246 ms | 16 ms | **15,511 ms** |

The search path improved by roughly **89×** compared with M8.0. Remaining RAG latency is dominated by VLM generation and, in some runs, Apple unified-memory/scheduler interference after heavy VLM work.

## Operational acceptance

Target Mac M9 release gate:

- Ruff: passed;
- Pytest: **109 passed**;
- native-import isolation: true;
- real FAISS worker kill/recovery: passed;
- six concurrent search requests: **1×200 + 5×429 + 0×5xx**;
- busy counter increment: **5**;
- all production-server workers alive after smoke tests.
