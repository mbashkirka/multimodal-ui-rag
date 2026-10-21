# M2 Final Results — Retrieval Benchmark & Evaluation

## Status

**M2 COMPLETE** on macOS Apple Silicon (MPS query encoding + CPU FAISS exact search).

The frozen M1 retriever was evaluated without changing model weights, preprocessing,
normalization, corpus, or FAISS index configuration.

## Benchmark

Primary external benchmark: **Screen2Words test split** mapped to the canonical RICO
corpus.

- Screen2Words source summaries: 112,085
- Unique Screen2Words screens: 22,417
- Official test screens: 4,310
- Canonical RICO corpus: 66,261 screens
- Relevant item definition: exact target RICO screen identity
- Search depth: top 100

Two reporting protocols were run:

1. **Primary** — one human summary per test screen: 4,310 queries.
2. **All-5 paraphrase robustness** — all five human summaries per test screen: 21,550 queries.

## Frozen retriever

- Encoder: `ViT-B-32-quickgelu`
- Pretrained weights: `openai`
- Embedding dimension: 512
- Embedding dtype: float32
- Normalization: L2
- Corpus index: FAISS `IndexFlatIP`
- Corpus vectors: 66,261
- Query encoder device: Apple MPS
- FAISS search device: CPU
- macOS native-process isolation: enabled

## Primary benchmark — 4,310 queries

| Metric | Result |
|---|---:|
| Recall@1 | 1.137% |
| Recall@5 | 3.248% |
| Recall@10 | 4.826% |
| Recall@50 | 10.696% |
| Recall@100 | 13.875% |
| MRR@10 | 0.02088 |
| nDCG@10 | 0.02726 |
| Hits within top 100 | 598 / 4,310 |
| Misses within top 100 | 3,712 / 4,310 |
| Median target rank among hits | 20 |
| Mean target rank among hits | 29.69 |

### Primary latency

| Stage | Result |
|---|---:|
| Query embedding throughput | 227.71 queries/s |
| Query embedding amortized latency | 4.392 ms/query |
| Embedding batch p50 | 557.27 ms |
| Embedding batch p95 | 560.20 ms |
| FAISS throughput | 8,335.30 queries/s |
| FAISS amortized latency | 0.120 ms/query |
| FAISS batch p50 | 28.02 ms |
| FAISS batch p95 | 29.01 ms |

About 97.3% of the measured online compute time per query is spent in query embedding,
not FAISS exact search. This makes the encoder the dominant latency component as well
as the main quality bottleneck for the next milestone.

## All-5 paraphrase benchmark — 21,550 queries

| Metric | Result |
|---|---:|
| Recall@1 | 1.267% |
| Recall@5 | 3.350% |
| Recall@10 | 4.831% |
| Recall@50 | 10.278% |
| Recall@100 | 13.740% |
| MRR@10 | 0.02198 |
| nDCG@10 | 0.02813 |
| Hits within top 100 | 2,961 / 21,550 |
| Misses within top 100 | 18,589 / 21,550 |
| Median target rank among hits | 22 |
| Mean target rank among hits | 30.94 |

### All-5 latency

| Stage | Result |
|---|---:|
| Query embedding throughput | 228.49 queries/s |
| Query embedding amortized latency | 4.377 ms/query |
| FAISS throughput | 8,374.97 queries/s |
| FAISS amortized latency | 0.119 ms/query |

## Interpretation

### 1. The baseline is technically correct but semantically weak for exact UI instance retrieval

Recall@10 is only 4.83% and Recall@100 is 13.87% on the primary benchmark. The exact
FAISS index introduces no approximate-nearest-neighbor recall loss, so these numbers
measure the limitations of the frozen CLIP representation under this benchmark rather
than an ANN indexing trade-off.

### 2. Paraphrase sensitivity is not the main failure mode

The all-five-summary protocol is extremely close to the primary protocol: Recall@10 is
4.831% versus 4.826%, and Recall@100 is 13.740% versus 13.875%. This indicates that the
major issue is not one unlucky wording of the Screen2Words query; the embedding space is
insufficiently discriminative for identifying the exact target UI among a large corpus
of visually and functionally similar screens.

### 3. FAISS is not the current bottleneck

Exact `IndexFlatIP` search takes about 0.12 ms/query in the measured batched benchmark,
while CLIP text embedding takes about 4.39 ms/query. There is therefore no evidence in
M2 that switching to an approximate index would materially improve the end-to-end
system at this stage.

### 4. The benchmark is intentionally strict

Each query has one relevant item: the exact RICO target screen. A visually or
functionally interchangeable screen with a different RICO ID is counted as a negative.
Therefore these metrics should be interpreted as **instance-retrieval** metrics and can
underestimate subjective semantic relevance. The benchmark remains valuable because it
is deterministic and can compare future encoders under the same protocol.

## Artifact hashes from the M2 reference run

### Screen2Words sources

- `screen_summaries.csv`: `534df052d42344d9646f021b5c335ef35dfcc5879a86aedeed67e9d84efa09e5`
- `test_screens.txt`: `276f0aab53d0677777ec80ec4e6ea1fe0db32577b295ec3e2045f873dc951454`

### Primary query artifacts

- `queries.jsonl`: `c21a062b3a43d0602bd9880918159885f4f6e1bf54a95c5f77058458880b0889`
- `query_embeddings.npy`: `ea2ebf09a71cb59059c9f7d0fc40067e74385368386b6ae744a21660f244953e`

### All-five query artifacts

- `queries.jsonl`: `e4d81c8762ba27b9baa03a823986f035c481e8da931bf96cbef56f78ccd1ee7b`
- `query_embeddings.npy`: `0a0f0ce118b7c4543bb8b3e6e27f75e92e6629530f3f2bd4338cb85d72154921`

## M2 conclusion

M2 establishes a reproducible quantitative baseline. The next meaningful intervention
is **M3: encoder modernization and controlled model comparison** while keeping the same
Screen2Words protocol and frozen corpus. Index approximation is deliberately deferred
because M2 shows that encoder quality, not FAISS latency, is the dominant problem.
