# M4 Experiment Protocol — Exact FlatIP vs HNSW

## Objective

M4 isolates the retrieval-index decision after M3 selected `google/siglip2-base-patch16-naflex`
with `max_num_patches=256`. The encoder, corpus, normalization, mapping, and benchmark queries are
frozen. M4 changes only the FAISS index and runtime HNSW search parameter.

## Frozen inputs

- Corpus: 66,261 canonical RICO screenshots.
- Encoder: SigLIP2 Base NaFlex, `max_num_patches=256`.
- Embedding dimension: 768, float32, L2-normalized.
- Exact oracle: `IndexFlatIP` built in M3.
- Benchmark: Screen2Words official test split, primary protocol, 4,310 queries.
- Query vectors: the persisted M3 NaFlex query embeddings are reused byte-for-byte.

Reusing query embeddings is deliberate: M4 measures index approximation and search latency only.
Encoder latency is excluded from this milestone.

## HNSW grid

Build three `IndexHNSWFlat` indexes over the identical frozen embeddings:

| Variant | M | efConstruction |
|---|---:|---:|
| HNSW-M16 | 16 | 200 |
| HNSW-M32 | 32 | 200 |
| HNSW-M48 | 48 | 200 |

For each index, sweep runtime `efSearch` over:

`16, 32, 64, 128, 256`.

The metric remains inner product over L2-normalized vectors, therefore the ranking target is the
same cosine-similarity ordering used by the exact FlatIP oracle.

## Metrics

### 1. ANN fidelity to the exact oracle

For every query, compute exact Top-10 using FlatIP and compare HNSW Top-10 against it.

- exact-neighbor Recall@1
- exact-neighbor Recall@5
- exact-neighbor Recall@10
- exact Top-1 agreement

Exact-neighbor Recall@10 is the primary approximation-quality metric. This avoids a subtle error:
an approximate index can occasionally increase Screen2Words target Recall@10 simply by dropping
higher-scoring exact neighbors. That is approximation noise, not proof of a better semantic model.

### 2. Downstream Screen2Words retention

Using the same single relevant RICO screen per query:

- task Recall@1
- task Recall@5
- task Recall@10
- MRR@10
- nDCG@10
- task Recall@10 retention relative to FlatIP

### 3. Search performance

The comparison fixes FAISS to one CPU thread for algorithmic reproducibility.

- online single-query p50 / p95 latency, Top-10
- batch-256 median throughput (queries/s), Top-10
- speedup relative to FlatIP under the same process/thread settings
- serialized index size in MiB
- HNSW build time

M3 latency numbers should not be compared directly with M4 latency numbers because M3 used the
library's normal thread configuration and included a different measurement purpose. M4 remeasures
FlatIP beside HNSW under identical conditions.

## Selection rule

HNSW is eligible for selection only if all three constraints are met:

1. exact-neighbor Recall@10 >= 0.95;
2. Screen2Words task Recall@10 retention >= 0.98 of FlatIP;
3. single-query p50 speedup >= 1.20x.

Among eligible HNSW configurations, select the fastest single-query configuration, using neighbor
recall and smaller index size as tie-breakers.

If no HNSW configuration satisfies the constraints, M4 selects FlatIP. This is an intentional and
valid outcome: at only 66k vectors, an ANN graph may not justify its extra complexity/memory.

## Commands

```bash
make m4-smoke
make m4-build
make m4-benchmark
```

The final comparison is written to:

```text
artifacts/evaluation/m4-exact-vs-hnsw/comparison.json
artifacts/evaluation/m4-exact-vs-hnsw/comparison.csv
```

## macOS native isolation

All M4 build/benchmark scripts are FAISS + NumPy only. They do not import Torch, OpenCLIP, or
Transformers, preserving the native-process isolation established after the Apple Silicon OpenMP
crashes found during M1/M3.
