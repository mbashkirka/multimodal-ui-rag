# M3 Final Results — Controlled Encoder Comparison

## Status

**M3 COMPLETE.** The frozen M2 Screen2Words protocol was used to compare the legacy OpenAI CLIP baseline against two SigLIP2 Base encoders. The selected M3 encoder is **SigLIP2 Base NaFlex with `max_num_patches=256`**.

## Frozen evaluation protocol

- Corpus: 66,261 canonical RICO screenshots.
- Benchmark: Screen2Words official test split.
- Primary evaluation: 4,310 screens × one human-written summary per screen.
- Retrieval: exact FAISS `IndexFlatIP` over L2-normalized embeddings.
- Search depth: 100.
- Primary selection metric: Recall@10.
- Supporting metrics: Recall@1/5/50/100, MRR@10, nDCG@10, query-embedding latency.
- Hardware: Apple Silicon Mac, PyTorch MPS for encoders; FAISS on CPU in a separate process.

## Primary comparison

| Encoder | Dim | R@1 | R@5 | R@10 | R@50 | R@100 | MRR@10 | nDCG@10 | Text embed ms/query |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CLIP ViT-B/32 QuickGELU (OpenAI) | 512 | 1.14% | 3.25% | 4.83% | 10.70% | 13.87% | 0.0209 | 0.0273 | 4.392 |
| SigLIP2 Base FixRes 224 | 768 | 4.55% | 11.46% | 14.94% | 27.05% | 32.92% | 0.0740 | 0.0917 | 5.713 |
| **SigLIP2 Base NaFlex p256** | **768** | **5.20%** | **12.06%** | **16.22%** | **30.19%** | **37.31%** | **0.0826** | **0.1013** | **5.667** |

## Improvement over the frozen CLIP baseline

For the selected NaFlex model:

- Recall@1: 1.14% → 5.20% (**4.57×**, +4.06 percentage points).
- Recall@5: 3.25% → 12.06% (**3.71×**, +8.82 pp).
- Recall@10: 4.83% → 16.22% (**3.36×**, +11.39 pp).
- Recall@50: 10.70% → 30.19% (**2.82×**, +19.49 pp).
- Recall@100: 13.87% → 37.31% (**2.69×**, +23.43 pp).
- MRR@10: 0.0209 → 0.0826 (**3.96×**).
- nDCG@10: 0.0273 → 0.1013 (**3.72×**).
- Text embedding latency: 4.392 → 5.667 ms/query (**+29.0%**).

The retrieval-quality gain is therefore much larger than the latency cost for this offline/interactive retrieval setting.

## NaFlex versus FixRes

NaFlex also wins the controlled comparison against FixRes at essentially the same query-encoding latency:

- Recall@10: 14.94% → 16.22% (**+8.54% relative**, +1.28 pp).
- Recall@100: 32.92% → 37.31% (**+13.32% relative**, +4.39 pp).
- MRR@10: 0.0740 → 0.0826 (**+11.72% relative**).
- nDCG@10: 0.0917 → 0.1013 (**+10.45% relative**).
- Text embedding latency: 5.713 → 5.667 ms/query (effectively unchanged; NaFlex was ~0.8% faster in this run).

This supports the M3 hypothesis that aspect-ratio-aware NaFlex preprocessing is useful for tall mobile UI screenshots, although this controlled experiment does not isolate aspect ratio as the only causal mechanism.

## Corpus build results

### SigLIP2 Base FixRes 224

- Model: `google/siglip2-base-patch16-224`.
- Backend: `transformers_siglip2`.
- Dimension: 768.
- Vectors: 66,261.
- Device: MPS.
- Batch size: 8.
- Full embedding build: 41m 09s (~26.84 screens/s).
- Embeddings SHA-256: `b96661f0b55c435faeea1251ec31aed5a333ba2006aca0234387076accfc4706`.
- FAISS index SHA-256: `48a7fa1445eb21ea745b3860cf08252301d45d74a1ba798c98745fa090e7e59d`.

### SigLIP2 Base NaFlex p256

- Model: `google/siglip2-base-patch16-naflex`.
- Backend: `transformers_siglip2`.
- `max_num_patches=256`.
- Dimension: 768.
- Vectors: 66,261.
- Device: MPS.
- Batch size: 8.
- Full embedding build: 50m 49s (~21.73 screens/s).
- Embeddings SHA-256: `509cd1bc04e7a7a578fc6f120e0d58f5fea24be9f1781e573377359c76521fdf`.
- FAISS index SHA-256: `a6f0cf7613f1e829252bf38f74750b8bb56cd518e8056aaac2def6bb59e43477`.

Both models use the same canonical image mapping artifact SHA-256:
`f87a45376a601e8903d6c170c439515f2b04fbb4221f2dc09c669a75ece5edda`.

## Benchmark artifact integrity

The primary Screen2Words query set is identical across all compared models:

- Queries SHA-256: `c21a062b3a43d0602bd9880918159885f4f6e1bf54a95c5f77058458880b0889`.
- FixRes query embeddings SHA-256: `c812b61d4da8ec3aa45e46f39f9c98b54d1e640050dfc153f3ff589f82e49c2e`.
- NaFlex query embeddings SHA-256: `021245c8f9863aa8c15efcc1d9a442d9571add3e4a0f36bbad8fffa50601aec4`.

This confirms that the comparison changed the encoder while keeping the primary query set fixed.

## Selection decision

**Selected encoder for subsequent milestones: `google/siglip2-base-patch16-naflex` with `max_num_patches=256`.**

The selection follows the predeclared M3 rule: highest Recall@10 on the frozen 4,310-query Screen2Words primary benchmark. NaFlex also leads on every other reported retrieval-quality metric and has effectively the same text-query latency as FixRes.

## Interpretation

M2 showed that exact FAISS search was not the quality bottleneck. M3 confirms that conclusion: replacing the encoder while leaving the corpus, benchmark and exact index strategy fixed raises Recall@10 by more than 3×. The dominant M3 improvement therefore comes from representation quality rather than search-index approximation.

The benchmark is strict instance retrieval: semantically interchangeable UI screens with a different RICO ID are counted as negatives. Absolute recall values should therefore not be interpreted as subjective semantic relevance.

## Next milestone

M4 will use the selected NaFlex embeddings as the frozen representation and compare exact `IndexFlatIP` against approximate nearest-neighbor indexing (primarily HNSW), measuring the recall/latency/memory trade-off while preserving the exact-index result as ground truth.
