# M5 Final Results — Coarse-to-Fine Reranking + Multimodal Fusion

## Status

**COMPLETE. Production text-only retrieval adds the dev-selected p512 reranker; multimodal fusion remains a separate controlled capability mode.**

M5 froze the M4 first-stage retriever (`google/siglip2-base-patch16-naflex`, `max_num_patches=256`, exact `IndexFlatIP`) and evaluated two additions:

1. a coarse-to-fine p512 candidate reranker selected on the official Screen2Words dev split;
2. a controlled text + visual-exemplar fusion mode selected on dev and evaluated once on test.

## Frozen first-stage baseline

| Split | R@1 | R@5 | R@10 | R@100 | MRR@10 | nDCG@10 |
|---|---:|---:|---:|---:|---:|---:|
| Screen2Words dev | 5.50% | 13.41% | 17.60% | 36.55% | 0.0889 | 0.1094 |
| Screen2Words test | 5.20% | 12.06% | 16.22% | 37.31% | 0.0826 | 0.1013 |

The dev split contains 2,364 screens; the frozen test split contains 4,310 screens.

## Track A — p512 coarse-to-fine reranking

### Full p512 corpus artifact

The same Base NaFlex checkpoint was re-encoded with `max_num_patches=512`:

- vectors: 66,261
- dimension: 768
- dtype: float32
- device: MPS
- build time: 1:45:47
- embedding SHA-256: `a162218b1c81bff934ff4421a219dead732363137677934d4708cf3e9a3e61c6`
- mapping SHA-256: `f87a45376a601e8903d6c170c439515f2b04fbb4221f2dc09c669a75ece5edda`

No second global FAISS index is required: the p512 matrix is used only to rescore first-stage candidates.

### Dev selection

The pre-registered grid covered candidate depths 20/50/100 and p256 base-rank weights 0.00/0.25/0.50/0.75 under weighted RRF (`c=60`). The selected configuration was:

```text
candidate_k = 20
base_weight = 0.50
reranker_weight = 0.50
RRF constant = 60
```

Dev result:

| System | Candidate oracle | R@1 | R@5 | R@10 | MRR@10 | nDCG@10 |
|---|---:|---:|---:|---:|---:|---:|
| p256 baseline | — | 5.50% | 13.41% | 17.60% | 0.0889 | 0.1094 |
| selected p512 reranker | 22.34% | 5.80% | 13.75% | **18.57%** | 0.0936 | 0.1152 |

The selected configuration improved dev R@10 by +0.97 percentage points (+5.5% relative).

### Frozen test result

| System | R@1 | R@5 | R@10 | MRR@10 | nDCG@10 |
|---|---:|---:|---:|---:|---:|
| p256 baseline | 5.20% | 12.06% | 16.22% | 0.0826 | 0.1013 |
| **p256 Top-20 -> p512 RRF** | **5.22%** | **12.55%** | **16.73%** | **0.0846** | **0.1041** |

The test gain is modest but consistent with dev:

- R@10: 16.22% -> 16.73% (+0.51 pp, +3.1% relative)
- R@5: 12.06% -> 12.55% (+0.49 pp, +4.1% relative)
- MRR@10: 0.0826 -> 0.0846 (+2.4% relative)
- nDCG@10: 0.1013 -> 0.1041 (+2.8% relative)

Measured candidate rescoring time was ~0.0209 ms/query in the test benchmark (FAISS/NumPy-only benchmark process). The principal cost is the additional p512 corpus matrix, approximately one extra 768-d float32 vector per screen (~194 MiB for 66,261 screens).

### Production decision

The text-only production path becomes:

```text
text query
   -> SigLIP2 Base NaFlex p256
   -> exact FlatIP Top-20
   -> gather p512 rows for those 20 candidates
   -> 50/50 weighted RRF between p256 and p512 ranks
   -> final Top-K
```

The p256-only path remains a valid low-memory fallback. The reranker is adopted because its frozen-test improvement is positive across R@5, R@10, MRR@10 and nDCG@10, while online rescoring overhead is negligible at Top-20.

## Track B — controlled multimodal visual-exemplar fusion

### Dev selection

For each target, the benchmark constructs the nearest **different** RICO screen under frozen p256 image embeddings as a visual exemplar. No self-reference is allowed.

Dev diagnostics:

- references: 2,364
- self references: 0
- median target/reference cosine: 0.8795
- selected weight: 0.50 text / 0.50 image
- selected R@10: 79.31%

| Text weight | Image weight | R@1 | R@5 | R@10 | R@100 | MRR@10 | nDCG@10 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 1.00 | 0.25% | 65.52% | 73.60% | 88.07% | 0.2939 | 0.4041 |
| 0.50 | 0.50 | 1.14% | **71.57%** | **79.31%** | **92.43%** | 0.3262 | 0.4428 |
| 1.00 | 0.00 | 5.50% | 13.41% | 17.60% | 36.55% | 0.0889 | 0.1094 |

### Frozen test result

- references: 4,310
- self references: 0
- median target/reference cosine: 0.8802
- frozen weight: 0.50 text / 0.50 image

| System | R@1 | R@5 | R@10 | R@100 | MRR@10 | nDCG@10 |
|---|---:|---:|---:|---:|---:|---:|
| p256 text-only | 5.20% | 12.06% | 16.22% | 37.31% | 0.0826 | 0.1013 |
| **text + visual exemplar (0.5/0.5)** | 1.53% | **69.74%** | **77.05%** | **91.04%** | **0.3193** | **0.4321** |

This result demonstrates strong complementarity between text and a highly related visual reference, especially at R@5/R@10. The lower R@1 is expected in this construction: the exemplar itself is a different screen and is intentionally extremely similar to the target.

### Important interpretation boundary

The 77.05% R@10 number is **not** an unbiased production estimate and is not comparable as a replacement for the 16.73% text-only production metric. The exemplar is target-conditioned: it is selected using knowledge of the target's nearest non-self neighbor. This track is therefore an upper-bound / capability diagnostic showing what the retriever can do when a user supplies a highly relevant visual reference.

For a future real multimodal product benchmark, visual exemplars must come from an external user-like process without target knowledge.

## Final M5 comparison

| System | R@1 | R@5 | R@10 | MRR@10 | nDCG@10 | Interpretation |
|---|---:|---:|---:|---:|---:|---|
| NaFlex p256 text-only | 5.20% | 12.06% | 16.22% | 0.0826 | 0.1013 | frozen M4 baseline |
| **p512 coarse-to-fine reranker** | **5.22%** | **12.55%** | **16.73%** | **0.0846** | **0.1041** | selected text-only production path |
| text + visual exemplar | 1.53% | 69.74% | 77.05% | 0.3193 | 0.4321 | controlled multimodal diagnostic |

## Portfolio/interview takeaway

M5 separates two often-confounded improvements. Multi-resolution reranking provides a small but honest text-only gain under strict dev/test discipline. Multimodal fusion demonstrates a much larger capability gain when a relevant reference image is available, while explicitly documenting that the benchmark constructs that reference with target knowledge. The production decision therefore adopts only the p512 reranker and reports visual fusion separately rather than presenting the 77.05% diagnostic result as an unbiased production metric.
