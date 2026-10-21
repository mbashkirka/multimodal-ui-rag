# M5 Experiment Protocol — Multimodal Fusion + Reranking

## 1. Frozen first-stage retriever

M5 does not change the M3/M4 winner:

```text
RICO 66,261 screens
  -> SigLIP2 Base NaFlex p256
  -> 768-d L2-normalized embeddings
  -> FAISS IndexFlatIP
  -> exact Top-N candidates
```

This preserves the M3 Screen2Words test baseline (Recall@10 = 16.22%) and the M4 decision to keep exact FlatIP at the current corpus scale.

## 2. Split discipline

M5 introduces the official Screen2Words development split for model-selection decisions.

- `dev_screens.txt`: 2,364 screens — used to choose M5 hyperparameters.
- `test_screens.txt`: 4,310 screens — used only after the M5 configuration is frozen.
- One human summary per screen is used for the primary protocol.

This prevents selecting candidate depth or fusion weights directly on the reporting test set.

## 3. Track A — coarse-to-fine multi-resolution reranking

### Stage 1

The frozen p256 exact retriever generates Top-N candidates.

### Stage 2

The same `google/siglip2-base-patch16-naflex` checkpoint re-encodes the corpus with
`max_num_patches=512`. No global p512 FAISS index is built. At query time, only the
stage-1 candidate rows are gathered from the p512 matrix and rescored.

The text representation is unchanged because `max_num_patches` affects image
preprocessing only.

### Rank fusion

Raw p256 and p512 cosine scores are not assumed to be calibrated. Candidate ordering
therefore uses weighted reciprocal-rank fusion (RRF):

```text
score(d) = w / (c + rank_p256(d)) + (1-w) / (c + rank_p512(d))
```

with `c = 60`.

Development grid:

- candidate_k: 20, 50, 100
- base-rank weight: 0.00, 0.25, 0.50, 0.75

Selection objective:

1. maximize Recall@10;
2. tie-break by MRR@10;
3. tie-break by nDCG@10;
4. prefer smaller candidate depth.

The selected configuration is persisted to `selection.json` and applied unchanged to test.

### Candidate ceiling

For each `candidate_k`, M5 reports candidate-oracle recall: the fraction of queries for
which the target is present in the first-stage candidate set. A reranker cannot exceed
this ceiling without changing candidate generation.

## 4. Track B — multimodal visual-exemplar fusion

This is a secondary controlled capability benchmark, not the primary production metric.
For each Screen2Words target, the visual reference is the nearest **different** RICO
screen under the frozen p256 image embedding. The reference is never the target itself.

The fused query is:

```text
q = normalize(alpha * q_text + (1-alpha) * q_reference_image)
```

Development grid:

- text weight: 0.25, 0.50, 0.75, 0.90
- endpoints 0.0 (image-only) and 1.0 (text-only) are reported as baselines but are not eligible as the selected multimodal configuration.

The selected fusion weight is frozen on dev and then evaluated on test.

### Important caveat

The visual exemplar is constructed with knowledge of the target: it is chosen as the
target's nearest non-self neighbor. This deliberately tests whether text can disambiguate
between a target and a highly similar visual exemplar. It is **not** an unbiased estimate
of how often a real user would supply such a reference image.

## 5. Native-process isolation

The p512 embedding build is Torch/Transformers-only. Reranking and fusion benchmarks are
FAISS/NumPy-only. This preserves the macOS ARM OpenMP isolation established in M1/M3.

## 6. M5 decision rule

The primary production decision is based on text-only Screen2Words test Recall@10 after
freezing the reranker configuration on dev. The multimodal track is reported separately
because it evaluates a different query setting.
