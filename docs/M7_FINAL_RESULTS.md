# M7 Final Results — Generation Evaluation Framework

## Status

**M7 COMPLETE.** The production retrieval/RAG architecture from M6.4 remains frozen. M7 evaluates generated answers rather than changing retrieval or generation architecture.

## Evaluation design

M7 deliberately separates three evaluation layers:

1. **Deterministic generation contracts** over 30 frozen Screen2Words dev holdout cases, stratified into 10 `rank1_exact`, 10 `top4_nonrank1`, and 10 `miss` cases. All 40 M6 prompt-design/evaluation cases were excluded before sampling.
2. **Automated visual judge diagnostic** using the same local Qwen2.5-VL-3B family as generation.
3. **Blinded human audit** over 12 stratified cases (4 per retrieval slice), where the evaluator sees only query, E1 screenshot, and generated answer. Retrieval slice, target identity, and automated judge scores remain hidden during scoring.

## Deterministic generation results (30 cases)

| Metric | Result |
| --- | ---: |
| Citation validity | **100.00%** |
| Non-empty answer rate | **100.00%** |
| Clean sentence-ending rate | **93.33%** |
| Generation mean latency | **10.14 s** |
| Generation median latency | **10.06 s** |
| Generation p95 latency | **13.47 s** |

The only structural issue observed was truncation: 2/30 answers did not end at a sentence boundary. Both occurred in the `rank1_exact` slice, where clean-ending rate was 80% (8/10); the other two slices were 100%.

## Automated judge rejection

The initial same-family automated judge assigned the identical maximum score `F2 R2 U0 C2` to all 30 original cases. M7.1 therefore introduced deterministic corruption calibration using deliberately irrelevant, unsupported-sentinel, and truncated answer variants.

Calibration result:

| Calibration metric | Result |
| --- | ---: |
| Contract validity | 100% |
| Irrelevant detection | **0%** |
| Unsupported-claim detection | **0%** |
| Truncation detection | **0%** |
| Perfect score on corrupted answers | **100%** |
| Unique judge score codes | **1** |
| Calibration gate | **FAIL** |

Therefore automated Qwen2.5-VL F/R/U/C scores are **not used as quality claims**. This is retained as an evaluator-failure ablation.

## Blinded human audit results (12 cases)

Human scores use a 0–2 rubric and normalized means for faithfulness, relevance, and completeness. Unsupported-claim rates count any case with unsupported severity >0; major unsupported claims are severity 2.

| Human metric | Result |
| --- | ---: |
| Faithfulness | **91.67%** |
| Relevance | **75.00%** |
| Completeness | **75.00%** |
| Cases with any unsupported claim | **58.33% (7/12)** |
| Cases with a major unsupported claim | **8.33% (1/12)** |

### Interpretation

The generator is generally visually grounded: the high faithfulness score and low major-unsupported rate indicate that gross hallucination is uncommon on this small blinded sample. However, the 58.33% any-unsupported rate shows frequent **minor speculative or unnecessary details**. Relevance and completeness at 75% indicate that the model often spends tokens on secondary visual details instead of focusing tightly on what the query asks, and sometimes omits useful request-relevant information.

The actionable generation bottleneck after M7 is therefore **answer discipline**, not retrieval architecture: shorter query-focused outputs, stronger suppression of speculative details, and explicit sentence-boundary control should be considered in future generation optimization. M7 itself does not tune those behaviors; it establishes the evaluation framework and records the baseline.

## Release-quality claims

The defensible M7 claims are:

- deterministic provenance/citation correctness: 100% on the 30-case holdout;
- non-empty generation: 100% on the same holdout;
- clean sentence ending: 93.33%;
- local Apple-Silicon generation latency: mean 10.14 s, p95 13.47 s for this protocol;
- blinded 12-case human audit: 91.67% faithfulness, 75% relevance, 75% completeness;
- any unsupported detail in 7/12 cases, but major unsupported claims in only 1/12;
- same-family automated VLM judging failed corruption calibration and is excluded from semantic quality reporting.

## Limitations

The human semantic audit contains only 12 cases and a single evaluator, so these percentages should be treated as a compact release audit rather than a statistically precise population estimate. Future work can expand the audit and measure inter-rater agreement without changing the M7 protocol.
