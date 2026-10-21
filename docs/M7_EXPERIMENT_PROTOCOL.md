# M7 — Generation Evaluation Framework

## Objective

M7 freezes the M6.4 production architecture and evaluates generated answers rather than changing retrieval or generation policy.

Frozen path:

`query -> SigLIP2 NaFlex p256 -> FlatIP -> Top-20 -> p512 RRF -> deterministic E1 -> Qwen2.5-VL grounded answer`

## Holdout construction

M7 uses Screen2Words **dev** because the official test split has already been observed in earlier milestones. To avoid prompt-design leakage within dev, every case persisted in the M6.2/M6.3 evaluation set is excluded before M7 selection.

The 30-case M7 holdout is stratified by frozen retrieval outcome:

- 10 `rank1_exact`: exact Screen2Words target is E1;
- 10 `top4_nonrank1`: target is in E2–E4 while production still answers from E1;
- 10 `miss`: target is absent from Top-4.

This makes retrieval-conditioned generation behavior visible instead of collapsing all failures into one score.

## Layer 1 — deterministic contracts

Measured without a model judge:

- non-empty answer rate;
- citation validity (`selected_evidence_id == E1`, exactly one `E1` citation, non-abstaining M6.4 contract);
- word count;
- clean sentence-boundary rate as a simple truncation diagnostic;
- generation latency mean/median/p95.

## Layer 2 — automated visual judge

The local judge sees the query, the generated answer, and the exact E1 screenshot used for generation. It emits only a compact categorical code:

`F2 R2 U0 C2`

where each axis is in `{0,1,2}`:

- **F — faithfulness**: are material claims visually supported?
- **R — relevance**: does the answer address the request?
- **U — unsupported-claim severity**: 0 none, 1 minor, 2 major;
- **C — completeness**: does the answer cover important visible aspects relevant to the request?

The compact code avoids the JSON-serialization failure mode found in M6.1. One deterministic retry is allowed only when the output contract is malformed.

### Important limitation

Generation and automated judging use the same local Qwen2.5-VL model family for Mac feasibility. Therefore judge scores are **diagnostic, not independent ground truth**. They must not be presented as human-equivalent evaluation.

## Layer 3 — human audit

`m7-report` exports `human_audit.csv`, stratified across the three retrieval slices. Human columns are intentionally blank and use the same 0–2 rubric as the automated judge. This supports later judge calibration/inter-rater analysis without changing the frozen model output.

## Selection / reporting discipline

M7 is an evaluation milestone, not a tuning loop. The initial 30-case holdout and rubric should be frozen before inspecting judge results. If generation prompts are changed later, a new holdout/version must be created rather than silently reusing M7 results.
