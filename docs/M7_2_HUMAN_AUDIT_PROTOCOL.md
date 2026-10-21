# M7.2 — Blinded human audit protocol

M7.1 proved that the same-family Qwen2.5-VL visual judge is saturated: deterministic corrupted answers received the same perfect score as originals. Automated F/R/U/C scores are therefore excluded from semantic quality claims.

M7.2 keeps the frozen 30-case M7 holdout and the already-generated answers. It creates a **12-case stratified human audit** (four cases from each retrieval slice) and blinds the reviewer to:

- retrieval slice;
- exact target filename;
- automated judge score.

The reviewer sees only the user query, the production E1 screenshot, and the generated answer. This matches the information needed to assess generation quality without leaking retrieval ground truth.

## Human rubric (0–2)

### Faithfulness
- 0: material claims conflict with or are unsupported by the screenshot;
- 1: mostly supported, but at least one material claim is uncertain/unsupported;
- 2: all material claims are directly supported.

### Relevance
- 0: does not address the request;
- 1: partially addresses it or focuses on secondary details;
- 2: directly addresses the request using visible evidence.

### Unsupported-claim severity
- 0: no material unsupported claim;
- 1: one minor unsupported/speculative claim;
- 2: one or more major unsupported/speculative claims.

### Completeness
- 0: misses the main visible information needed for the request;
- 1: covers the main point but omits useful visible details;
- 2: covers the important visible information relevant to the request.

## Blinding and reproducibility

`m7-human-audit` creates a deterministic, shuffled audit bundle. The public audit manifest and HTML omit retrieval slice, target filename, and automated judge score. A separate `audit_key.json` is used only after scoring to aggregate by retrieval slice.

The completed audit CSV is the semantic quality source for M7 release reporting. Automated judge scores remain archived only as a failed evaluator ablation.
