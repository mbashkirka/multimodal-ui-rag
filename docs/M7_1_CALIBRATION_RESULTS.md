# M7.1 calibration results

Observed on the six-source-case / 24-judgement deterministic corruption calibration:

- contract validity: **100%**;
- irrelevant detection: **0%**;
- unsupported-sentinel detection: **0%**;
- truncation detection: **0%**;
- corrupted answers receiving perfect `F2 R2 U0 C2`: **100%**;
- unique score codes: **1**;
- calibration result: **FAIL**;
- judge latency: mean **8.93 s**, median **8.74 s**, p95 **12.60 s**.

The same-family Qwen2.5-VL judge is therefore treated as saturated and unreliable for semantic generation quality claims. Its 30-case all-perfect F/R/U/C table is retained only as a failed evaluator ablation. M7.2 makes blinded human audit the semantic reporting source.
