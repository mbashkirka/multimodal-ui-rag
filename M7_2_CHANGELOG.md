# M7.2 changelog

- Rejected same-family Qwen2.5-VL F/R/U/C scores as release quality metrics after M7.1 corruption calibration failed every sensitivity gate.
- Added deterministic, stratified, blinded 12-case human audit bundle.
- Added self-contained local HTML reviewer UI with no external dependencies and CSV export.
- Hidden retrieval slice, target filename, and automated judge score during human scoring to reduce evaluator bias.
- Added strict CSV validation and post-audit aggregation by retrieval slice.
- Added `make m7-human-audit` and `make m7-human-report`.
- Project version bumped to 0.7.2.
