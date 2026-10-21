# M7.1 changelog — calibrated generation evaluation

M7.0 automated visual judging saturated at the maximum code `F2 R2 U0 C2` for all 30 holdout cases, including retrieval-miss cases. M7.1 therefore does not accept those scores as evidence of perfect generation quality.

## Changes
- Added a judge calibration/stress-test stage (`make m7-calibrate`).
- Calibration keeps query+screenshot fixed and evaluates four deterministic answer variants: original, irrelevant answer, unsupported sentinel claim, and severe truncation.
- Added explicit sensitivity gates for irrelevance, unsupported claims, and truncation.
- Added score-diversity and corrupted-perfect-score gates.
- `m7-report` now ingests calibration and marks automated F/R/U/C as unreliable when calibration fails or is missing.
- Same-family VLM judging remains diagnostic even after a calibration pass; human audit remains required.
- Version bumped to 0.7.1.
