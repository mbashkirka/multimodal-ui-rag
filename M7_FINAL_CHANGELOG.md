# M7 FINAL Changelog

M7 is closed as COMPLETE after the blinded human audit.

Final reporting policy:
- keep M6.4 retrieval/generation architecture frozen;
- report deterministic 30-case generation metrics;
- reject same-family automated F/R/U/C judge scores after corruption-calibration failure;
- use the blinded 12-case human audit as the semantic-quality source;
- retain all M7.0/M7.1 judge failures as documented negative ablations.

Reference artifacts added:
- `docs/M7_FINAL_RESULTS.md`
- `docs/m7_reference_run.json`
- `docs/m7_reference_summary.csv`

Observed final human audit:
- faithfulness 91.67%
- relevance 75.00%
- completeness 75.00%
- any unsupported claim 58.33% (7/12)
- major unsupported claim 8.33% (1/12)

M7 identifies the next generation-quality bottleneck as answer discipline: query focus, suppression of minor speculative details, and sentence-boundary control.
