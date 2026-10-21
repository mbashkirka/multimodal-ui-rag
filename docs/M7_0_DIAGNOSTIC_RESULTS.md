# M7.0 diagnostic results

Observed on the 30-case fresh Screen2Words-dev generation holdout:

- 10 rank1-exact, 10 target-elsewhere-top4, 10 retrieval-miss cases.
- citation-valid rate: 100%.
- non-empty answer rate: 100%.
- clean sentence ending: 93.33% overall; 80% on rank1-exact.
- generation latency: mean 10.14 s, median 10.06 s, p95 13.47 s.
- automated judge contract validity: 100%; retry rate: 0%.
- automated judge emitted `F2 R2 U0 C2` for every one of the 30 cases.

The all-perfect, zero-variance judge result is treated as evaluator saturation rather than as a credible claim of 100% faithfulness/relevance/completeness. M7.1 adds mandatory corruption calibration before automated scores may be interpreted.
