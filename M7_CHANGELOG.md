# M7 changelog

- Bumped project version to `0.7.0`.
- Added generation-evaluation package with deterministic checks and compact visual-judge parsing.
- Added a fresh Screen2Words-dev generation holdout builder excluding persisted M6 evaluation cases.
- Added frozen M6.4 batch answer generation with one reusable MLX VLM process.
- Added same-family visual judge for faithfulness, relevance, unsupported-claim severity, and completeness.
- Added retry-on-format-failure only; semantic scores are never auto-corrected.
- Added slice-aware report generation and a 12-case human-audit CSV template.
- Added M7 make targets and regression tests.
