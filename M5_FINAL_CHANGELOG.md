# M5 FINAL Changelog

- Marked M5 complete.
- Frozen the dev-selected text-only production configuration: p256 FlatIP Top-20 -> p512 weighted RRF (`base_weight=0.50`, `c=60`).
- Recorded frozen-test improvement: Recall@10 16.22% -> 16.73%.
- Recorded p512 full-corpus embedding artifact SHA-256 and build duration.
- Recorded the 0.50/0.50 text+visual-exemplar diagnostic result (test Recall@10 77.05%) with an explicit target-conditioned benchmark caveat.
- Added `docs/M5_FINAL_RESULTS.md`, `docs/m5_reference_run.json`, and `docs/m5_reference_comparison.csv`.
- Updated README milestone status and production retrieval stack.
- No production source code was changed after the successfully executed M5 experiment.
