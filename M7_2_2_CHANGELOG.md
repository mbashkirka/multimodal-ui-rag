# M7.2.2 Changelog — Browser-free human audit

M7.2.2 removes the browser/JavaScript export step from the critical human-audit workflow.

## Added
- `scripts/m7_human_audit_cli.py`: terminal-based blinded audit wizard.
- `make m7-human-audit-cli`: scores the existing 12-case M7.2 audit without a browser.
- macOS Preview auto-open for each evidence screenshot.
- Autosave after every completed case to `human_audit_completed.csv`.
- Resume support after interruption; already scored audit IDs are skipped.
- `--restart` and `--no-open-images` script options.
- Atomic CSV writes and parser validation before every commit.

## Unchanged
- Frozen 30-case M7 generation holdout.
- Generated answers, automated judge results, and failed M7.1 calibration.
- Blinded 12-case audit selection and private audit key.
- M1–M6 retrieval/RAG artifacts and legacy source files.
