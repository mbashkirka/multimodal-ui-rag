# M7.2.1 Changelog

## Human-audit export reliability hotfix

- Replaced the fragile `file://`-only Blob download workflow with a localhost HTTP save path.
- Added `scripts/m7_human_audit_server.py`, a Python-stdlib-only local server.
- Added `make m7-human-audit-serve`; it opens `audit.html` at `http://127.0.0.1:8765/` and accepts `/save` POSTs.
- Completed scores are validated with the same CSV parser used by `m7-human-report` before atomically writing `human_audit_completed.csv` directly into the audit artifact directory.
- Kept a browser download fallback and added a clipboard fallback for users opening `audit.html` directly with `file://`.
- No retrieval, generation, judge, holdout, or already-generated answer artifacts are changed.
