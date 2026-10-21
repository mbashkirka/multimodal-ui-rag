# M10.0.1 changelog

Quality-gate hotfix only. No ML, retrieval, RAG, API, or serving semantics changed.

- Fixed Ruff `I001` import ordering in `scripts/m10_check_release.py`.
- Fixed Ruff `I001` import ordering in `tests/test_m10_release.py`.
- Synchronized the M10 release-candidate version to `0.10.0.1` across package metadata, release checks, tests, and reference metadata.
- Preserved all M1–M9 runtime behavior and M10 portfolio claims.
