# M9.1 Changelog — Quality Gate Repair

- Version `0.9.1`.
- Fixed all 44 Ruff violations surfaced by the first real `make m9-quality` run on macOS.
- No retrieval, reranking, RAG, API contract, worker topology, timeout, metrics, or model behavior changes.
- Applied import ordering/modern typing cleanups and removed unused imports/variables.
- Preserved broad exception handling only at deliberate evaluation/process boundaries with targeted `BLE001` suppressions.
- Reworked worker boundary error serialization to avoid nested blind exception handling and `except: pass`.
- Narrowed the legacy human-audit HTTP validation catch to expected I/O/type/value failures.
- Corrected exception types in JSON schema parsing (`TypeError` for invalid container types).
- Regression suite remains green: 108 passed, 1 environment-dependent FAISS skip in the build environment.
- `compileall` passes.
- Legacy source hashes remain unchanged.
