# M1 Changelog

## Reproducible Legacy Baseline

### Preserved
- Original `retrieval_baseline.py` -> `legacy/retrieval_baseline.py`
- Original `id_utils.py` -> `legacy/id_utils.py`
- Original `RICO_installation.sh` -> `legacy/RICO_installation.sh`
- Original `image_index_map.json` -> `data_storage/image_index_map.json`

### Changed
- Replaced mixed `clip` + Hugging Face tokenizer logic with one OpenCLIP stack.
- Added Apple Silicon MPS detection with CPU fallback.
- Removed Linux-only setup from the primary workflow.
- Added macOS bootstrap and cross-platform RICO downloader.
- Added deterministic batched CLIP indexing using exact `IndexFlatIP`.
- Added artifact manifest and SHA-256 checks.
- Added text, image, and weighted multimodal retrieval.
- Added CLI, smoke test, unit tests, and lint configuration.

### Deferred to M2+
- Retrieval benchmark and Recall@K / MRR / nDCG.
- Alternative embedding models and ANN/HNSW.
- Reranking, VLM generation, API, Docker.

## M1.1 macOS data-integrity hotfix

Observed against the official RICO archive on macOS:

- Original legacy mapping rows: **66,271**.
- Official unique screenshot JPEGs: **66,261**.
- The difference is exactly 10 historical Finder-style aliases such as
  `68471 (1).jpg`; each has a canonical counterpart such as `68471.jpg`.

Changes:

- Preserve `data_storage/image_index_map.json` byte-for-byte as legacy evidence.
- Canonicalize only aliases whose canonical counterpart exists in the same mapping.
- Index **66,261 unique screenshots** rather than duplicating ten identical vectors.
- `make data` now validates an already-extracted canonical corpus before downloading
  or rescanning the 6 GB archive.
- A successful validation removes the stale archive unless `--keep-archive` is used.
- Added mapping-integrity tests for the 66,271 -> 66,261 canonicalization contract.

## M1.2 deterministic retrieval smoke artifact

Changes:

- Added `make index-smoke` to build an exact `IndexFlatIP` artifact over the first
  100 canonical RICO screenshots using a conservative macOS/MPS batch size of 16.
- The smoke artifact has its own name and cannot overwrite the future full-corpus
  baseline artifact.
- Added `make verify-smoke` for an end-to-end artifact contract test.
- Verification checks manifest hashes, vector count, 512-d dimension, FAISS/map
  consistency, and image self-retrieval.
- `71018.jpg` must retrieve itself at rank 1 with cosine similarity >= 0.999.
- Added a text-query Top-5 printout after the deterministic self-retrieval assertion.
- Renamed full build/search Make targets to `index-full` and `search-full` to avoid
  accidentally starting a 66k-screen build when only a smoke test was intended.

## M1.3 — macOS ARM native-runtime isolation

- Fixed `Segmentation fault: 11` during indexing on Apple Silicon.
- Root cause pattern: PyTorch/OpenCLIP and `faiss-cpu` can load conflicting OpenMP runtimes when imported in one process on macOS ARM.
- Corpus embedding and FAISS index construction now run in separate Python interpreters.
- Query encoding and FAISS search are likewise isolated, so retrieval does not reintroduce the same crash.
- Added persisted `.npy` embedding artifacts and an embedding-stage metadata contract.
- Added `test_native_import_isolation.py` to guard the process boundary.
- `make setup` now reuses an existing compatible `.venv` and performs PyTorch/OpenCLIP and FAISS diagnostics in separate interpreters.

## M1.4 — resumable full-corpus artifact

- Added resumable corpus embedding for the 66,261-screen RICO build.
- Partial state is persisted as `artifacts/embeddings/.<artifact>.partial.npy` plus
  `artifacts/metadata/<artifact>-embedding-progress.json` after every flushed batch.
- Re-running `make index-full` resumes from the last completed batch when the artifact
  contract (model, weights, source mapping hash, corpus size, dtype) is unchanged.
- Added explicit `make index-full-restart` for an intentional destructive rebuild.
- Added `make verify-full`, which validates full-corpus counts, SHA-256 hashes,
  native-process isolation, resumable-build metadata, image self-retrieval, and a
  text-retrieval Top-5 query.
- Added separate `clean-smoke`; `clean-artifacts` also removes partial/progress state.

## M1 final reference run — 2026-08-19

- Full 66,261-screen RICO corpus indexed successfully on Apple MPS.
- 10/10 tests passed.
- Full build completed in 22:14 (4,142 batches at batch size 16).
- Exact `IndexFlatIP` artifact verified end-to-end.
- `71018.jpg` self-retrieval returned rank 1 with score `1.000000`.
- Reference artifact hashes and sizes recorded in `docs/M1_FINAL_RESULTS.md` and `docs/m1_reference_run.json`.
- M1 is complete; M2 begins retrieval-quality evaluation.
