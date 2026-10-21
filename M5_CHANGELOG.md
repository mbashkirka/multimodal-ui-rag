# M5 Changelog — Multimodal Fusion + Coarse-to-Fine Reranking

## Goal

Improve the frozen M4 retrieval stack without changing the production first-stage index.
M5 separates two questions:

1. Can higher-resolution NaFlex representations reorder a small candidate set better than the p256 retriever?
2. When a user provides text plus a visually similar reference screen, does multimodal fusion improve retrieval?

## Changes

- Project version: `0.5.0`.
- Added official Screen2Words `dev_screens.txt` download and validation (2,364 screens).
- Generalized Screen2Words benchmark preparation to explicit `dev`/`test` split names.
- Added p512 multi-resolution reranker path using the same `google/siglip2-base-patch16-naflex` checkpoint.
- Added weighted reciprocal-rank fusion (RRF) between p256 stage-1 rank and p512 stage-2 rank.
- Added dev-only selection of `candidate_k` and RRF weight; test evaluation consumes the frozen dev selection.
- Added controlled multimodal visual-exemplar benchmark using the nearest non-self RICO screen as the reference image.
- Added dev-only selection of text/image fusion weight; test uses the frozen dev weight.
- Preserved macOS native isolation: M5 reranking/fusion evaluation scripts are FAISS+NumPy only; no Torch is loaded in those processes.
- Added M5 regression tests for RRF, fusion normalization, non-self reference selection, dev/test protocol separation, and native-import isolation.

## Frozen inputs

- Corpus: 66,261 canonical RICO screenshots.
- Stage-1 encoder: SigLIP2 Base NaFlex, `max_num_patches=256`.
- Stage-1 index: FAISS `IndexFlatIP` exact search.
- Primary metric: Screen2Words exact-instance Recall@10.
- Test split is not used for M5 hyperparameter selection.
