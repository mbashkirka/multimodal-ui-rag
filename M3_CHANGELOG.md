# M3 Changelog — Controlled Encoder Comparison

## Goal

Replace the legacy CLIP encoder while freezing the M2 evaluation protocol.  M3 changes only the dual encoder; RICO, canonical mapping, Screen2Words test split, FAISS `IndexFlatIP`, search depth, and metrics stay fixed.

## Candidates

1. `google/siglip2-base-patch16-224` — SigLIP2 Base fixed-resolution 224.
2. `google/siglip2-base-patch16-naflex` — SigLIP2 Base NaFlex with `max_num_patches=256`.

The two candidates are deliberately from the same SigLIP2 Base family.  The second candidate tests whether aspect-ratio-preserving NaFlex preprocessing improves retrieval for tall mobile screenshots.

## Engineering changes

- Added a backward-compatible encoder factory.
- Existing M1/M2 CLIP manifests without `backend` still resolve to `open_clip`.
- Added Transformers SigLIP2 encoder with L2-normalized text/image features.
- Preserved macOS native-process isolation: encoder/PyTorch and FAISS never share one Python interpreter.
- Preserved resumable corpus embedding builds.
- Added generic self-retrieval verification for any encoder.
- Added controlled comparison report against the frozen M2 reference run.

## Experiment protocol

Run both 100-screen smoke checks first.  Only after both pass, build full 66,261-screen indexes and run the *primary* Screen2Words benchmark.  Do not choose a winner from smoke retrieval scores.  The winner is selected by primary benchmark metrics, with Recall@10 as the main ranking metric and Recall@100 / MRR@10 / nDCG@10 as supporting metrics.  Query-embedding latency is reported as a cost tradeoff.

## M3.1 compatibility patch

The initial Transformers 5.15.1 smoke run exposed a return-type mismatch: `get_image_features()` / `get_text_features()` return pooled model-output objects rather than raw tensors. M3.1 extracts `pooler_output`, makes SigLIP2 lowercase + 64-token text preprocessing explicit, and adds regression tests. See `M3_1_CHANGELOG.md`.
