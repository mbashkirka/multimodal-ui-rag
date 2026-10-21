# M6.4 Changelog — Generation-Only Grounded VLM

## Decision
M6.4 removes the VLM from evidence acceptance/rejection in the production path.

The decision is empirical:
- M6.2 categorical selector: 50% exact evidence selection on retrieval-hit dev cases vs 55% for deterministic retrieval rank-1; mean selector latency 10.89 s.
- M6.3 verifier: only 1/11 exact-rank1 cases accepted; 90.91% false-abstain rate; mean answer latency 9.19 s.

Therefore the frozen M5 retriever/reranker owns evidence selection. The local VLM receives the selected rank-1 screenshot and performs grounded description/answer generation only. It cannot rerank or reject the evidence.

## Changes
- production architecture: deterministic rank-1 -> single-image grounded VLM answer;
- removed `INSUFFICIENT_EVIDENCE` from the production answer prompt;
- application always attaches the selected E1 citation deterministically for rank1 policy;
- `vlm-selector` remains available as a diagnostic ablation;
- retained legacy `insufficient=False` field in provider output only for compatibility;
- fixed an unrelated latent `vlm-generate` CLI argument mismatch;
- project version bumped to 0.6.4.
