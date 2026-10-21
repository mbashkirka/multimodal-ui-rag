# M6.3 production grounding protocol

## Frozen retrieval

- SigLIP2 Base NaFlex p256 query/corpus representation
- exact FAISS IndexFlatIP candidate generation
- Top-20 candidate pool
- p512 corpus rescoring with the M5 frozen 50/50 RRF configuration

## Production evidence policy

The default policy is deterministic retrieval rank-1 (`E1`). The M6.2 contact-sheet VLM selector is not in the production path because on the Screen2Words dev diagnostic it selected the exact target in 50% of retrieval-hit cases, below the 55% deterministic rank-1 baseline.

## VLM role

Qwen2.5-VL-3B-Instruct-4bit sees only the selected rank-1 screenshot. It either produces a concise grounded answer or emits `INSUFFICIENT_EVIDENCE`. The application converts that sentinel to a deterministic abstention object and constructs citations itself.

## Evaluation

`make m6-answer-eval` runs the single-image answer/verifier stage over the frozen 40-case dev diagnostic. The strict metric is the accept/false-abstain rate when the exact target is already rank-1. Abstention on non-exact or retrieval-miss cases is diagnostic only because exact RICO identity is stricter than semantic relevance.
