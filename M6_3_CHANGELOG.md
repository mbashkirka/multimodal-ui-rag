# M6.3 changelog

M6.2 dev evaluation established that the categorical Qwen evidence selector is structurally reliable but not beneficial for retrieval quality: parse validity 100%, exact evidence selection 50% on retrieval-hit cases, versus 55% for the deterministic retrieval rank-1 baseline. Selector mean latency was 10.89 s and abstention on retrieval misses was 0%.

M6.3 therefore changes the production default to deterministic rank-1 evidence. The frozen M5 retriever remains unchanged. Qwen2.5-VL now receives only the rank-1 screenshot and acts as a single-image grounded verifier/answer generator; it may return `INSUFFICIENT_EVIDENCE`, which the application converts to deterministic abstention. The M6.2 categorical selector remains available explicitly as `--evidence-policy vlm-selector` for ablation and research diagnostics.

Added `scripts/m6_answer_eval.py` and `make m6-answer-eval` to quantify false abstention when the exact Screen2Words target is already rank-1. Abstention on non-exact and retrieval-miss cases is intentionally reported only as diagnostic because semantically similar non-target screens can still support the natural-language request.
