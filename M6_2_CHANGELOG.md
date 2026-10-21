# M6.2 Changelog — Categorical Grounding Selector

M6.1 proved that the full retrieval → VLM path runs on Apple Silicon, but the 40-case Screen2Words test diagnostic exposed a weak output contract: only 12.5% of generations satisfied the large JSON/citation schema, exact evidence selection on retrieval-hit cases was 10%, and mean generation latency was 43.7 s/case.

M6.2 changes the architecture rather than increasing generation length:

- evidence selection is a **categorical decision**: `E1 | E2 | E3 | E4 | NONE`;
- all four candidates are rendered into **one visibly-labelled 2×2 contact sheet**, eliminating implicit multi-image order → evidence-ID mapping;
- selector output is parsed as a short label; the application constructs citations deterministically;
- interactive RAG becomes two-stage while keeping the MLX model loaded once: selector first, then answer generation from only the selected screenshot;
- the selector evaluation no longer asks the model to generate explanations, observations, confidence, and JSON simultaneously;
- M6.2 development evaluation moves to the Screen2Words **dev split**, because the M6.1 failure analysis already observed the test diagnostic;
- evaluation now reports the retrieval-rank-1 baseline on hit cases, allowing the VLM selector to be compared against simply trusting E1;
- no new model, corpus embeddings, FAISS index, or M5 reranker artifacts are required.

The M6.1 test diagnostic is retained as failure-analysis evidence, not as the final M6 result.
