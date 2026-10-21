# M2 — Retrieval Benchmark & Evaluation

## Goal

Measure the quality of the frozen M1 CLIP + FAISS baseline instead of judging retrieval examples
subjectively.

## Benchmark source

M2 uses the Screen2Words dataset released by Google Research and collaborators. Screen2Words
contains human-written concise functional summaries for RICO mobile screenshots. The project uses
the authors' official **test screen split** only for evaluation.

Paper citation:

Bryan Wang, Gang Li, Xin Zhou, Zhourong Chen, Tovi Grossman, and Yang Li. 2021.
*Screen2Words: Automatic Mobile UI Summarization with Multimodal Learning.* UIST 2021.

Dataset repository: `google-research-datasets/screen2words`.

## Protocols

### Primary

- 4,310 test screens.
- One deterministic human summary per screen (the first summary in the released CSV).
- One exact relevant RICO screen per query.
- Every screen therefore receives equal weight.

### Paraphrase robustness

- Same 4,310 test screens.
- All five human summaries per screen.
- 21,550 text queries total.
- Measures sensitivity to alternative natural-language descriptions.

## Metrics

Because each query has one exact relevant screen:

- Recall@1, Recall@5, Recall@10, Recall@50, Recall@100.
- MRR@10.
- nDCG@10.
- Median / mean target rank among hits within the search depth.
- Query-embedding throughput and batch latency.
- FAISS exact-search throughput and batch latency.

## Important interpretation

This is an **instance-retrieval** benchmark. A visually or functionally interchangeable screen is
still considered a negative when it has a different RICO screen ID. As a result, exact-ID Recall
can understate subjective semantic usefulness. This limitation is intentional and should be stated
when presenting the benchmark.

## Leakage policy

Screen2Words is used only for evaluation in M2. The M1 CLIP model is frozen and the FAISS corpus
index is unchanged. Future model selection should use a separate validation protocol; the test
metrics should remain a final report rather than a tuning target.
