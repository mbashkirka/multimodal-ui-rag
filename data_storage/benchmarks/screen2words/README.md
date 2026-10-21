# Screen2Words benchmark data

M2 uses the **official Screen2Words test split** as an external text-to-UI retrieval benchmark.
The source data is not bundled in the project archive; `make benchmark-data` downloads two small
files directly from the archived `google-research-datasets/screen2words` repository:

- `screen_summaries.csv` — five human-written functional summaries per annotated RICO screen.
- `split/test_screens.txt` — the official 4,310-screen test split.

The benchmark never trains or tunes the retrieval model on these summaries. It uses the existing
M1 CLIP + FAISS artifact unchanged.

Primary protocol: one summary per test screen (4,310 queries), exact RICO screen identity as the
single relevant item. Paraphrase protocol: all five summaries (21,550 queries).

See `docs/M2_BENCHMARK_PROTOCOL.md` for evaluation details and citation information.
