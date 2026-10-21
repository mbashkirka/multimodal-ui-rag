# Interview Guide

## One-line resume description

**Built a production-style multimodal UI RAG system over 66k RICO screenshots, improving exact Screen2Words Recall@10 from 4.83% to 16.73%, serving warm search in 171 ms with isolated persistent workers, and validating generation through calibrated evaluation plus blinded human audit.**

## 30-second version

> I started from a legacy CLIP + FAISS UI-search prototype and treated it as an end-to-end ML systems project. I built a reproducible RICO pipeline and Screen2Words benchmark, replaced CLIP with SigLIP2 NaFlex, tested exact versus HNSW indexing, added a p512 coarse-to-fine reranker, and integrated a local Qwen vision-language model for grounded answers. I also removed VLM selector/verifier designs when experiments showed they hurt quality. The final system is a FastAPI service with isolated persistent workers, metrics, recovery, backpressure, CI, and a human-audited generation evaluation.

## 90-second version

> The key idea was to make every architectural change earn its complexity. The initial CLIP baseline achieved 4.83% Recall@10 on exact Screen2Words instance retrieval. Under the same frozen benchmark, SigLIP2 NaFlex p256 reached 16.22%, and p512 Top-20 reranking improved that to 16.73%. I tested HNSW, but rejected it because the best configuration only reached 92.88% exact-neighbor Recall@10 against a 95% gate, even though it was faster.
>
> For RAG, I tried several VLM roles. A contact-sheet selector was worse than simply taking rank one—50% versus 55% exact selection on hit cases—and a verifier falsely abstained on 90.9% of cases where the exact target was already rank one. So the final design is deterministic retrieval/provenance with the VLM used only for grounded interpretation. I also stress-tested the automated VLM judge; it failed to detect deliberately corrupted answers, so I discarded its perfect scores and used a blinded human audit instead.
>
> On the systems side, the original API reloaded models per request and search took about 15 seconds. Persistent isolated workers reduced warm search to 171 milliseconds while preserving macOS native-library isolation. M9 added structured logs, metrics, worker restart, timeouts, CI, and controlled 429 backpressure.

## Five technical decisions worth discussing

### 1. Why exact FAISS instead of HNSW?

Because the measured bottleneck was not exact-search latency, and the tested HNSW configurations failed the required exact-neighbor recall gate. The final choice optimizes for correctness under the proven corpus size rather than using ANN by default.

### 2. Why p256 retrieval plus p512 reranking?

NaFlex p512 is more expensive. Using p256 globally and p512 only on Top-20 candidates gives a coarse-to-fine design that adds visual detail without building/searching another full p512 index.

### 3. Why did the VLM not select evidence?

Because it did not outperform the deterministic retrieval order. The selector achieved 50% exact selection on retrieval-hit cases versus 55% for always choosing E1. The verifier was even worse, falsely abstaining 90.91% of the time when the exact target was already E1.

### 4. How did you evaluate hallucination?

I first created an automated VLM judge but calibrated it before trusting it. Deliberately irrelevant, unsupported, and truncated variants were all rated perfect, so the judge was rejected. Final semantic quality comes from a blinded 12-case human audit plus deterministic citation/completion checks on 30 cases.

### 5. Why separate processes?

On macOS, co-loading the PyTorch/OpenCLIP and FAISS native stacks caused a native OpenMP crash. Process isolation fixed stability. Later, persistent isolated workers preserved that safety but removed repeated model startup, reducing warm search from ~15.2 seconds to 171 ms.

## Likely interviewer questions

### “Why is final Recall@10 only 16.73%?”

The benchmark is exact-instance retrieval among 66k highly similar mobile screens. A semantically interchangeable but different RICO screen is counted as wrong. The useful signal is the controlled improvement from 4.83% to 16.73% under the same protocol. The Top-20 oracle at M5 was ~21.65%, so candidate generation remains the main quality bottleneck.

### “Why not use a bigger VLM?”

The project targets a local Apple-Silicon portfolio deployment. The 3B 4-bit model makes the complete RAG path runnable locally. M7 shows its remaining weaknesses honestly: relevance/completeness are 75% and minor unsupported details are common. A larger or fine-tuned model is a reasonable future optimization, but it is outside the frozen release.

### “Why return 429 instead of queueing?”

The validated Mac profile has one heavy inference slot to avoid unified-memory contention. A bounded queue could be added, but explicit backpressure is easier to reason about and prevents latency explosions under overload. The M9 smoke proves overload produces controlled 429s rather than 5xx failures.

### “Is Docker production-ready?”

Docker is intentionally scoped to CI/contracts. The validated data plane is native Apple Silicon with MLX. I do not claim Linux/container inference parity for the frozen model/runtime without a separate benchmark.

### “What would you do next?”

For retrieval: improve candidate generation, potentially with stronger UI-specific representation learning or hybrid text/structure signals. For generation: shorten answers and suppress speculative details, then expand blinded human evaluation with multiple raters. For serving: add a bounded queue/SLO tracking and profile unified-memory pressure under longer mixed workloads.

## What this project demonstrates

- controlled ML experimentation rather than metric cherry-picking;
- multimodal retrieval and RAG architecture;
- evaluation methodology and evaluator calibration;
- model/system trade-off analysis;
- production API design;
- native-process debugging on Apple Silicon;
- operational resilience, observability, CI, and backpressure;
- honest documentation of negative results and limitations.
