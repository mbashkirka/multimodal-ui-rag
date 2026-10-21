# Portfolio Case Study — From Legacy CLIP Retrieval to Production Multimodal UI RAG

## 1. Problem

The starting point was a small research-style UI retrieval prototype: CLIP embeddings, a historical RICO image map, and FAISS nearest-neighbor search. It could retrieve screenshots, but it lacked reproducible data handling, evaluation, artifact contracts, modern encoders, generation, serving, and operational safeguards.

The portfolio goal was to turn that prototype into a system that could answer three questions credibly:

1. **Does retrieval quality actually improve?**
2. **Does the VLM add useful grounded behavior instead of hiding retrieval errors?**
3. **Can the final system run as an observable, resilient local service?**

## 2. Baseline first, optimization second

M1 preserved the original files for auditability and rebuilt the pipeline around deterministic corpus canonicalization, manifests, hashes, resumable embeddings, exact FAISS search, and macOS native-process isolation.

M2 then froze the system and evaluated it on the official Screen2Words test split. The legacy CLIP baseline achieved only **4.83% Recall@10** for exact target-screen retrieval. FAISS exact search was fast; the representation was the bottleneck.

That measurement prevented premature optimization of the index.

## 3. Encoder modernization

M3 compared the legacy CLIP encoder against two SigLIP2 Base variants under the same Screen2Words protocol.

| Encoder | R@10 | R@100 | MRR@10 |
|---|---:|---:|---:|
| CLIP ViT-B/32 | 4.83% | 13.87% | 0.0209 |
| SigLIP2 Base FixRes | 14.94% | 32.92% | 0.0740 |
| **SigLIP2 Base NaFlex p256** | **16.22%** | **37.31%** | **0.0826** |

NaFlex p256 became the frozen production encoder.

## 4. Rejecting an attractive but wrong optimization

M4 tested HNSW because approximate search looked like the obvious serving optimization. The gate required at least 95% exact-neighbor Recall@10, 98% task R@10 retention, and a 1.2× p50 speedup.

The best tested HNSW configuration reached **92.88% neighbor Recall@10**, even though it retained 99% of task R@10 and was ~6× faster in single-query p50 latency. Because it missed the exact-neighbor gate, HNSW was rejected and exact `IndexFlatIP` remained the production index.

This is a useful portfolio decision: **do not keep an optimization merely because it is faster**.

## 5. Coarse-to-fine reranking

M5 kept global p256 retrieval but rescored only the Top-20 candidates with the same NaFlex checkpoint at p512 resolution. A weighted RRF fusion selected on dev improved frozen-test R@10 from **16.22% to 16.73%**.

The gain is modest, but real and honestly reported. The Top-20 candidate oracle was 21.65%, showing that candidate generation remains the larger bottleneck.

A target-conditioned visual-exemplar diagnostic reached 77.05% R@10. Because it uses target knowledge to construct the exemplar, it is explicitly reported as a diagnostic upper bound—not production performance.

## 6. What the VLM should—and should not—do

M6 explored multiple VLM roles using local Qwen2.5-VL-3B on MLX.

| Variant | Result | Decision |
|---|---|---|
| Multi-image JSON grounding | 12.5% valid JSON+citations; 10% selection on hits | Reject |
| Categorical VLM selector | 50% exact selection vs 55% always-E1 baseline | Reject |
| VLM verifier / abstention gate | 90.91% false abstention when exact target already E1 | Reject |
| **Generation-only VLM** | deterministic E1 + grounded single-image answer | **Select** |

The final contract is simple: **retrieval selects E1, application code owns evidence IDs/citations, and the VLM interprets only E1**.

## 7. Evaluating the evaluator

M7 initially used the same Qwen2.5-VL family as an automated visual judge. It returned perfect scores for all 30 answers. Instead of accepting those numbers, the project added corruption calibration.

The calibration injected irrelevant, unsupported-sentinel, and truncated answers. The judge detected **0%** of all three corruption types and still gave a perfect score to **100%** of corrupted answers.

The automated semantic scores were discarded.

A blinded 12-case human audit produced:

- faithfulness: **91.67%**;
- relevance: **75.00%**;
- completeness: **75.00%**;
- any unsupported detail: **58.33% (7/12)**;
- major unsupported claim: **8.33% (1/12)**.

The important lesson was that the generator's main problem is answer discipline and minor speculation, not frequent catastrophic hallucination.

## 8. Production serving

M8 wrapped the frozen pipeline in FastAPI with versioned endpoints, request IDs, readiness/model identity, structured errors, and sanitized public responses.

The first serving implementation reused the CLI subprocess architecture and therefore reloaded models for every request:

- `/v1/search`: **15.2 s**;
- `/v1/rag`: **34.1 s**.

M8.1 replaced per-request subprocesses with three persistent isolated workers: encoder, FAISS, and VLM. Warm `/v1/search` dropped to **171 ms**, an ~**89×** improvement over M8.0. RAG latency remained dominated by local VLM generation at ~15–17 s.

## 9. Operational hardening

M9 added:

- JSON structured logs;
- Prometheus-compatible metrics;
- worker startup/request timeouts;
- dead-worker restart and one transport retry;
- graceful shutdown;
- concurrency backpressure;
- CI and repository-wide Ruff/test/release gates.

Acceptance on the target Mac:

- **109 tests passed**;
- Ruff, release contracts, and compileall passed;
- a real FAISS worker was killed and restarted successfully;
- six concurrent search requests produced **1×200 + 5×429 + 0×5xx**;
- all workers remained healthy after the tests.

## 10. Final architecture

The final system intentionally separates responsibilities:

- **SigLIP2 p256**: global semantic retrieval;
- **exact FAISS FlatIP**: deterministic candidate search;
- **p512 RRF reranker**: local candidate refinement;
- **deterministic application layer**: evidence selection and provenance;
- **Qwen2.5-VL**: grounded interpretation of one selected screenshot;
- **FastAPI + persistent native workers**: serving;
- **metrics / recovery / backpressure / CI**: operational reliability.

The strongest portfolio signal is not any single metric. It is the repeated pattern of **measure → define a gate → keep or reject the component → preserve the evidence**.
