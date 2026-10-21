# M6.4 Production Protocol

## Frozen retrieval
1. text query -> SigLIP2 Base NaFlex p256;
2. exact FAISS FlatIP retrieval;
3. Top-20 candidate set;
4. p512 corpus rescoring with 50/50 weighted RRF;
5. deterministic rank-1 evidence E1.

## Grounded generation
Qwen2.5-VL-3B-Instruct-4bit receives only E1 at the bounded M6 image budget. The model must describe only visible content and relate it to the user request. It is not asked to rerank, verify, accept, or reject evidence.

Evidence provenance and citation IDs are created by deterministic application code. The VLM cannot invent evidence IDs.

## Why no VLM gate
M6.2 showed that VLM reranking underperformed deterministic rank-1 retrieval. M6.3 showed 90.91% false abstention even when the exact target was already E1. Keeping either decision layer would make the system worse.

## Evaluation boundary
M6 establishes the end-to-end multimodal RAG architecture and records negative ablations. Formal semantic generation evaluation (faithfulness/usefulness/hallucination scoring) is intentionally moved to M7, where judge/human evaluation can be designed without conflating it with retrieval metrics.
