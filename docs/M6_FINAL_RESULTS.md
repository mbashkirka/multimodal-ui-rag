# M6 FINAL RESULTS — Full Multimodal RAG

## Status

**M6 COMPLETE.** The final production architecture is M6.4: frozen M5 retrieval/reranking selects the rank-1 screenshot deterministically; a local Qwen2.5-VL-3B-Instruct 4-bit model receives only that screenshot and performs grounded answer generation. The VLM does not rerank evidence and does not gate/reject the retriever output.

Final user-verified environment:

- `ui-rag==0.6.4`
- Python 3.12.0
- Apple Silicon / macOS
- `mlx==0.32.0`
- `mlx-vlm==0.6.8`
- model: `mlx-community/Qwen2.5-VL-3B-Instruct-4bit`
- visual budget: 1,003,520 pixels (1280 Qwen merged visual tokens)
- regression suite: **73 passed**

## Frozen retrieval stack

```text
text query
   ↓
SigLIP2 Base NaFlex p256
   ↓
FAISS IndexFlatIP
   ↓
Top-20 candidates
   ↓
NaFlex p512 candidate rescoring
   ↓
50/50 weighted RRF, c=60
   ↓
deterministic rank-1 evidence E1
```

The retrieval stack is inherited unchanged from M5. Frozen Screen2Words test Recall@10 is 16.73% after the p512 coarse-to-fine reranker.

## Final M6.4 generation path

```text
E1 screenshot selected by retrieval
        ↓
resize only if needed to bounded visual budget
        ↓
Qwen2.5-VL-3B-Instruct-4bit on MLX
        ↓
grounded natural-language answer
        ↓
application attaches deterministic E1 citation
```

Responsibilities are deliberately separated:

- retrieval/reranking chooses evidence;
- application code owns evidence IDs and provenance;
- VLM interprets the selected image and writes the answer;
- VLM cannot invent or alter evidence IDs;
- formal semantic generation scoring is deferred to M7.

## M6 ablation history

### M6.1 — four-image structured JSON grounding

Stable Apple-Silicon inference was achieved after bounding image resolution, but the one-pass four-image structured contract performed poorly on a 40-case balanced test diagnostic:

- structured JSON + citation validity: **12.5%**
- evidence selection accuracy on retrieval-hit cases: **10%**
- end-to-end exact selection: **5%**
- miss abstention rate: **0%** (diagnostic)
- mean generation latency: **43.67 s/case**

Decision: reject monolithic multi-image JSON generation as a production grounding contract.

### M6.2 — categorical contact-sheet selector

The model was reduced to a five-way `E1/E2/E3/E4/NONE` decision and JSON/citations moved into deterministic application code. Screen2Words dev, 40 deterministic cases:

- selector parse validity: **100%**
- VLM exact evidence selection on retrieval-hit cases: **50%**
- deterministic retrieval rank-1 baseline on the same hit cases: **55%**
- absolute gain over rank-1: **-5 percentage points**
- end-to-end exact selection: **25%**
- mean selector latency: **10.89 s**
- median latency: **11.15 s**
- p95 latency: **15.85 s**

Decision: reject VLM reranking/selection. It underperformed the frozen retrieval rank-1 and added substantial latency.

### M6.3 — deterministic rank-1 + VLM verifier/abstention

The VLM saw only E1 and could either answer or emit `INSUFFICIENT_EVIDENCE`. Screen2Words dev, same 40-case diagnostic:

- exact-rank1 hit cases: **11**
- accepted when exact target was already rank-1: **1/11 = 9.09%**
- false abstention when exact target was already rank-1: **10/11 = 90.91%**
- non-exact-rank1 hit abstention: **100%** (diagnostic)
- retrieval-miss abstention: **100%** (diagnostic)
- mean answer latency: **9.19 s**
- median latency: **8.09 s**
- p95 latency: **12.62 s**

Decision: reject the VLM as an evidence gate/verifier. Its abstention policy was far too conservative and frequently discarded correct retrieval output.

### M6.4 — generation-only VLM — SELECTED

The VLM no longer makes any retrieval or gating decision. E1 is fixed by the retriever, citations are deterministic, and Qwen performs only single-image grounded answer generation.

User-verified smoke query:

```text
dark banking login screen
```

Retrieved E1:

```text
61042.jpg
```

The VLM correctly described the screen as Wright-Patt Credit Union Mobile Banking and identified visible login elements including Username, Password, Save Username, Log in, dark-blue styling, and navigation elements. The output carried deterministic `E1` provenance and `abstain=false`.

## Final decision

**Production M6 configuration:**

```text
M5 retriever/reranker → deterministic E1 → single-image Qwen2.5-VL grounded answer
```

Rejected production alternatives remain in the repository as measured ablations:

1. multi-image free-form/JSON grounding;
2. VLM contact-sheet evidence selector;
3. VLM evidence verifier/abstention gate.

This is an empirical architecture decision, not a preference: both VLM decision layers were shown to make the frozen retrieval system worse.

## M7 boundary

M6 establishes a stable end-to-end multimodal RAG path and deterministic provenance. M7 should evaluate **generation quality**, separately from retrieval quality, including at least faithfulness to visible evidence, query relevance, unsupported-claim/hallucination rate, answer completeness, and latency.
