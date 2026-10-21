# Demo Guide

This is a recruiter/interviewer-facing demo plan. It is designed to show the strongest system properties in roughly five minutes without rebuilding the corpus.

## Pre-demo checklist

1. Use the validated macOS Apple Silicon environment.
2. Ensure the frozen p256 manifest, FlatIP index, p512 reranker artifact, and RICO screenshots are present.
3. Activate the environment.
4. Run the fast release gate before the meeting if desired.

```bash
source .venv/bin/activate
make m9-quality
```

## 5-minute live demo

### 1. Start the service

```bash
make m9-serve
```

Explain while workers load:

> The API host does not import PyTorch, FAISS, or MLX. Each native stack runs in a separate persistent worker. That preserves macOS stability while avoiding per-request model startup.

### 2. Show service identity and health

In another terminal:

```bash
curl -sS http://127.0.0.1:8000/ready | python -m json.tool
curl -sS http://127.0.0.1:8000/v1/model-info | python -m json.tool
```

Point out:

- all three workers are alive;
- serving mode is `persistent-isolated-workers`;
- evidence policy is deterministic rank 1;
- the artifact fingerprint identifies the frozen runtime.

### 3. Search

```bash
curl -sS \
  -X POST http://127.0.0.1:8000/v1/search \
  -H 'Content-Type: application/json' \
  -H 'X-Request-ID: demo-search' \
  -d '{"query":"dark banking login screen","top_k":4}' \
  | python -m json.tool
```

Expected frozen rank-1 screenshot: `61042.jpg`.

Explain the retrieval path:

> SigLIP2 p256 retrieves Top-20 from exact FlatIP, then only those candidates are rescored with p512 and fused using weighted reciprocal-rank fusion.

### 4. Grounded RAG

```bash
curl -sS \
  -X POST http://127.0.0.1:8000/v1/rag \
  -H 'Content-Type: application/json' \
  -H 'X-Request-ID: demo-rag' \
  -d '{"query":"dark banking login screen","max_tokens":128}' \
  | python -m json.tool
```

Expected behavior:

- evidence E1 is `61042.jpg`;
- `selected_evidence_id` is `E1`;
- citations are valid;
- answer identifies a Wright-Patt Credit Union mobile-banking login screen and describes visible login UI elements.

Important design explanation:

> The VLM does not choose or approve evidence. Earlier selector/verifier experiments made the system worse. Retrieval chooses E1, and application code owns the citation. The VLM only interprets the screenshot.

### 5. Show observability

```bash
curl -sS http://127.0.0.1:8000/metrics | grep '^ui_rag_' | head -40
```

Mention worker liveness/restarts/timeouts, active inference, busy rejections, and stage/request duration metrics.

## Optional 2-minute engineering add-on

Run the existing load smoke:

```bash
make m9-load-smoke
```

The validated profile intentionally permits one heavy request. Under concurrency 6, the target-Mac acceptance produced one `200` and five controlled `429` responses with no `5xx` failures.

## What to show in GitHub instead of running live

If model startup or local environment setup is not appropriate during an interview, open these files:

1. `README.md` — system summary and key results;
2. `docs/ARCHITECTURE.md` — online/process diagrams;
3. `docs/BENCHMARKS.md` — quantitative progression;
4. `docs/PORTFOLIO_CASE_STUDY.md` — decision story;
5. `docs/M9_FINAL_RESULTS.md` — production acceptance.

## Demo fallback if the VLM is slow

The strongest fast demo is `/v1/search`, whose warm reference latency is 171 ms. If the interviewer does not want to wait for local VLM generation, show search live and use the validated M8/M9 reference result for the RAG response. Be explicit that this is a recorded reference result rather than a live call.

## Demo talking points to avoid

Do not claim:

- 77.05% production retrieval recall from M5 visual fusion;
- 100% semantic generation quality from the failed same-family judge;
- Linux Docker inference parity;
- statistically precise human-evaluation percentages from 12 cases;
- ANN superiority—the selected production index is exact FlatIP.
