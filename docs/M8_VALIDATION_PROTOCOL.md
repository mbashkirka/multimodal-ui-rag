# M8 validation protocol (macOS Apple Silicon)

M8 is an API-serving milestone. M5 retrieval, M6.4 generation, and M7 quality results are frozen.

## 1. Install

```bash
python -m pip install -e ".[rag,api,dev]"
```

## 2. Regression suite

```bash
make test
```

Expected on the validated Mac environment: 96 passing tests.

## 3. Control-plane smoke

```bash
make m8-smoke
```

This checks `/health`, `/ready`, and `/v1/model-info` without running expensive inference. `/ready` must return HTTP 200 before data-plane validation.

## 4. Start service

```bash
make m8-serve
```

Keep this terminal open. In a second terminal run:

```bash
curl -sS http://127.0.0.1:8000/health
curl -sS http://127.0.0.1:8000/ready
curl -sS http://127.0.0.1:8000/v1/model-info
```

## 5. Search smoke

```bash
curl -sS -X POST http://127.0.0.1:8000/v1/search \
  -H 'Content-Type: application/json' \
  -H 'X-Request-ID: m8-search-smoke' \
  -d '{"query":"dark banking login screen","top_k":4}'
```

Acceptance checks: HTTP 200; `X-Request-ID` echoed; four results; no absolute `path` field; latency contains encode/search/rerank stages.

## 6. RAG smoke

```bash
curl -sS -X POST http://127.0.0.1:8000/v1/rag \
  -H 'Content-Type: application/json' \
  -H 'X-Request-ID: m8-rag-smoke' \
  -d '{"query":"dark banking login screen","max_tokens":128}'
```

Acceptance checks: HTTP 200; selected evidence is deterministic `E1`; citations are valid; answer does not expose `raw_text` or local filesystem paths; latency includes generation time.

## 7. OpenAPI

Browse `http://127.0.0.1:8000/docs` and verify both data-plane POST endpoints and all three control-plane GET endpoints are present.
