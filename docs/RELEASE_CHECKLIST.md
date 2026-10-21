# M10 Public Release Checklist

## Code quality

- [ ] `make m9-quality` passes.
- [ ] `make m10-release-check` passes.
- [ ] version in `pyproject.toml` matches `ui_rag.__version__`.
- [ ] public OpenAPI routes remain `/health`, `/ready`, `/v1/model-info`, `/v1/search`, `/v1/rag`.
- [ ] `/metrics` remains intentionally hidden from OpenAPI.
- [ ] FastAPI host preserves native import isolation.

## Repository hygiene

- [ ] no `__pycache__/`, `.pyc`, `.pytest_cache/`, `.ruff_cache/`, or `.DS_Store` files in the release archive;
- [ ] no RICO screenshot corpus committed;
- [ ] no model weights/cache committed;
- [ ] no generated `.npy`/`.faiss` production artifacts committed;
- [ ] no absolute personal home-directory paths in portfolio-facing docs;
- [ ] legacy original files and `image_index_map.json` remain byte-for-byte unchanged.

## Claims / evaluation integrity

- [ ] production retrieval claim is **16.73% Screen2Words R@10**, not the target-conditioned 77.05% visual diagnostic;
- [ ] HNSW is described as rejected under the defined neighbor-recall gate;
- [ ] same-family automated judge scores are marked unreliable and excluded from semantic quality claims;
- [ ] human semantic results are described as a 12-case one-evaluator release audit;
- [ ] warm search latency claim uses the dedicated M8.1 171 ms reference, not the cold/concurrent load-smoke latency;
- [ ] Docker is described as CI/contract scope, not validated Linux data-plane parity.

## Demo readiness

- [ ] `/ready` shows encoder/FAISS/VLM workers alive;
- [ ] `/v1/model-info` reports the expected artifact fingerprint;
- [ ] `/v1/search` returns the validated rank-1 result for the demo query;
- [ ] `/v1/rag` returns E1 provenance and a grounded answer;
- [ ] `/metrics` returns worker/request counters;
- [ ] a recorded fallback result is available if live VLM generation is too slow for the interview.

## GitHub polish

- [ ] repository description is concise and matches README scope;
- [ ] topics/tags include relevant terms such as `multimodal-rag`, `faiss`, `siglip2`, `fastapi`, `mlx`, `vision-language-model`, `retrieval`;
- [ ] no license is claimed unless the repository owner intentionally chooses one;
- [ ] README renders Mermaid diagrams correctly on GitHub;
- [ ] CI workflow passes after push.
