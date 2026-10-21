.PHONY: setup data smoke index-smoke verify-smoke index-full index-full-restart verify-full search-full search-image-full test lint clean-artifacts clean-smoke benchmark-data benchmark-smoke benchmark-primary benchmark-all benchmark-clean m3-setup m3-smoke-fixres m3-smoke-naflex m3-smoke m3-index-fixres m3-index-naflex m3-verify-fixres m3-verify-naflex m3-benchmark-fixres m3-benchmark-naflex m3-compare m3-all5-fixres m3-all5-naflex m3-clean m4-smoke m4-build-m16 m4-build-m32 m4-build-m48 m4-build m4-benchmark m4-clean m5-data m5-smoke-reranker m5-build-reranker m5-prepare-dev m5-rerank-dev m5-rerank-test m5-fusion-dev m5-fusion-test m5-compare m5-clean m6-setup m6-diagnose m6-smoke-vlm m6-smoke-rag m6-prepare-eval m6-eval m6-clean

PYTHON ?= .venv/bin/python
SMOKE_ARTIFACT ?= clip-vit-b32-quickgelu-openai-smoke100
FULL_ARTIFACT ?= clip-vit-b32-quickgelu-openai
FULL_BATCH_SIZE ?= 16

setup:
	./scripts/bootstrap_macos.sh

data:
	$(PYTHON) scripts/download_rico.py

smoke:
	$(PYTHON) scripts/smoke_test_clip.py --image data_storage/RICO_dataset/71018.jpg --text "mobile application screen"

# `index` is an orchestrator. On macOS ARM it runs OpenCLIP/PyTorch embedding
# and FAISS construction in separate Python interpreters to avoid libomp clashes.
index-smoke:
	$(PYTHON) -m ui_rag index \
		--limit 100 \
		--batch-size 16 \
		--artifact-name $(SMOKE_ARTIFACT)

verify-smoke:
	$(PYTHON) scripts/verify_smoke_retrieval.py \
		--manifest artifacts/metadata/$(SMOKE_ARTIFACT)-manifest.json

# Full RICO build is resumable at embedding-batch boundaries. Re-running this
# command after an interruption continues from the persisted progress state.
index-full:
	$(PYTHON) -m ui_rag index \
		--batch-size $(FULL_BATCH_SIZE) \
		--artifact-name $(FULL_ARTIFACT)

# Explicit destructive rebuild of the full embedding/index artifact.
index-full-restart:
	$(PYTHON) -m ui_rag index \
		--batch-size $(FULL_BATCH_SIZE) \
		--artifact-name $(FULL_ARTIFACT) \
		--restart

verify-full:
	$(PYTHON) scripts/verify_full_retrieval.py \
		--manifest artifacts/metadata/$(FULL_ARTIFACT)-manifest.json

search-full:
	$(PYTHON) -m ui_rag search \
		--manifest artifacts/metadata/$(FULL_ARTIFACT)-manifest.json \
		--query "mobile application screen" \
		--top-k 5

search-image-full:
	$(PYTHON) -m ui_rag search \
		--manifest artifacts/metadata/$(FULL_ARTIFACT)-manifest.json \
		--image data_storage/RICO_dataset/71018.jpg \
		--top-k 5

test:
	$(PYTHON) -m pytest -q

lint:
	$(PYTHON) -m ruff check src tests scripts

clean-smoke:
	rm -f artifacts/embeddings/.$(SMOKE_ARTIFACT).partial.npy \
		artifacts/embeddings/$(SMOKE_ARTIFACT).npy \
		artifacts/indexes/$(SMOKE_ARTIFACT)-flatip.faiss \
		artifacts/metadata/$(SMOKE_ARTIFACT)-embedding-progress.json \
		artifacts/metadata/$(SMOKE_ARTIFACT)-embedding-stage.json \
		artifacts/metadata/$(SMOKE_ARTIFACT)-manifest.json \
		artifacts/metadata/$(SMOKE_ARTIFACT)-image-map.json

clean-artifacts:
	rm -f artifacts/embeddings/*.npy \
		artifacts/embeddings/.*.partial.npy \
		artifacts/indexes/*.faiss \
		artifacts/metadata/*-embedding-progress.json \
		artifacts/metadata/*-embedding-stage.json \
		artifacts/metadata/*-manifest.json \
		artifacts/metadata/*-image-map.json \
		artifacts/metadata/latest-manifest.json

# ------------------------------- M2 benchmark -------------------------------
SCREEN2WORDS_PRIMARY_RUN ?= screen2words-test-primary
SCREEN2WORDS_ALL_RUN ?= screen2words-test-all5
BENCHMARK_SMOKE_RUN ?= screen2words-test-smoke200

benchmark-data:
	$(PYTHON) scripts/download_screen2words.py

benchmark-smoke: benchmark-data
	$(PYTHON) -m ui_rag benchmark \
		--manifest artifacts/metadata/$(FULL_ARTIFACT)-manifest.json \
		--run-name $(BENCHMARK_SMOKE_RUN) \
		--variant primary \
		--limit-screens 200 \
		--embedding-batch-size 64 \
		--search-batch-size 128 \
		--search-k 100

benchmark-primary: benchmark-data
	$(PYTHON) -m ui_rag benchmark \
		--manifest artifacts/metadata/$(FULL_ARTIFACT)-manifest.json \
		--run-name $(SCREEN2WORDS_PRIMARY_RUN) \
		--variant primary \
		--embedding-batch-size 128 \
		--search-batch-size 256 \
		--search-k 100

benchmark-all: benchmark-data
	$(PYTHON) -m ui_rag benchmark \
		--manifest artifacts/metadata/$(FULL_ARTIFACT)-manifest.json \
		--run-name $(SCREEN2WORDS_ALL_RUN) \
		--variant all \
		--embedding-batch-size 128 \
		--search-batch-size 256 \
		--search-k 100

benchmark-clean:
	rm -rf artifacts/evaluation/screen2words-test-*

# ------------------------- M3 controlled encoders -------------------------
SIGLIP2_FIXRES_MODEL ?= google/siglip2-base-patch16-224
SIGLIP2_NAFLEX_MODEL ?= google/siglip2-base-patch16-naflex
SIGLIP2_FIXRES_ARTIFACT ?= siglip2-base-p16-224
SIGLIP2_NAFLEX_ARTIFACT ?= siglip2-base-p16-naflex-p256
SIGLIP2_FIXRES_SMOKE ?= siglip2-base-p16-224-smoke100
SIGLIP2_NAFLEX_SMOKE ?= siglip2-base-p16-naflex-p256-smoke100
M3_CORPUS_BATCH ?= 8
M3_QUERY_BATCH ?= 64

m3-setup:
	$(PYTHON) -m pip install "transformers==5.15.1"
	$(PYTHON) -m pip install -e . --no-deps
	$(PYTHON) -c "import transformers; print('Transformers:', transformers.__version__)"

m3-smoke-fixres:
	$(PYTHON) -m ui_rag index \
		--artifact-name $(SIGLIP2_FIXRES_SMOKE) \
		--model-name $(SIGLIP2_FIXRES_MODEL) \
		--pretrained huggingface \
		--encoder-backend transformers_siglip2 \
		--limit 100 \
		--batch-size $(M3_CORPUS_BATCH)
	$(PYTHON) scripts/verify_encoder_smoke.py \
		--manifest artifacts/metadata/$(SIGLIP2_FIXRES_SMOKE)-manifest.json

m3-smoke-naflex:
	$(PYTHON) -m ui_rag index \
		--artifact-name $(SIGLIP2_NAFLEX_SMOKE) \
		--model-name $(SIGLIP2_NAFLEX_MODEL) \
		--pretrained huggingface \
		--encoder-backend transformers_siglip2 \
		--max-num-patches 256 \
		--limit 100 \
		--batch-size $(M3_CORPUS_BATCH)
	$(PYTHON) scripts/verify_encoder_smoke.py \
		--manifest artifacts/metadata/$(SIGLIP2_NAFLEX_SMOKE)-manifest.json

m3-smoke: m3-smoke-fixres m3-smoke-naflex

m3-index-fixres:
	$(PYTHON) -m ui_rag index \
		--artifact-name $(SIGLIP2_FIXRES_ARTIFACT) \
		--model-name $(SIGLIP2_FIXRES_MODEL) \
		--pretrained huggingface \
		--encoder-backend transformers_siglip2 \
		--batch-size $(M3_CORPUS_BATCH)

m3-index-naflex:
	$(PYTHON) -m ui_rag index \
		--artifact-name $(SIGLIP2_NAFLEX_ARTIFACT) \
		--model-name $(SIGLIP2_NAFLEX_MODEL) \
		--pretrained huggingface \
		--encoder-backend transformers_siglip2 \
		--max-num-patches 256 \
		--batch-size $(M3_CORPUS_BATCH)

m3-verify-fixres:
	$(PYTHON) scripts/verify_encoder_smoke.py \
		--manifest artifacts/metadata/$(SIGLIP2_FIXRES_ARTIFACT)-manifest.json

m3-verify-naflex:
	$(PYTHON) scripts/verify_encoder_smoke.py \
		--manifest artifacts/metadata/$(SIGLIP2_NAFLEX_ARTIFACT)-manifest.json

m3-benchmark-fixres: benchmark-data
	$(PYTHON) -m ui_rag benchmark \
		--manifest artifacts/metadata/$(SIGLIP2_FIXRES_ARTIFACT)-manifest.json \
		--run-name screen2words-test-primary-siglip2-fixres \
		--variant primary \
		--embedding-batch-size $(M3_QUERY_BATCH) \
		--search-batch-size 256 \
		--search-k 100

m3-benchmark-naflex: benchmark-data
	$(PYTHON) -m ui_rag benchmark \
		--manifest artifacts/metadata/$(SIGLIP2_NAFLEX_ARTIFACT)-manifest.json \
		--run-name screen2words-test-primary-siglip2-naflex \
		--variant primary \
		--embedding-batch-size $(M3_QUERY_BATCH) \
		--search-batch-size 256 \
		--search-k 100

m3-compare:
	$(PYTHON) scripts/compare_m3_results.py

m3-all5-fixres: benchmark-data
	$(PYTHON) -m ui_rag benchmark \
		--manifest artifacts/metadata/$(SIGLIP2_FIXRES_ARTIFACT)-manifest.json \
		--run-name screen2words-test-all5-siglip2-fixres \
		--variant all \
		--embedding-batch-size $(M3_QUERY_BATCH) \
		--search-batch-size 256 \
		--search-k 100

m3-all5-naflex: benchmark-data
	$(PYTHON) -m ui_rag benchmark \
		--manifest artifacts/metadata/$(SIGLIP2_NAFLEX_ARTIFACT)-manifest.json \
		--run-name screen2words-test-all5-siglip2-naflex \
		--variant all \
		--embedding-batch-size $(M3_QUERY_BATCH) \
		--search-batch-size 256 \
		--search-k 100

m3-clean:
	rm -f artifacts/embeddings/siglip2-*.npy \
		artifacts/embeddings/.siglip2-*.partial.npy \
		artifacts/indexes/siglip2-*.faiss \
		artifacts/metadata/siglip2-*.json
	rm -rf artifacts/evaluation/screen2words-test-*-siglip2-* \
		artifacts/evaluation/m3-comparison

# ------------------------- M4 exact vs ANN/HNSW -------------------------
# M4 reuses the frozen M3 winner. No RICO image encoding is performed here.
M4_EXACT_MANIFEST ?= artifacts/metadata/siglip2-base-p16-naflex-p256-manifest.json
M4_QUERY_DIR ?= artifacts/evaluation/screen2words-test-primary-siglip2-naflex
M4_QUERY_FILE ?= $(M4_QUERY_DIR)/queries.jsonl
M4_QUERY_EMBEDDINGS ?= $(M4_QUERY_DIR)/query_embeddings.npy
M4_QUERY_STAGE ?= $(M4_QUERY_DIR)/query_embedding_stage.json
M4_EF_CONSTRUCTION ?= 200
M4_EF_SEARCH ?= 16 32 64 128 256
M4_BUILD_THREADS ?= 0
M4_BENCH_THREADS ?= 1
M4_ADD_BATCH ?= 4096

M4_HNSW_M16 ?= siglip2-naflex-p256-hnsw-m16-efc200
M4_HNSW_M32 ?= siglip2-naflex-p256-hnsw-m32-efc200
M4_HNSW_M48 ?= siglip2-naflex-p256-hnsw-m48-efc200

m4-smoke:
	@test -f $(M4_EXACT_MANIFEST) || (echo "Missing frozen M3 NaFlex manifest: $(M4_EXACT_MANIFEST)"; exit 2)
	$(PYTHON) scripts/m4_hnsw_smoke.py \
		--manifest $(M4_EXACT_MANIFEST) \
		--limit 1000 \
		--m 32 \
		--ef-construction 100 \
		--ef-search 64 \
		--threads 1

m4-build-m16:
	$(PYTHON) -m ui_rag build-hnsw \
		--source-manifest $(M4_EXACT_MANIFEST) \
		--artifact-name $(M4_HNSW_M16) \
		--m 16 \
		--ef-construction $(M4_EF_CONSTRUCTION) \
		--add-batch-size $(M4_ADD_BATCH) \
		--threads $(M4_BUILD_THREADS)

m4-build-m32:
	$(PYTHON) -m ui_rag build-hnsw \
		--source-manifest $(M4_EXACT_MANIFEST) \
		--artifact-name $(M4_HNSW_M32) \
		--m 32 \
		--ef-construction $(M4_EF_CONSTRUCTION) \
		--add-batch-size $(M4_ADD_BATCH) \
		--threads $(M4_BUILD_THREADS)

m4-build-m48:
	$(PYTHON) -m ui_rag build-hnsw \
		--source-manifest $(M4_EXACT_MANIFEST) \
		--artifact-name $(M4_HNSW_M48) \
		--m 48 \
		--ef-construction $(M4_EF_CONSTRUCTION) \
		--add-batch-size $(M4_ADD_BATCH) \
		--threads $(M4_BUILD_THREADS)

m4-build: m4-build-m16 m4-build-m32 m4-build-m48

m4-benchmark:
	@test -f $(M4_QUERY_FILE) || (echo "Missing M3 NaFlex query cache: $(M4_QUERY_FILE). Run make m3-benchmark-naflex if needed."; exit 2)
	@test -f $(M4_QUERY_EMBEDDINGS) || (echo "Missing M3 NaFlex query embeddings: $(M4_QUERY_EMBEDDINGS). Run make m3-benchmark-naflex if needed."; exit 2)
	@test -f artifacts/metadata/$(M4_HNSW_M16)-manifest.json || (echo "Missing HNSW M16. Run make m4-build first."; exit 2)
	@test -f artifacts/metadata/$(M4_HNSW_M32)-manifest.json || (echo "Missing HNSW M32. Run make m4-build first."; exit 2)
	@test -f artifacts/metadata/$(M4_HNSW_M48)-manifest.json || (echo "Missing HNSW M48. Run make m4-build first."; exit 2)
	$(PYTHON) scripts/m4_benchmark.py \
		--exact-manifest $(M4_EXACT_MANIFEST) \
		--hnsw-manifest artifacts/metadata/$(M4_HNSW_M16)-manifest.json \
		--hnsw-manifest artifacts/metadata/$(M4_HNSW_M32)-manifest.json \
		--hnsw-manifest artifacts/metadata/$(M4_HNSW_M48)-manifest.json \
		--queries $(M4_QUERY_FILE) \
		--query-embeddings $(M4_QUERY_EMBEDDINGS) \
		--query-stage $(M4_QUERY_STAGE) \
		--output-dir artifacts/evaluation/m4-exact-vs-hnsw \
		--ef-search $(M4_EF_SEARCH) \
		--search-k 10 \
		--latency-k 10 \
		--latency-sample 1000 \
		--batch-size 256 \
		--rounds 3 \
		--threads $(M4_BENCH_THREADS) \
		--min-neighbor-recall 0.95 \
		--min-task-retention 0.98 \
		--min-speedup 1.20

m4-clean:
	rm -f artifacts/indexes/siglip2-naflex-p256-hnsw-*.faiss \
		artifacts/metadata/siglip2-naflex-p256-hnsw-*-manifest.json
	rm -rf artifacts/evaluation/m4-exact-vs-hnsw


# -------------------- M5 multimodal fusion + reranking --------------------
# Frozen production retriever from M4: SigLIP2 Base NaFlex p256 + FlatIP.
M5_BASE_MANIFEST ?= artifacts/metadata/siglip2-base-p16-naflex-p256-manifest.json
M5_MODEL ?= google/siglip2-base-patch16-naflex
M5_RERANK_PATCHES ?= 512
M5_RERANK_ARTIFACT ?= siglip2-base-p16-naflex-p512-rerank
M5_RERANK_SMOKE ?= siglip2-base-p16-naflex-p512-rerank-smoke100
M5_RERANK_BATCH ?= 4
M5_DEV_RUN ?= screen2words-dev-primary-siglip2-naflex
M5_TEST_RUN ?= screen2words-test-primary-siglip2-naflex
M5_DEV_DIR ?= artifacts/evaluation/$(M5_DEV_RUN)
M5_TEST_DIR ?= artifacts/evaluation/$(M5_TEST_RUN)

m5-data: benchmark-data

m5-smoke-reranker:
	@test -f $(M5_BASE_MANIFEST) || (echo "Missing frozen M3/M4 NaFlex manifest: $(M5_BASE_MANIFEST)"; exit 2)
	$(PYTHON) -m ui_rag index \
		--artifact-name $(M5_RERANK_SMOKE) \
		--model-name $(M5_MODEL) \
		--pretrained huggingface \
		--encoder-backend transformers_siglip2 \
		--max-num-patches $(M5_RERANK_PATCHES) \
		--limit 100 \
		--batch-size $(M5_RERANK_BATCH)
	$(PYTHON) scripts/verify_encoder_smoke.py \
		--manifest artifacts/metadata/$(M5_RERANK_SMOKE)-manifest.json

# Full p512 corpus representation used only for candidate rescoring. No global
# p512 FAISS index is built: stage 1 remains the frozen p256 FlatIP retriever.
m5-build-reranker:
	$(PYTHON) -m ui_rag embed \
		--artifact-name $(M5_RERANK_ARTIFACT) \
		--model-name $(M5_MODEL) \
		--pretrained huggingface \
		--encoder-backend transformers_siglip2 \
		--max-num-patches $(M5_RERANK_PATCHES) \
		--batch-size $(M5_RERANK_BATCH)

m5-prepare-dev: m5-data
	$(PYTHON) -m ui_rag benchmark \
		--manifest $(M5_BASE_MANIFEST) \
		--test-screens data_storage/benchmarks/screen2words/source/dev_screens.txt \
		--split-name dev \
		--run-name $(M5_DEV_RUN) \
		--variant primary \
		--embedding-batch-size 64 \
		--search-batch-size 256 \
		--search-k 100

m5-rerank-dev:
	@test -f artifacts/metadata/$(M5_RERANK_ARTIFACT)-embedding-stage.json || (echo "Missing p512 reranker embeddings. Run make m5-build-reranker."; exit 2)
	@test -f $(M5_DEV_DIR)/query_embeddings.npy || (echo "Missing dev query cache. Run make m5-prepare-dev."; exit 2)
	$(PYTHON) scripts/m5_rerank_benchmark.py \
		--exact-manifest $(M5_BASE_MANIFEST) \
		--rerank-stage artifacts/metadata/$(M5_RERANK_ARTIFACT)-embedding-stage.json \
		--queries $(M5_DEV_DIR)/queries.jsonl \
		--query-embeddings $(M5_DEV_DIR)/query_embeddings.npy \
		--query-stage $(M5_DEV_DIR)/query_embedding_stage.json \
		--output-dir artifacts/evaluation/m5-rerank-dev \
		--candidate-k 20 50 100 \
		--base-weight 0.0 0.25 0.5 0.75 \
		--rrf-constant 60 \
		--threads 1

m5-rerank-test:
	@test -f artifacts/evaluation/m5-rerank-dev/selection.json || (echo "Missing dev selection. Run make m5-rerank-dev."; exit 2)
	@test -f $(M5_TEST_DIR)/query_embeddings.npy || (echo "Missing frozen M3 test query cache. Run make m3-benchmark-naflex if needed."; exit 2)
	$(PYTHON) scripts/m5_rerank_benchmark.py \
		--exact-manifest $(M5_BASE_MANIFEST) \
		--rerank-stage artifacts/metadata/$(M5_RERANK_ARTIFACT)-embedding-stage.json \
		--queries $(M5_TEST_DIR)/queries.jsonl \
		--query-embeddings $(M5_TEST_DIR)/query_embeddings.npy \
		--query-stage $(M5_TEST_DIR)/query_embedding_stage.json \
		--output-dir artifacts/evaluation/m5-rerank-test \
		--selection artifacts/evaluation/m5-rerank-dev/selection.json \
		--threads 1

# Secondary controlled multimodal benchmark. The reference image is the nearest
# non-self RICO screen to each target under frozen p256 image embeddings.
m5-fusion-dev:
	@test -f $(M5_DEV_DIR)/query_embeddings.npy || (echo "Missing dev query cache. Run make m5-prepare-dev."; exit 2)
	$(PYTHON) scripts/m5_fusion_benchmark.py \
		--manifest $(M5_BASE_MANIFEST) \
		--queries $(M5_DEV_DIR)/queries.jsonl \
		--query-embeddings $(M5_DEV_DIR)/query_embeddings.npy \
		--query-stage $(M5_DEV_DIR)/query_embedding_stage.json \
		--output-dir artifacts/evaluation/m5-fusion-dev \
		--text-weight 0.0 0.25 0.50 0.75 0.90 1.0 \
		--search-k 100 \
		--threads 1

m5-fusion-test:
	@test -f artifacts/evaluation/m5-fusion-dev/selection.json || (echo "Missing fusion dev selection. Run make m5-fusion-dev."; exit 2)
	$(PYTHON) scripts/m5_fusion_benchmark.py \
		--manifest $(M5_BASE_MANIFEST) \
		--queries $(M5_TEST_DIR)/queries.jsonl \
		--query-embeddings $(M5_TEST_DIR)/query_embeddings.npy \
		--query-stage $(M5_TEST_DIR)/query_embedding_stage.json \
		--output-dir artifacts/evaluation/m5-fusion-test \
		--selection artifacts/evaluation/m5-fusion-dev/selection.json \
		--search-k 100 \
		--threads 1

m5-compare:
	$(PYTHON) scripts/compare_m5_results.py

m5-clean:
	rm -f artifacts/embeddings/$(M5_RERANK_ARTIFACT).npy \
		artifacts/embeddings/.$(M5_RERANK_ARTIFACT).partial.npy \
		artifacts/embeddings/$(M5_RERANK_SMOKE).npy \
		artifacts/embeddings/.$(M5_RERANK_SMOKE).partial.npy \
		artifacts/indexes/$(M5_RERANK_SMOKE)-flatip.faiss \
		artifacts/metadata/$(M5_RERANK_ARTIFACT)-*.json \
		artifacts/metadata/$(M5_RERANK_SMOKE)-*.json
	rm -rf artifacts/evaluation/$(M5_DEV_RUN) \
		artifacts/evaluation/m5-rerank-dev \
		artifacts/evaluation/m5-rerank-test \
		artifacts/evaluation/m5-fusion-dev \
		artifacts/evaluation/m5-fusion-test \
		artifacts/evaluation/m5-comparison

# ------------------------- M6 grounded multimodal RAG -------------------------
# Frozen production retriever from M5; M6 adds evidence packaging + local MLX VLM.
M6_BASE_MANIFEST ?= artifacts/metadata/siglip2-base-p16-naflex-p256-manifest.json
M6_RERANK_STAGE ?= artifacts/metadata/siglip2-base-p16-naflex-p512-rerank-embedding-stage.json
M6_MODEL ?= mlx-community/Qwen2.5-VL-3B-Instruct-4bit
M6_MAX_IMAGE_PIXELS ?= 1003520
M6_DEV_DIR ?= artifacts/evaluation/screen2words-dev-primary-siglip2-naflex
M6_CASES ?= artifacts/evaluation/m6-grounding-v2/dev/cases.jsonl
M6_EVAL_DIR ?= artifacts/evaluation/m6-grounding-v2/dev/selector-eval
M6_ANSWER_EVAL_DIR ?= artifacts/evaluation/m6-grounding-v2/dev/rank1-answer-eval

m6-setup:
	$(PYTHON) -m pip install "mlx==0.32.0" "mlx-vlm==0.6.8"
	$(PYTHON) -m pip install -e . --no-deps
	$(PYTHON) -c "import mlx_vlm; print('mlx-vlm import: OK')"


m6-diagnose:
	$(PYTHON) scripts/m6_mlx_diagnose.py

m6-smoke-vlm:
	@test -f data_storage/RICO_dataset/71018.jpg || (echo "Missing RICO image. Run make data first."; exit 2)
	$(PYTHON) scripts/m6_vlm_smoke.py \
		--image data_storage/RICO_dataset/71018.jpg \
		--model $(M6_MODEL) \
		--max-image-pixels $(M6_MAX_IMAGE_PIXELS)

m6-smoke-rag:
	@test -f $(M6_BASE_MANIFEST) || (echo "Missing frozen p256 manifest: $(M6_BASE_MANIFEST)"; exit 2)
	@test -f $(M6_RERANK_STAGE) || (echo "Missing frozen p512 reranker stage: $(M6_RERANK_STAGE)"; exit 2)
	$(PYTHON) -m ui_rag rag \
		--query "dark banking login screen" \
		--base-manifest $(M6_BASE_MANIFEST) \
		--rerank-stage $(M6_RERANK_STAGE) \
		--candidate-k 20 \
		--evidence-k 4 \
		--model $(M6_MODEL) \
		--evidence-policy rank1 \
		--max-tokens 96 \
		--temperature 0.0 \
		--max-image-pixels $(M6_MAX_IMAGE_PIXELS) \
		--json

# M6.2 selection protocol: tune/debug on Screen2Words dev, not the already-observed test diagnostic.
# Build 20 retrieval-hit + 20 retrieval-miss cases from the frozen M5 dev cache.
m6-prepare-eval:
	@test -f $(M6_DEV_DIR)/queries.jsonl || (echo "Missing M5 dev query cache. Run make m5-prepare-dev first."; exit 2)
	@test -f $(M6_DEV_DIR)/query_embeddings.npy || (echo "Missing M5 dev query embeddings."; exit 2)
	@test -f $(M6_RERANK_STAGE) || (echo "Missing M5 p512 reranker stage."; exit 2)
	mkdir -p artifacts/evaluation/m6-grounding-v2/dev
	$(PYTHON) scripts/m6_prepare_eval_cases.py \
		--base-manifest $(M6_BASE_MANIFEST) \
		--rerank-stage $(M6_RERANK_STAGE) \
		--queries $(M6_DEV_DIR)/queries.jsonl \
		--query-embeddings $(M6_DEV_DIR)/query_embeddings.npy \
		--output $(M6_CASES) \
		--evidence-k 4 \
		--candidate-k 20 \
		--base-weight 0.5 \
		--rrf-constant 60 \
		--hit-cases 20 \
		--miss-cases 20 \
		--threads 1

# M6.2 ablation: load Qwen once, show a labeled contact sheet, emit E1..E4/NONE.
m6-eval:
	@test -f $(M6_CASES) || (echo "Missing M6.2 dev cases. Run make m6-prepare-eval."; exit 2)
	$(PYTHON) scripts/m6_selector_eval.py \
		--cases $(M6_CASES) \
		--output-dir $(M6_EVAL_DIR) \
		--model $(M6_MODEL) \
		--selector-max-tokens 8 \
		--max-image-pixels $(M6_MAX_IMAGE_PIXELS)

# Historical M6.3 diagnostic: deterministic E1 followed by the rejected verifier/answer policy.
# Retained for reproducibility. It measures false abstention when the exact target is already rank-1; other abstention
# rates are diagnostic because non-target RICO screens may still be semantically valid.
m6-answer-eval:
	@test -f $(M6_CASES) || (echo "Missing M6.3 dev cases. Run make m6-prepare-eval."; exit 2)
	$(PYTHON) scripts/m6_answer_eval.py \
		--cases $(M6_CASES) \
		--output-dir $(M6_ANSWER_EVAL_DIR) \
		--model $(M6_MODEL) \
		--answer-max-tokens 64 \
		--max-image-pixels $(M6_MAX_IMAGE_PIXELS)

m6-clean:
	rm -rf artifacts/evaluation/m6-grounding-v2

# ------------------------- M7 generation evaluation -------------------------
# Frozen M6.4 production path: deterministic E1 evidence + local grounded VLM answer.
M7_MODEL ?= mlx-community/Qwen2.5-VL-3B-Instruct-4bit
M7_MAX_IMAGE_PIXELS ?= 1003520
M7_DEV_DIR ?= artifacts/evaluation/screen2words-dev-primary-siglip2-naflex
M7_CASES ?= artifacts/evaluation/m7-generation/holdout/cases.jsonl
M7_GENERATION_DIR ?= artifacts/evaluation/m7-generation/answers
M7_JUDGE_DIR ?= artifacts/evaluation/m7-generation/judge
M7_REPORT_DIR ?= artifacts/evaluation/m7-generation/report
M7_CALIBRATION_DIR ?= artifacts/evaluation/m7-generation/judge-calibration

.PHONY: m7-smoke m7-prepare m7-generate m7-judge m7-calibrate m7-report m7-all m7-clean

m7-smoke:
	$(PYTHON) -c "from ui_rag.generation_eval.judge import parse_judge_code; assert parse_judge_code('F2 R2 U0 C2'); print('M7 evaluator smoke PASSED')"

# Fresh generation holdout: exclude M6 prompt-design/eval cases from Screen2Words dev,
# then select equal slices for exact-rank1, target-elsewhere-top4, and retrieval miss.
m7-prepare:
	@test -f $(M7_DEV_DIR)/queries.jsonl || (echo "Missing M5 dev query cache. Run make m5-prepare-dev first."; exit 2)
	@test -f $(M7_DEV_DIR)/query_embeddings.npy || (echo "Missing M5 dev query embeddings."; exit 2)
	@test -f $(M6_RERANK_STAGE) || (echo "Missing frozen M5 p512 reranker stage."; exit 2)
	mkdir -p artifacts/evaluation/m7-generation/holdout
	$(PYTHON) scripts/m7_prepare_cases.py \
		--base-manifest $(M6_BASE_MANIFEST) \
		--rerank-stage $(M6_RERANK_STAGE) \
		--queries $(M7_DEV_DIR)/queries.jsonl \
		--query-embeddings $(M7_DEV_DIR)/query_embeddings.npy \
		--output $(M7_CASES) \
		--exclude-cases $(M6_CASES) \
		--rank1-exact 10 \
		--top4-nonrank1 10 \
		--miss 10 \
		--candidate-k 20 \
		--evidence-k 4 \
		--base-weight 0.5 \
		--rrf-constant 60 \
		--threads 1

m7-generate:
	@test -f $(M7_CASES) || (echo "Missing M7 holdout. Run make m7-prepare."; exit 2)
	$(PYTHON) scripts/m7_generate_answers.py \
		--cases $(M7_CASES) \
		--output-dir $(M7_GENERATION_DIR) \
		--model $(M7_MODEL) \
		--max-tokens 128 \
		--max-image-pixels $(M7_MAX_IMAGE_PIXELS)

m7-judge:
	@test -f $(M7_GENERATION_DIR)/answers.jsonl || (echo "Missing M7 answers. Run make m7-generate."; exit 2)
	$(PYTHON) scripts/m7_judge_answers.py \
		--answers $(M7_GENERATION_DIR)/answers.jsonl \
		--output-dir $(M7_JUDGE_DIR) \
		--model $(M7_MODEL) \
		--max-tokens 20 \
		--max-image-pixels $(M7_MAX_IMAGE_PIXELS)

m7-calibrate:
	@test -f $(M7_GENERATION_DIR)/answers.jsonl || (echo "Missing M7 answers. Run make m7-generate."; exit 2)
	$(PYTHON) scripts/m7_calibrate_judge.py \
		--answers $(M7_GENERATION_DIR)/answers.jsonl \
		--output-dir $(M7_CALIBRATION_DIR) \
		--model $(M7_MODEL) \
		--per-slice 2 \
		--max-tokens 20 \
		--max-image-pixels $(M7_MAX_IMAGE_PIXELS)

m7-report:
	@test -f $(M7_JUDGE_DIR)/judged.jsonl || (echo "Missing judged answers. Run make m7-judge."; exit 2)
	$(PYTHON) scripts/m7_report.py \
		--judged $(M7_JUDGE_DIR)/judged.jsonl \
		--output-dir $(M7_REPORT_DIR) \
		--human-audit-size 12 \
		--calibration-summary $(M7_CALIBRATION_DIR)/calibration_summary.json

m7-all: m7-prepare m7-generate m7-judge m7-calibrate m7-report

m7-clean:
	rm -rf artifacts/evaluation/m7-generation

# ------------------------- M7.2 blinded human audit -------------------------
M7_HUMAN_AUDIT_DIR ?= artifacts/evaluation/m7-generation/human-audit
M7_HUMAN_COMPLETED ?= $(M7_HUMAN_AUDIT_DIR)/human_audit_completed.csv

.PHONY: m7-human-audit m7-human-audit-serve m7-human-audit-cli m7-human-report

m7-human-audit:
	@test -f $(M7_JUDGE_DIR)/judged.jsonl || (echo "Missing M7 judged answers. Run make m7-judge first."; exit 2)
	$(PYTHON) scripts/m7_build_human_audit.py \
		--judged $(M7_JUDGE_DIR)/judged.jsonl \
		--output-dir $(M7_HUMAN_AUDIT_DIR) \
		--size 12
	@echo "Audit bundle ready. Recommended browser-free workflow: make m7-human-audit-cli"

m7-human-audit-serve:
	@test -f $(M7_HUMAN_AUDIT_DIR)/audit.html || (echo "Missing audit bundle. Run make m7-human-audit first."; exit 2)
	$(PYTHON) scripts/m7_human_audit_server.py --audit-dir $(M7_HUMAN_AUDIT_DIR) --port 8765

m7-human-audit-cli:
	@test -f $(M7_HUMAN_AUDIT_DIR)/audit_manifest.json || (echo "Missing audit bundle. Run make m7-human-audit first."; exit 2)
	$(PYTHON) scripts/m7_human_audit_cli.py \
		--audit-dir $(M7_HUMAN_AUDIT_DIR) \
		--output $(M7_HUMAN_COMPLETED)

m7-human-report:
	@test -f $(M7_HUMAN_COMPLETED) || (echo "Missing completed human audit: $(M7_HUMAN_COMPLETED)"; exit 2)
	@test -f $(M7_HUMAN_AUDIT_DIR)/audit_key.json || (echo "Missing audit key. Run make m7-human-audit first."; exit 2)
	@test -f $(M7_REPORT_DIR)/report.json || (echo "Missing M7.1 deterministic report. Run make m7-report first."; exit 2)
	$(PYTHON) scripts/m7_human_report.py \
		--scores $(M7_HUMAN_COMPLETED) \
		--audit-key $(M7_HUMAN_AUDIT_DIR)/audit_key.json \
		--deterministic-report $(M7_REPORT_DIR)/report.json \
		--output $(M7_REPORT_DIR)/human_report.json

# ------------------------- M8 production FastAPI service -------------------------
M8_HOST ?= 127.0.0.1
M8_PORT ?= 8000

.PHONY: m8-setup m8-smoke m8-serve m8-latency-smoke

m8-setup:
	$(PYTHON) -m pip install -e ".[rag,api,dev]"
	$(PYTHON) -c "import fastapi, uvicorn; print('FastAPI:', fastapi.__version__, 'Uvicorn:', uvicorn.__version__)"

# Control-plane smoke validates /health, /ready and /v1/model-info without heavy inference.
m8-smoke:
	$(PYTHON) scripts/m8_api_smoke.py

m8-serve:
	$(PYTHON) -m ui_rag.api.run --host $(M8_HOST) --port $(M8_PORT)

# Live steady-state latency check; requires `make m8-serve` in another terminal.
m8-latency-smoke:
	$(PYTHON) scripts/m8_latency_smoke.py --base-url http://$(M8_HOST):$(M8_PORT)

# ------------------------- M9 engineering / operational hardening -------------------------
M9_BASE_URL ?= http://127.0.0.1:8000

.PHONY: m9-setup m9-quality m9-smoke m9-serve m9-worker-recovery-smoke m9-load-smoke m9-metrics-smoke m9-docker-ci

m9-setup:
	$(PYTHON) -m pip install -c requirements/macos-proven.txt -e ".[rag,api,dev]"
	$(PYTHON) -c "import fastapi, uvicorn, httpx2; import ui_rag; print('ui-rag:', ui_rag.__version__, 'FastAPI:', fastapi.__version__, 'Uvicorn:', uvicorn.__version__, 'HTTPX2:', httpx2.__version__)"

m9-quality: lint test
	$(PYTHON) scripts/m9_check_release.py
	$(PYTHON) -m compileall -q src scripts tests
	@echo "M9 local quality gate PASSED"

m9-smoke:
	$(PYTHON) scripts/m9_check_release.py
	$(PYTHON) scripts/m8_api_smoke.py
	@echo "M9 control-plane / release smoke PASSED"

m9-serve:
	$(PYTHON) -m ui_rag.api.run --host $(M8_HOST) --port $(M8_PORT)

# Loads only the FAISS worker, kills it, and verifies one-shot automatic restart.
m9-worker-recovery-smoke:
	$(PYTHON) scripts/m9_worker_recovery_smoke.py

# Requires `make m9-serve` in another terminal. Exercises 429 backpressure + Prometheus busy counter.
m9-load-smoke:
	$(PYTHON) scripts/m9_load_smoke.py --base-url $(M9_BASE_URL) --requests 6 --concurrency 6

m9-metrics-smoke:
	@curl -fsS $(M9_BASE_URL)/metrics | grep -E '^(ui_rag_inference_|ui_rag_worker_|ui_rag_http_requests_total)' | head -40

m9-docker-ci:
	docker compose -f docker-compose.ci.yml build
	docker compose -f docker-compose.ci.yml run --rm quality

# ------------------------- M10 portfolio release -------------------------
.PHONY: m10-docs-check m10-release-check m10-smoke m10-demo-info

m10-docs-check:
	$(PYTHON) scripts/m10_check_release.py

m10-release-check: m9-quality m10-docs-check
	@echo "M10 portfolio release candidate gate PASSED"

m10-smoke: m10-docs-check m9-smoke
	@echo "M10 portfolio/control-plane smoke PASSED"

m10-demo-info:
	@echo "See docs/DEMO.md"
	@echo "Start API: make m9-serve"
	@echo "Open docs: http://127.0.0.1:8000/docs"
