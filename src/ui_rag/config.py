from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data_storage"
RICO_DIR = DATA_DIR / "RICO_dataset"
LEGACY_IMAGE_MAP = DATA_DIR / "image_index_map.json"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
EMBEDDING_DIR = ARTIFACT_DIR / "embeddings"
INDEX_DIR = ARTIFACT_DIR / "indexes"
METADATA_DIR = ARTIFACT_DIR / "metadata"
DEFAULT_MODEL_NAME = "ViT-B-32-quickgelu"
DEFAULT_PRETRAINED = "openai"
DEFAULT_ARTIFACT_NAME = "clip-vit-b32-quickgelu-openai"

BENCHMARK_DIR = DATA_DIR / "benchmarks"
SCREEN2WORDS_DIR = BENCHMARK_DIR / "screen2words"
SCREEN2WORDS_SOURCE_DIR = SCREEN2WORDS_DIR / "source"
SCREEN2WORDS_SUMMARIES = SCREEN2WORDS_SOURCE_DIR / "screen_summaries.csv"
SCREEN2WORDS_TEST_SCREENS = SCREEN2WORDS_SOURCE_DIR / "test_screens.txt"
SCREEN2WORDS_DEV_SCREENS = SCREEN2WORDS_SOURCE_DIR / "dev_screens.txt"
EVALUATION_DIR = ARTIFACT_DIR / "evaluation"

# M3 controlled encoder candidates
SIGLIP2_FIXRES_MODEL = "google/siglip2-base-patch16-224"
SIGLIP2_NAFLEX_MODEL = "google/siglip2-base-patch16-naflex"
SIGLIP2_PRETRAINED = "huggingface"
SIGLIP2_NAFLEX_MAX_PATCHES = 256

# M5 multi-resolution reranking
SIGLIP2_NAFLEX_RERANK_MAX_PATCHES = 512

# M6 grounded multimodal RAG
M6_VLM_MODEL = "mlx-community/Qwen2.5-VL-3B-Instruct-4bit"
M6_EVIDENCE_K = 4
M6_CANDIDATE_K = 20
M6_VLM_MAX_IMAGE_PIXELS = 1280 * 28 * 28
