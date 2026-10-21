from pathlib import Path


def test_m4_makefile_freezes_encoder_and_sweeps_hnsw():
    text = Path("Makefile").read_text(encoding="utf-8")
    assert "M4_EXACT_MANIFEST ?= artifacts/metadata/siglip2-base-p16-naflex-p256-manifest.json" in text
    assert "M4_EF_SEARCH ?= 16 32 64 128 256" in text
    assert "m4-build-m16:" in text
    assert "m4-build-m32:" in text
    assert "m4-build-m48:" in text
    assert "m4-benchmark:" in text


def test_m4_benchmark_does_not_import_torch():
    text = Path("scripts/m4_benchmark.py").read_text(encoding="utf-8")
    assert "import faiss" in text
    assert "import torch" not in text
    assert "transformers" not in text


def test_m4_hnsw_builder_does_not_import_torch():
    text = Path("src/ui_rag/indexing/hnsw_build.py").read_text(encoding="utf-8")
    assert "import faiss" in text
    assert "import torch" not in text
