from pathlib import Path


def test_benchmark_orchestrator_does_not_import_native_stacks():
    source = Path("src/ui_rag/evaluation/orchestrator.py").read_text(encoding="utf-8")
    assert "import faiss" not in source
    assert "import torch" not in source
    assert "import open_clip" not in source


def test_benchmark_stage_native_ownership():
    embed_source = Path("src/ui_rag/evaluation/embed_queries.py").read_text(encoding="utf-8")
    faiss_source = Path("src/ui_rag/evaluation/faiss_eval.py").read_text(encoding="utf-8")
    assert "import faiss" not in embed_source
    assert "import torch" not in faiss_source
    assert "open_clip" not in faiss_source
