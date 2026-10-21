from ui_rag.config import PROJECT_ROOT


def _source(relative: str) -> str:
    return (PROJECT_ROOT / relative).read_text(encoding="utf-8")


def test_embedding_stage_does_not_import_faiss() -> None:
    source = _source("src/ui_rag/embeddings/build.py")
    assert "import faiss" not in source
    assert "from faiss" not in source


def test_faiss_stage_does_not_import_torch_or_open_clip() -> None:
    source = _source("src/ui_rag/indexing/faiss_build.py")
    assert "import torch" not in source
    assert "import open_clip" not in source


def test_cli_has_no_eager_native_ml_imports() -> None:
    source = _source("src/ui_rag/cli.py")
    top = source.split("def _embed_command", 1)[0]
    assert "import faiss" not in top
    assert "import torch" not in top
    assert "import open_clip" not in top
