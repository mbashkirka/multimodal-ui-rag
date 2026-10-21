from ui_rag.config import PROJECT_ROOT


def test_embedding_stage_exposes_resume_state_without_faiss() -> None:
    source = (PROJECT_ROOT / "src/ui_rag/embeddings/build.py").read_text(encoding="utf-8")
    assert "embedding-progress.json" in source
    assert ".partial.npy" in source
    assert "Resuming embeddings at row" in source
    assert "import faiss" not in source


def test_full_make_target_does_not_restart_by_default() -> None:
    makefile = (PROJECT_ROOT / "Makefile").read_text(encoding="utf-8")
    full_block = makefile.split("index-full:", 1)[1].split("index-full-restart:", 1)[0]
    assert "--restart" not in full_block
