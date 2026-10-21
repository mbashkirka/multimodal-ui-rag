from pathlib import Path


def test_m6_retrieval_keeps_native_boundaries():
    source = Path("src/ui_rag/rag/retrieve.py").read_text(encoding="utf-8")
    assert "faiss" not in source.split("import numpy", 1)[0]
    assert '"encode-query"' in source
    assert '"faiss-search"' in source
    assert "reciprocal_rank_fusion_order" in source


def test_m6_vlm_is_child_process_from_rag_orchestrator():
    source = Path("src/ui_rag/rag/orchestrator.py").read_text(encoding="utf-8")
    assert '"vlm-ground"' in source
    assert "subprocess.run" in source
    assert "mlx_vlm" not in source
