from pathlib import Path


def test_m5_faiss_scripts_do_not_import_torch():
    root = Path(__file__).resolve().parents[1]
    for name in ["m5_rerank_benchmark.py", "m5_fusion_benchmark.py"]:
        source = (root / "scripts" / name).read_text(encoding="utf-8")
        assert "import faiss" in source
        assert "import torch" not in source
        assert "transformers" not in source
