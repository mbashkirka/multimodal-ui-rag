from pathlib import Path


def test_mlx_import_is_lazy():
    source = Path("src/ui_rag/rag/providers/mlx_vlm_provider.py").read_text(encoding="utf-8")
    # No top-level MLX import: ordinary unit tests and retrieval CLI remain usable without MLX.
    lines = source.splitlines()
    before_class = "\n".join(lines[:10])
    assert "from mlx_vlm" not in before_class
    assert "run `make m6-setup`" in source
