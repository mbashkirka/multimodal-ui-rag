from pathlib import Path


def test_faiss_contract_is_subprocess_isolated() -> None:
    source = Path("tests/test_faiss_contract.py").read_text(encoding="utf-8")
    top_level_prefix = source.split("def test_flat_ip", 1)[0]

    # The literal import is allowed only inside the child-process code string,
    # never in the module prefix executed during pytest collection.
    assert "import faiss" not in top_level_prefix
    assert "subprocess.run" in source
    assert "sys.executable" in source
