from pathlib import Path

from ui_rag.evaluation.dataset import prepare_screen2words_queries


def test_screen2words_dev_split_name_is_recorded(tmp_path: Path):
    summaries = tmp_path / "summaries.csv"
    summaries.write_text(
        "screenId,summary\n10,first summary\n10,second summary\n10,third summary\n10,fourth summary\n10,fifth summary\n",
        encoding="utf-8",
    )
    dev = tmp_path / "dev.txt"
    dev.write_text("10\n", encoding="utf-8")
    mapping = tmp_path / "mapping.json"
    mapping.write_text('{"0":"10.jpg"}', encoding="utf-8")
    queries, meta = prepare_screen2words_queries(
        summaries_path=summaries,
        test_screens_path=dev,
        legacy_mapping_path=mapping,
        variant="primary",
        split_name="dev",
    )
    assert meta["split"] == "dev"
    assert queries[0].query_id.startswith("s2w-dev-")


def test_m5_makefile_keeps_dev_selection_separate_from_test():
    root = Path(__file__).resolve().parents[1]
    makefile = (root / "Makefile").read_text(encoding="utf-8")
    assert "--split-name dev" in makefile
    assert "m5-rerank-dev" in makefile
    assert "m5-rerank-test" in makefile
    assert "--selection artifacts/evaluation/m5-rerank-dev/selection.json" in makefile
    assert "m5-fusion-dev" in makefile
    assert "m5-fusion-test" in makefile
