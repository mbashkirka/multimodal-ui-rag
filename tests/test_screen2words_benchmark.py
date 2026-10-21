import csv
import json
from pathlib import Path

from ui_rag.evaluation.dataset import prepare_screen2words_queries


def test_prepare_primary_and_all_variants(tmp_path: Path):
    summaries = tmp_path / "summaries.csv"
    with summaries.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["screenId", "summary"])
        for screen_id in ["10", "20"]:
            for idx in range(5):
                writer.writerow([screen_id, f"summary {idx} for {screen_id}"])

    test_screens = tmp_path / "test.txt"
    test_screens.write_text("10\n20\n", encoding="utf-8")

    mapping = tmp_path / "mapping.json"
    mapping.write_text(json.dumps({"0": "10.jpg", "1": "20.jpg"}), encoding="utf-8")

    primary, meta_primary = prepare_screen2words_queries(
        summaries_path=summaries,
        test_screens_path=test_screens,
        legacy_mapping_path=mapping,
        variant="primary",
    )
    all_queries, meta_all = prepare_screen2words_queries(
        summaries_path=summaries,
        test_screens_path=test_screens,
        legacy_mapping_path=mapping,
        variant="all",
    )

    assert len(primary) == 2
    assert len(all_queries) == 10
    assert meta_primary["num_screens"] == 2
    assert meta_all["num_queries"] == 10
    assert primary[0].target_filename == "10.jpg"
