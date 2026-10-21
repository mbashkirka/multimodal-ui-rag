from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from ui_rag.config import PROJECT_ROOT
from ui_rag.evaluation.dataset import prepare_screen2words_queries, write_queries_jsonl
from ui_rag.indexing.manifest import sha256_file, write_json


def prepare_benchmark(
    *,
    summaries_path: Path,
    test_screens_path: Path,
    legacy_mapping_path: Path,
    output_dir: Path,
    variant: str,
    limit_screens: int | None,
    split_name: str = "test",
) -> dict:
    queries, metadata = prepare_screen2words_queries(
        summaries_path=summaries_path,
        test_screens_path=test_screens_path,
        legacy_mapping_path=legacy_mapping_path,
        variant=variant,
        limit_screens=limit_screens,
        split_name=split_name,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    queries_path = output_dir / "queries.jsonl"
    write_queries_jsonl(queries_path, queries)
    metadata = {
        **metadata,
        "queries_path": str(queries_path),
        "queries_sha256": sha256_file(queries_path),
        "source": {
            "screen_summaries": str(summaries_path),
            "screen_summaries_sha256": sha256_file(summaries_path),
            "split_screens": str(test_screens_path),
            "split_screens_sha256": sha256_file(test_screens_path),
            "split_name": split_name,
        },
    }
    write_json(output_dir / "benchmark.json", metadata)
    return metadata


def run_benchmark_isolated(
    *,
    manifest_path: Path,
    summaries_path: Path,
    test_screens_path: Path,
    legacy_mapping_path: Path,
    run_name: str,
    variant: str,
    limit_screens: int | None,
    embedding_batch_size: int,
    search_batch_size: int,
    search_k: int,
    device: str | None,
    verify_hashes: bool,
    split_name: str = "test",
) -> dict:
    """Run prepare -> torch-only query embedding -> faiss-only evaluation."""
    run_dir = PROJECT_ROOT / "artifacts" / "evaluation" / run_name
    metadata = prepare_benchmark(
        summaries_path=summaries_path,
        test_screens_path=test_screens_path,
        legacy_mapping_path=legacy_mapping_path,
        output_dir=run_dir,
        variant=variant,
        limit_screens=limit_screens,
        split_name=split_name,
    )
    queries_path = Path(metadata["queries_path"])
    embeddings_path = run_dir / "query_embeddings.npy"
    query_stage_path = run_dir / "query_embedding_stage.json"

    embed_cmd = [
        sys.executable,
        "-m",
        "ui_rag",
        "benchmark-embed",
        "--manifest",
        str(manifest_path),
        "--queries",
        str(queries_path),
        "--output",
        str(embeddings_path),
        "--stage-output",
        str(query_stage_path),
        "--batch-size",
        str(embedding_batch_size),
    ]
    if device:
        embed_cmd += ["--device", device]
    subprocess.run(embed_cmd, check=True)

    eval_cmd = [
        sys.executable,
        "-m",
        "ui_rag",
        "benchmark-eval",
        "--manifest",
        str(manifest_path),
        "--queries",
        str(queries_path),
        "--query-embeddings",
        str(embeddings_path),
        "--query-stage",
        str(query_stage_path),
        "--output-dir",
        str(run_dir),
        "--search-k",
        str(search_k),
        "--batch-size",
        str(search_batch_size),
    ]
    if not verify_hashes:
        eval_cmd += ["--skip-hash-check"]
    completed = subprocess.run(eval_cmd, check=True, capture_output=True, text=True)
    payload = json.loads(completed.stdout)
    return payload
