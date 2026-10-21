from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path


def _run_json(command: list[str]) -> dict:
    proc = subprocess.run(command, check=True, text=True, capture_output=True)
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "child process did not return valid JSON\n"
            f"command: {' '.join(command)}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        ) from exc


def search_isolated(
    *,
    manifest_path: Path,
    query: str | None,
    image_path: Path | None,
    top_k: int,
    text_weight: float,
    device: str | None,
    verify_hashes: bool,
) -> dict:
    """Run query encoding and FAISS search in different Python interpreters."""
    with tempfile.TemporaryDirectory(prefix="ui-rag-query-") as tmp_dir:
        vector_path = Path(tmp_dir) / "query.npy"
        encode_cmd = [
            sys.executable,
            "-m",
            "ui_rag",
            "encode-query",
            "--manifest",
            str(manifest_path),
            "--output",
            str(vector_path),
            "--text-weight",
            str(text_weight),
            "--json",
        ]
        if query:
            encode_cmd += ["--query", query]
        if image_path is not None:
            encode_cmd += ["--image", str(image_path)]
        if device:
            encode_cmd += ["--device", device]
        encoder_info = _run_json(encode_cmd)

        search_cmd = [
            sys.executable,
            "-m",
            "ui_rag",
            "faiss-search",
            "--manifest",
            str(manifest_path),
            "--query-vector",
            str(vector_path),
            "--top-k",
            str(top_k),
            "--json",
        ]
        if not verify_hashes:
            search_cmd.append("--skip-hash-check")
        search_payload = _run_json(search_cmd)

    return {
        "query": query,
        "image": str(image_path) if image_path is not None else None,
        "text_weight": text_weight,
        "encoder": encoder_info,
        "results": search_payload["results"],
    }
