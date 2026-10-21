from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from ui_rag.indexing.manifest import read_json, sha256_file
from ui_rag.reranking.multires import reciprocal_rank_fusion_order


def _run_json(cmd: list[str]) -> dict:
    proc = subprocess.run(cmd, check=True, text=True, capture_output=True)
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"child did not return JSON: {' '.join(cmd)}\n{proc.stdout}\n{proc.stderr}") from exc


def retrieve_m5_text(
    *,
    query: str,
    base_manifest_path: Path,
    rerank_stage_path: Path,
    candidate_k: int = 20,
    base_weight: float = 0.5,
    rrf_constant: int = 60,
    top_k: int = 5,
    device: str | None = None,
    verify_hashes: bool = True,
) -> dict:
    """Run the frozen M5 production retriever without co-loading torch and FAISS."""
    total_started = time.perf_counter()
    if top_k <= 0 or candidate_k < top_k:
        raise ValueError("candidate_k must be >= top_k > 0")
    base_manifest = read_json(base_manifest_path)
    rerank_stage = read_json(rerank_stage_path)
    if base_manifest["mapping"]["artifact_sha256"] != rerank_stage["mapping"]["artifact_sha256"]:
        raise RuntimeError("base/reranker mappings are not aligned")
    rerank_path = Path(rerank_stage["embedding"]["path"])
    if not rerank_path.is_absolute():
        from ui_rag.config import PROJECT_ROOT
        rerank_path = PROJECT_ROOT / rerank_path
    if verify_hashes and sha256_file(rerank_path) != rerank_stage["embedding"]["sha256"]:
        raise RuntimeError("p512 reranker embedding hash mismatch")

    with tempfile.TemporaryDirectory(prefix="ui-rag-m6-retrieve-") as tmp:
        q_path = Path(tmp) / "query.npy"
        encode_cmd = [
            sys.executable, "-m", "ui_rag", "encode-query",
            "--manifest", str(base_manifest_path), "--query", query,
            "--output", str(q_path), "--json",
        ]
        if device:
            encode_cmd += ["--device", device]
        encode_started = time.perf_counter()
        encoder = _run_json(encode_cmd)
        encode_ms = (time.perf_counter() - encode_started) * 1000.0
        search_started = time.perf_counter()
        search_cmd = [
            sys.executable, "-m", "ui_rag", "faiss-search",
            "--manifest", str(base_manifest_path), "--query-vector", str(q_path),
            "--top-k", str(candidate_k), "--json",
        ]
        if not verify_hashes:
            search_cmd.append("--skip-hash-check")
        search = _run_json(search_cmd)
        search_ms = (time.perf_counter() - search_started) * 1000.0
        q = np.asarray(np.load(q_path), dtype=np.float32).reshape(-1)

    rerank_started = time.perf_counter()
    rows = search["results"]
    ids = np.asarray([int(x["faiss_id"]) for x in rows], dtype=np.int64)
    rerank_emb = np.load(rerank_path, mmap_mode="r")
    fine_scores = np.asarray(rerank_emb[ids] @ q, dtype=np.float32)
    ordered_ids, fused_scores = reciprocal_rank_fusion_order(
        ids, fine_scores, base_weight=base_weight, rrf_constant=rrf_constant
    )
    by_id = {int(x["faiss_id"]): x for x in rows}
    reranked: list[dict] = []
    for rank, (faiss_id, fused) in enumerate(zip(ordered_ids[:top_k], fused_scores[:top_k]), start=1):
        row = dict(by_id[int(faiss_id)])
        row["base_rank"] = int(row["rank"])
        row["rank"] = rank
        row["base_score"] = float(row["score"])
        row["score"] = float(fused)
        row["fine_score"] = float(fine_scores[np.flatnonzero(ids == faiss_id)[0]])
        reranked.append(row)
    rerank_ms = (time.perf_counter() - rerank_started) * 1000.0
    total_ms = (time.perf_counter() - total_started) * 1000.0
    return {
        "query": query,
        "encoder": encoder,
        "retrieval": {
            "stage1": "SigLIP2 Base NaFlex p256 + FlatIP",
            "candidate_k": candidate_k,
            "stage2": "same checkpoint p512 corpus rescoring",
            "base_weight": base_weight,
            "fine_weight": 1.0 - base_weight,
            "rrf_constant": rrf_constant,
        },
        "results": reranked,
        "timings_ms": {
            "encode_ms": encode_ms,
            "search_ms": search_ms,
            "rerank_ms": rerank_ms,
            "total_ms": total_ms,
        },
    }
