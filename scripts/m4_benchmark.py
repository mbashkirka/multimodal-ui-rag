from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from statistics import median

import faiss
import numpy as np

from ui_rag.config import PROJECT_ROOT
from ui_rag.evaluation.ann_metrics import (
    choose_ann_configuration,
    exact_neighbor_recall,
    task_rank_metrics,
    top1_agreement,
)
from ui_rag.evaluation.dataset import read_queries_jsonl
from ui_rag.evaluation.metrics import percentile
from ui_rag.indexing.manifest import read_json, sha256_file, write_json


def _resolve_project_path(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _load_index(manifest: dict) -> faiss.Index:
    path = _resolve_project_path(manifest["index"]["path"])
    if sha256_file(path) != manifest["index"]["sha256"]:
        raise RuntimeError(f"index hash mismatch: {path}")
    return faiss.read_index(str(path))


def _load_mapping(manifest: dict) -> dict[int, str]:
    path = _resolve_project_path(manifest["mapping"]["artifact_path"])
    if sha256_file(path) != manifest["mapping"]["artifact_sha256"]:
        raise RuntimeError(f"mapping hash mismatch: {path}")
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    return {int(k): v for k, v in raw.items()}


def _target_ranks(ids: np.ndarray, target_ids: np.ndarray) -> list[int | None]:
    ranks: list[int | None] = []
    for row, target in zip(ids, target_ids, strict=True):
        pos = np.flatnonzero(row == target)
        ranks.append(int(pos[0]) + 1 if len(pos) else None)
    return ranks


def _search(index: faiss.Index, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    return index.search(np.ascontiguousarray(queries, dtype=np.float32), k)


def _single_query_latency(
    index: faiss.Index,
    queries: np.ndarray,
    *,
    k: int,
    sample_size: int,
    warmup: int,
) -> dict:
    n = min(sample_size, len(queries))
    if n <= 0:
        raise ValueError("latency sample must be positive")
    sample_ids = np.linspace(0, len(queries) - 1, num=n, dtype=np.int64)
    for idx in sample_ids[: min(warmup, n)]:
        index.search(np.ascontiguousarray(queries[idx : idx + 1], dtype=np.float32), k)

    latencies: list[float] = []
    for idx in sample_ids:
        started = time.perf_counter()
        index.search(np.ascontiguousarray(queries[idx : idx + 1], dtype=np.float32), k)
        latencies.append((time.perf_counter() - started) * 1000.0)

    return {
        "sample_queries": n,
        "p50_ms": percentile(latencies, 0.50),
        "p95_ms": percentile(latencies, 0.95),
        "mean_ms": float(sum(latencies) / len(latencies)),
    }


def _batch_latency(
    index: faiss.Index,
    queries: np.ndarray,
    *,
    k: int,
    batch_size: int,
    rounds: int,
) -> dict:
    if batch_size <= 0 or rounds <= 0:
        raise ValueError("batch_size and rounds must be positive")

    warm = np.ascontiguousarray(queries[: min(batch_size, len(queries))], dtype=np.float32)
    index.search(warm, k)

    round_qps: list[float] = []
    batch_latencies_ms: list[float] = []
    for _ in range(rounds):
        started_round = time.perf_counter()
        for start in range(0, len(queries), batch_size):
            batch = np.ascontiguousarray(queries[start : start + batch_size], dtype=np.float32)
            started_batch = time.perf_counter()
            index.search(batch, k)
            batch_latencies_ms.append((time.perf_counter() - started_batch) * 1000.0)
        elapsed = time.perf_counter() - started_round
        round_qps.append(len(queries) / elapsed)

    return {
        "batch_size": batch_size,
        "rounds": rounds,
        "qps_median": float(median(round_qps)),
        "qps_min": float(min(round_qps)),
        "qps_max": float(max(round_qps)),
        "batch_p50_ms": percentile(batch_latencies_ms, 0.50),
        "batch_p95_ms": percentile(batch_latencies_ms, 0.95),
    }


def _index_size_mib(manifest: dict) -> float:
    path = _resolve_project_path(manifest["index"]["path"])
    return path.stat().st_size / (1024.0 * 1024.0)


def _benchmark_one(
    *,
    label: str,
    kind: str,
    index: faiss.Index,
    manifest: dict,
    query_embeddings: np.ndarray,
    target_ids: np.ndarray,
    exact_ids: np.ndarray,
    search_k: int,
    latency_k: int,
    latency_sample: int,
    warmup: int,
    batch_size: int,
    rounds: int,
    exact_task_r10: float,
    exact_single_p50: float | None,
    exact_batch_qps: float | None,
    ef_search: int | None = None,
) -> dict:
    _, ids = _search(index, query_embeddings, search_k)
    ranks = _target_ranks(ids, target_ids)
    task = task_rank_metrics(ranks)

    if kind == "exact":
        ann_quality = {
            "exact_neighbor_recall_at_1": 1.0,
            "exact_neighbor_recall_at_5": 1.0,
            "exact_neighbor_recall_at_10": 1.0,
            "exact_top1_agreement": 1.0,
        }
    else:
        exact_list = exact_ids.tolist()
        ann_list = ids.tolist()
        ann_quality = {
            "exact_neighbor_recall_at_1": exact_neighbor_recall(exact_list, ann_list, 1),
            "exact_neighbor_recall_at_5": exact_neighbor_recall(exact_list, ann_list, 5),
            "exact_neighbor_recall_at_10": exact_neighbor_recall(exact_list, ann_list, 10),
            "exact_top1_agreement": top1_agreement(exact_list, ann_list),
        }

    single = _single_query_latency(
        index,
        query_embeddings,
        k=latency_k,
        sample_size=latency_sample,
        warmup=warmup,
    )
    batch = _batch_latency(
        index,
        query_embeddings,
        k=latency_k,
        batch_size=batch_size,
        rounds=rounds,
    )

    task["recall_at_10_retention_vs_exact"] = (
        task["recall_at_10"] / exact_task_r10 if exact_task_r10 else 0.0
    )
    latency = {
        "single_query": single,
        "batch": batch,
        "single_query_p50_speedup_vs_exact": (
            exact_single_p50 / single["p50_ms"]
            if exact_single_p50 is not None and single["p50_ms"]
            else 1.0
        ),
        "batch_qps_speedup_vs_exact": (
            batch["qps_median"] / exact_batch_qps
            if exact_batch_qps is not None and exact_batch_qps
            else 1.0
        ),
    }

    index_info = {
        "type": manifest["index"]["type"],
        "size_mib": _index_size_mib(manifest),
        "sha256": manifest["index"]["sha256"],
    }
    if kind == "hnsw":
        index_info.update(
            {
                "m": manifest["index"]["m"],
                "ef_construction": manifest["index"]["ef_construction"],
                "ef_search": ef_search,
                "build_seconds": manifest["index"].get("build_seconds"),
            }
        )

    return {
        "label": label,
        "kind": kind,
        "index": index_info,
        "ann_quality": ann_quality,
        "task": task,
        "latency": latency,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exact-manifest", required=True)
    parser.add_argument("--hnsw-manifest", action="append", required=True)
    parser.add_argument("--queries", required=True)
    parser.add_argument("--query-embeddings", required=True)
    parser.add_argument("--query-stage", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--ef-search", nargs="+", type=int, default=[16, 32, 64, 128, 256])
    parser.add_argument("--search-k", type=int, default=10)
    parser.add_argument("--latency-k", type=int, default=10)
    parser.add_argument("--latency-sample", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--min-neighbor-recall", type=float, default=0.95)
    parser.add_argument("--min-task-retention", type=float, default=0.98)
    parser.add_argument("--min-speedup", type=float, default=1.20)
    args = parser.parse_args()

    if args.search_k < 10:
        raise ValueError("M4 requires search_k >= 10")
    if any(value <= 0 for value in args.ef_search):
        raise ValueError("efSearch values must be positive")

    faiss.omp_set_num_threads(args.threads)

    exact_manifest_path = Path(args.exact_manifest).resolve()
    exact_manifest = read_json(exact_manifest_path)
    if exact_manifest["index"]["type"] != "IndexFlatIP":
        raise ValueError("exact manifest must use IndexFlatIP")

    queries_path = Path(args.queries).resolve()
    query_embeddings_path = Path(args.query_embeddings).resolve()
    query_stage_path = Path(args.query_stage).resolve()
    query_stage = read_json(query_stage_path)
    if sha256_file(queries_path) != query_stage["queries_sha256"]:
        raise RuntimeError("queries hash mismatch")
    if sha256_file(query_embeddings_path) != query_stage["output_sha256"]:
        raise RuntimeError("query embedding hash mismatch")

    queries = read_queries_jsonl(queries_path)
    query_embeddings = np.load(query_embeddings_path, mmap_mode="r")
    if query_embeddings.dtype != np.float32 or query_embeddings.ndim != 2:
        raise RuntimeError("query embeddings must be a 2D float32 array")
    if len(query_embeddings) != len(queries):
        raise RuntimeError("query count and embedding count differ")
    if query_embeddings.shape[1] != int(exact_manifest["embedding"]["dimension"]):
        raise RuntimeError("query embedding dimension differs from frozen NaFlex corpus")

    mapping = _load_mapping(exact_manifest)
    filename_to_id = {filename: idx for idx, filename in mapping.items()}
    target_ids = np.asarray([filename_to_id[q["target_filename"]] for q in queries], dtype=np.int64)

    exact_index = _load_index(exact_manifest)
    if int(exact_index.ntotal) != len(mapping):
        raise RuntimeError("exact index/map count mismatch")
    _, exact_ids = _search(exact_index, query_embeddings, args.search_k)
    exact_ranks = _target_ranks(exact_ids, target_ids)
    exact_task = task_rank_metrics(exact_ranks)

    exact_row = _benchmark_one(
        label="FlatIP exact",
        kind="exact",
        index=exact_index,
        manifest=exact_manifest,
        query_embeddings=query_embeddings,
        target_ids=target_ids,
        exact_ids=exact_ids,
        search_k=args.search_k,
        latency_k=args.latency_k,
        latency_sample=args.latency_sample,
        warmup=args.warmup,
        batch_size=args.batch_size,
        rounds=args.rounds,
        exact_task_r10=exact_task["recall_at_10"],
        exact_single_p50=None,
        exact_batch_qps=None,
    )
    exact_single_p50 = exact_row["latency"]["single_query"]["p50_ms"]
    exact_batch_qps = exact_row["latency"]["batch"]["qps_median"]

    # Make exact speedup/retention fields definitionally 1.0.
    exact_row["task"]["recall_at_10_retention_vs_exact"] = 1.0
    exact_row["latency"]["single_query_p50_speedup_vs_exact"] = 1.0
    exact_row["latency"]["batch_qps_speedup_vs_exact"] = 1.0

    rows: list[dict] = [exact_row]
    for hnsw_manifest_value in args.hnsw_manifest:
        manifest_path = Path(hnsw_manifest_value).resolve()
        manifest = read_json(manifest_path)
        if manifest["index"]["type"] != "IndexHNSWFlat":
            raise ValueError(f"not an HNSW manifest: {manifest_path}")
        if manifest["embedding"]["artifact_sha256"] != exact_manifest["embedding"]["artifact_sha256"]:
            raise RuntimeError("HNSW and exact indexes do not share identical frozen embeddings")
        if manifest["mapping"]["artifact_sha256"] != exact_manifest["mapping"]["artifact_sha256"]:
            raise RuntimeError("HNSW and exact indexes do not share identical mapping")

        index = _load_index(manifest)
        for ef_search in args.ef_search:
            index.hnsw.efSearch = int(ef_search)
            label = f"HNSW M{manifest['index']['m']} efS{ef_search}"
            rows.append(
                _benchmark_one(
                    label=label,
                    kind="hnsw",
                    index=index,
                    manifest=manifest,
                    query_embeddings=query_embeddings,
                    target_ids=target_ids,
                    exact_ids=exact_ids,
                    search_k=args.search_k,
                    latency_k=args.latency_k,
                    latency_sample=args.latency_sample,
                    warmup=args.warmup,
                    batch_size=args.batch_size,
                    rounds=args.rounds,
                    exact_task_r10=exact_task["recall_at_10"],
                    exact_single_p50=exact_single_p50,
                    exact_batch_qps=exact_batch_qps,
                    ef_search=ef_search,
                )
            )

    selection = choose_ann_configuration(
        rows,
        min_neighbor_recall_at_10=args.min_neighbor_recall,
        min_task_recall10_retention=args.min_task_retention,
        min_single_query_speedup=args.min_speedup,
    )

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "benchmark": {
            "milestone": "M4",
            "encoder_frozen": {
                "backend": exact_manifest["embedding"].get("backend"),
                "model_name": exact_manifest["embedding"]["model_name"],
                "max_num_patches": exact_manifest["embedding"].get("max_num_patches"),
                "embedding_sha256": exact_manifest["embedding"]["artifact_sha256"],
            },
            "queries": len(queries),
            "search_k": args.search_k,
            "latency_k": args.latency_k,
            "threads": args.threads,
            "ef_search_values": args.ef_search,
            "quality_constraints": {
                "min_exact_neighbor_recall_at_10": args.min_neighbor_recall,
                "min_task_recall10_retention_vs_exact": args.min_task_retention,
                "min_single_query_p50_speedup_vs_exact": args.min_speedup,
            },
            "note": (
                "M4 compares index approximation only. Query embeddings are reused from the frozen "
                "M3 NaFlex benchmark, so encoder latency is intentionally excluded."
            ),
        },
        "rows": rows,
        "selection": selection,
    }
    json_path = output_dir / "comparison.json"
    csv_path = output_dir / "comparison.csv"
    write_json(json_path, payload)

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "label",
                "kind",
                "M",
                "efConstruction",
                "efSearch",
                "neighbor_R@10",
                "top1_agreement",
                "task_R@10",
                "task_R@10_retention",
                "MRR@10",
                "nDCG@10",
                "single_p50_ms",
                "single_p95_ms",
                "single_speedup",
                "batch_qps",
                "batch_speedup",
                "index_size_mib",
                "build_seconds",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row["label"],
                    row["kind"],
                    row["index"].get("m", ""),
                    row["index"].get("ef_construction", ""),
                    row["index"].get("ef_search", ""),
                    row["ann_quality"]["exact_neighbor_recall_at_10"],
                    row["ann_quality"]["exact_top1_agreement"],
                    row["task"]["recall_at_10"],
                    row["task"]["recall_at_10_retention_vs_exact"],
                    row["task"]["mrr_at_10"],
                    row["task"]["ndcg_at_10"],
                    row["latency"]["single_query"]["p50_ms"],
                    row["latency"]["single_query"]["p95_ms"],
                    row["latency"]["single_query_p50_speedup_vs_exact"],
                    row["latency"]["batch"]["qps_median"],
                    row["latency"]["batch_qps_speedup_vs_exact"],
                    row["index"]["size_mib"],
                    row["index"].get("build_seconds", ""),
                ]
            )

    print("M4 exact-vs-HNSW comparison")
    print(
        "label                  neighR10 taskR10 retain  single_p50  speedup  batch_qps  sizeMiB"
    )
    for row in rows:
        print(
            f"{row['label']:<22} "
            f"{row['ann_quality']['exact_neighbor_recall_at_10']*100:7.2f}% "
            f"{row['task']['recall_at_10']*100:7.2f}% "
            f"{row['task']['recall_at_10_retention_vs_exact']*100:6.1f}% "
            f"{row['latency']['single_query']['p50_ms']:10.4f} "
            f"{row['latency']['single_query_p50_speedup_vs_exact']:7.2f}x "
            f"{row['latency']['batch']['qps_median']:10.1f} "
            f"{row['index']['size_mib']:8.2f}"
        )
    print(f"recommendation={selection['recommended_label']}")
    print(f"reason={selection['reason']}")
    print(f"json={json_path.relative_to(PROJECT_ROOT)}")
    print(f"csv={csv_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
