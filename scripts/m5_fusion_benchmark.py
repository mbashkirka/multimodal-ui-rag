from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import faiss
import numpy as np

from ui_rag.config import PROJECT_ROOT
from ui_rag.evaluation.dataset import read_queries_jsonl
from ui_rag.evaluation.metrics import summarize_ranks
from ui_rag.fusion.exemplar import choose_nonself_reference_ids, fuse_text_and_reference
from ui_rag.indexing.manifest import read_json, sha256_file, write_json


def _load_map(path: Path) -> dict[int, str]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {int(k): v for k, v in raw.items()}


def _best(rows: list[dict]) -> dict:
    eligible = [row for row in rows if 0.0 < row["text_weight"] < 1.0]
    if not eligible:
        raise ValueError("dev fusion grid must contain at least one true multimodal weight")
    return max(
        eligible,
        key=lambda row: (
            row["metrics"]["recall_at_10"],
            row["metrics"]["mrr_at_10"],
            row["metrics"]["ndcg_at_10"],
            row["text_weight"],
        ),
    )


def evaluate(args: argparse.Namespace) -> dict:
    manifest_path = Path(args.manifest)
    queries_path = Path(args.queries)
    query_embeddings_path = Path(args.query_embeddings)
    query_stage_path = Path(args.query_stage)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = read_json(manifest_path)
    query_stage = read_json(query_stage_path)
    index_path = PROJECT_ROOT / manifest["index"]["path"]
    corpus_embeddings_path = PROJECT_ROOT / manifest["embedding"]["path"]
    mapping_path = PROJECT_ROOT / manifest["mapping"]["artifact_path"]
    for path in (index_path, corpus_embeddings_path, mapping_path, queries_path, query_embeddings_path, query_stage_path):
        if not path.exists():
            raise FileNotFoundError(path)
    if sha256_file(index_path) != manifest["index"]["sha256"]:
        raise RuntimeError("index hash mismatch")
    if sha256_file(corpus_embeddings_path) != manifest["embedding"]["sha256"]:
        raise RuntimeError("corpus embedding hash mismatch")
    if sha256_file(mapping_path) != manifest["mapping"]["artifact_sha256"]:
        raise RuntimeError("mapping hash mismatch")
    if sha256_file(queries_path) != query_stage["queries_sha256"]:
        raise RuntimeError("query file changed after embedding")
    if sha256_file(query_embeddings_path) != query_stage["output_sha256"]:
        raise RuntimeError("query embedding hash mismatch")

    queries = read_queries_jsonl(queries_path)
    image_map = _load_map(mapping_path)
    filename_to_id = {name: idx for idx, name in image_map.items()}
    target_ids = np.asarray([filename_to_id[q["target_filename"]] for q in queries], dtype=np.int64)
    q_text = np.load(query_embeddings_path, mmap_mode="r")
    corpus = np.load(corpus_embeddings_path, mmap_mode="r")
    dim = int(manifest["embedding"]["dimension"])
    if q_text.shape != (len(queries), dim):
        raise RuntimeError(f"query shape mismatch: {q_text.shape}")
    if corpus.shape != (int(manifest["index"]["num_vectors"]), dim):
        raise RuntimeError(f"corpus shape mismatch: {corpus.shape}")

    index = faiss.read_index(str(index_path))
    faiss.omp_set_num_threads(args.threads)

    # Build a controlled visual exemplar for every target: the nearest *other*
    # corpus screen under the frozen p256 image embedding. The target itself is
    # never used as the image query, preventing trivial self-retrieval.
    target_vectors = np.ascontiguousarray(corpus[target_ids], dtype=np.float32)
    ref_started = time.perf_counter()
    _, neighbor_ids = index.search(target_vectors, args.reference_search_k)
    reference_ids = choose_nonself_reference_ids(neighbor_ids, target_ids)
    reference_seconds = time.perf_counter() - ref_started
    q_image = np.ascontiguousarray(corpus[reference_ids], dtype=np.float32)
    ref_cos = np.sum(target_vectors * q_image, axis=1)

    if args.selection:
        selection = read_json(Path(args.selection))
        weights = [float(selection["selected"]["text_weight"])]
        mode = "frozen_test"
    else:
        weights = args.text_weight
        mode = "dev_selection"

    rows: list[dict] = []
    for weight in weights:
        fused = fuse_text_and_reference(q_text, q_image, text_weight=weight)
        started = time.perf_counter()
        _, ids = index.search(fused, args.search_k)
        search_seconds = time.perf_counter() - started
        ranks: list[int | None] = []
        for i, target in enumerate(target_ids):
            pos = np.flatnonzero(ids[i] == target)
            ranks.append(int(pos[0]) + 1 if len(pos) else None)
        rows.append(
            {
                "text_weight": float(weight),
                "image_weight": float(1.0 - weight),
                "metrics": summarize_ranks(ranks, args.search_k),
                "timing": {
                    "search_seconds": search_seconds,
                    "amortized_search_ms_per_query": search_seconds * 1000.0 / len(queries) if queries else None,
                },
            }
        )

    selected = rows[0] if args.selection else _best(rows)
    payload = {
        "milestone": "M5",
        "track": "multimodal_visual_exemplar_fusion",
        "mode": mode,
        "protocol": {
            "text": "Screen2Words human summary",
            "image": "nearest non-self RICO screen to the target under frozen p256 image embeddings",
            "fusion": "L2-normalized weighted sum in the shared SigLIP2 embedding space",
            "selection_split": "dev only" if mode == "dev_selection" else "frozen from dev selection",
            "caveat": (
                "The visual exemplar is constructed with knowledge of the target by selecting its nearest non-self "
                "neighbor. This is a controlled disambiguation/stress benchmark, not an unbiased estimate of real-world "
                "reference-image availability."
            ),
        },
        "num_queries": len(queries),
        "reference": {
            "search_k": args.reference_search_k,
            "construction_seconds": reference_seconds,
            "mean_target_reference_cosine": float(np.mean(ref_cos)),
            "median_target_reference_cosine": float(np.median(ref_cos)),
            "min_target_reference_cosine": float(np.min(ref_cos)),
            "max_target_reference_cosine": float(np.max(ref_cos)),
            "self_reference_count": int(np.sum(reference_ids == target_ids)),
        },
        "grid": rows,
        "selected": selected,
        "artifacts": {
            "manifest": str(manifest_path),
            "queries": str(queries_path),
            "query_embeddings": str(query_embeddings_path),
        },
    }
    write_json(output_dir / "comparison.json", payload)
    np.save(output_dir / "reference_ids.npy", reference_ids)
    with (output_dir / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["text_weight", "image_weight", "R@1", "R@5", "R@10", "R@50", "R@100", "MRR@10", "nDCG@10", "search_ms"])
        for row in rows:
            m = row["metrics"]
            writer.writerow([
                row["text_weight"], row["image_weight"], m["recall_at_1"], m["recall_at_5"], m["recall_at_10"],
                m["recall_at_50"], m["recall_at_100"], m["mrr_at_10"], m["ndcg_at_10"], row["timing"]["amortized_search_ms_per_query"],
            ])
    if not args.selection:
        write_json(
            output_dir / "selection.json",
            {
                "selected_on": "Screen2Words dev controlled visual-exemplar benchmark",
                "primary_metric": "recall_at_10",
                "tie_breakers": ["mrr_at_10", "ndcg_at_10", "larger_text_weight"],
                "selected": selected,
            },
        )

    print("M5 multimodal visual-exemplar fusion")
    print(
        f"references={len(reference_ids)} self_refs={int(np.sum(reference_ids == target_ids))} "
        f"median_target_ref_cos={float(np.median(ref_cos)):.4f}"
    )
    print("text_w image_w  R@1    R@5    R@10   R@100  MRR@10 nDCG@10 search_ms")
    for row in rows:
        m = row["metrics"]
        print(
            f"{row['text_weight']:>6.2f} {row['image_weight']:>7.2f} "
            f"{m['recall_at_1']*100:>6.2f}% {m['recall_at_5']*100:>6.2f}% "
            f"{m['recall_at_10']*100:>6.2f}% {m['recall_at_100']*100:>6.2f}% "
            f"{m['mrr_at_10']:>7.4f} {m['ndcg_at_10']:>8.4f} "
            f"{row['timing']['amortized_search_ms_per_query']:>9.4f}"
        )
    print(
        f"selected=text_w={selected['text_weight']:.2f}/image_w={selected['image_weight']:.2f} "
        f"R@10={selected['metrics']['recall_at_10']*100:.2f}%"
    )
    print(f"json={output_dir / 'comparison.json'}")
    return payload


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--queries", required=True)
    p.add_argument("--query-embeddings", required=True)
    p.add_argument("--query-stage", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--text-weight", nargs="+", type=float, default=[0.0, 0.25, 0.5, 0.75, 0.9, 1.0])
    p.add_argument("--selection", default=None)
    p.add_argument("--reference-search-k", type=int, default=8)
    p.add_argument("--search-k", type=int, default=100)
    p.add_argument("--threads", type=int, default=1)
    return p


if __name__ == "__main__":
    evaluate(build_parser().parse_args())
