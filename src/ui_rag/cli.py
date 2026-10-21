from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from ui_rag.config import (
    DEFAULT_ARTIFACT_NAME,
    DEFAULT_MODEL_NAME,
    DEFAULT_PRETRAINED,
    LEGACY_IMAGE_MAP,
    M6_CANDIDATE_K,
    M6_EVIDENCE_K,
    M6_VLM_MAX_IMAGE_PIXELS,
    M6_VLM_MODEL,
    METADATA_DIR,
    RICO_DIR,
    SCREEN2WORDS_SUMMARIES,
    SCREEN2WORDS_TEST_SCREENS,
)


def _embed_command(args: argparse.Namespace) -> None:
    # Lazy import: this child owns torch/selected-encoder and must not import faiss.
    from ui_rag.embeddings.build import build_embeddings

    stage = build_embeddings(
        image_dir=Path(args.image_dir),
        mapping_path=Path(args.mapping),
        artifact_name=args.artifact_name,
        model_name=args.model_name,
        pretrained=args.pretrained,
        encoder_backend=args.encoder_backend,
        max_num_patches=args.max_num_patches,
        batch_size=args.batch_size,
        device=args.device,
        limit=args.limit,
        restart=args.restart,
    )
    print(json.dumps(stage, ensure_ascii=False, indent=2))


def _build_faiss_command(args: argparse.Namespace) -> None:
    # Lazy import: this child owns faiss/numpy and must not import torch/selected-encoder.
    from ui_rag.indexing.faiss_build import build_faiss_index

    manifest = build_faiss_index(
        artifact_name=args.artifact_name,
        add_batch_size=args.add_batch_size,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def _index_command(args: argparse.Namespace) -> None:
    """Compatibility command that orchestrates two isolated child processes."""
    embed_cmd = [
        sys.executable,
        "-m",
        "ui_rag",
        "embed",
        "--image-dir",
        args.image_dir,
        "--mapping",
        args.mapping,
        "--artifact-name",
        args.artifact_name,
        "--model-name",
        args.model_name,
        "--pretrained",
        args.pretrained,
        "--batch-size",
        str(args.batch_size),
    ]
    if args.encoder_backend:
        embed_cmd += ["--encoder-backend", args.encoder_backend]
    if args.max_num_patches is not None:
        embed_cmd += ["--max-num-patches", str(args.max_num_patches)]
    if args.limit is not None:
        embed_cmd += ["--limit", str(args.limit)]
    if args.device:
        embed_cmd += ["--device", args.device]
    if args.restart:
        embed_cmd += ["--restart"]
    subprocess.run(embed_cmd, check=True)

    faiss_cmd = [
        sys.executable,
        "-m",
        "ui_rag",
        "build-faiss",
        "--artifact-name",
        args.artifact_name,
        "--add-batch-size",
        str(args.add_batch_size),
    ]
    subprocess.run(faiss_cmd, check=True)




def _build_hnsw_command(args: argparse.Namespace) -> None:
    # Lazy import: M4 HNSW construction is FAISS/NumPy-only and reuses frozen embeddings.
    from ui_rag.indexing.hnsw_build import build_hnsw_from_manifest

    manifest = build_hnsw_from_manifest(
        source_manifest_path=Path(args.source_manifest),
        artifact_name=args.artifact_name,
        m=args.m,
        ef_construction=args.ef_construction,
        add_batch_size=args.add_batch_size,
        threads=args.threads,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def _encode_query_command(args: argparse.Namespace) -> None:
    from ui_rag.retrieval.query_encoder import encode_query_to_file

    payload = encode_query_to_file(
        manifest_path=Path(args.manifest),
        output_path=Path(args.output),
        query=args.query,
        image_path=Path(args.image) if args.image else None,
        text_weight=args.text_weight,
        device=args.device,
    )
    if args.json:
        print(json.dumps(payload, ensure_ascii=False))
    else:
        print(
            f"query embedding saved: {payload['output']} "
            f"device={payload['device']} dim={payload['dimension']}"
        )


def _faiss_search_command(args: argparse.Namespace) -> None:
    from ui_rag.retrieval.faiss_search import search_faiss_from_file

    results = search_faiss_from_file(
        manifest_path=Path(args.manifest),
        query_vector_path=Path(args.query_vector),
        top_k=args.top_k,
        verify_hashes=not args.skip_hash_check,
    )
    payload = {"results": results}
    if args.json:
        print(json.dumps(payload, ensure_ascii=False))
        return
    for item in results:
        print(
            f"{item['rank']:>2}. score={item['score']:.6f} "
            f"faiss_id={item['faiss_id']} file={item['filename']}"
        )


def _search_command(args: argparse.Namespace) -> None:
    # Parent process imports neither torch nor faiss; native stacks live in children.
    from ui_rag.retrieval.orchestrator import search_isolated

    payload = search_isolated(
        manifest_path=Path(args.manifest),
        query=args.query,
        image_path=Path(args.image) if args.image else None,
        top_k=args.top_k,
        text_weight=args.text_weight,
        device=args.device,
        verify_hashes=not args.skip_hash_check,
    )
    if args.json:
        print(json.dumps(payload, ensure_ascii=False))
        return
    print(
        f"query_encoder_device={payload['encoder']['device']} "
        f"query_type={payload['encoder']['query_type']}"
    )
    for item in payload["results"]:
        print(
            f"{item['rank']:>2}. score={item['score']:.6f} "
            f"faiss_id={item['faiss_id']} file={item['filename']}"
        )



def _benchmark_embed_command(args: argparse.Namespace) -> None:
    # Lazy import: this child owns torch/selected-encoder and must not import faiss.
    from ui_rag.evaluation.embed_queries import embed_benchmark_queries

    payload = embed_benchmark_queries(
        manifest_path=Path(args.manifest),
        queries_path=Path(args.queries),
        output_path=Path(args.output),
        stage_path=Path(args.stage_output),
        batch_size=args.batch_size,
        device=args.device,
    )
    print(json.dumps(payload, ensure_ascii=False))


def _benchmark_eval_command(args: argparse.Namespace) -> None:
    # Lazy import: this child owns faiss/numpy and must not import torch/selected-encoder.
    from ui_rag.evaluation.faiss_eval import evaluate_faiss_benchmark

    payload = evaluate_faiss_benchmark(
        manifest_path=Path(args.manifest),
        queries_path=Path(args.queries),
        query_embeddings_path=Path(args.query_embeddings),
        query_stage_path=Path(args.query_stage),
        output_dir=Path(args.output_dir),
        search_k=args.search_k,
        batch_size=args.batch_size,
        verify_hashes=not args.skip_hash_check,
    )
    print(json.dumps(payload, ensure_ascii=False))


def _benchmark_command(args: argparse.Namespace) -> None:
    # Parent owns neither torch nor faiss; benchmark stages run in isolated children.
    from ui_rag.evaluation.orchestrator import run_benchmark_isolated

    payload = run_benchmark_isolated(
        manifest_path=Path(args.manifest),
        summaries_path=Path(args.summaries),
        test_screens_path=Path(args.test_screens),
        legacy_mapping_path=Path(args.mapping),
        run_name=args.run_name,
        variant=args.variant,
        limit_screens=args.limit_screens,
        embedding_batch_size=args.embedding_batch_size,
        search_batch_size=args.search_batch_size,
        search_k=args.search_k,
        device=args.device,
        verify_hashes=not args.skip_hash_check,
        split_name=args.split_name,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _vlm_generate_command(args: argparse.Namespace) -> None:
    # Dedicated child process: owns MLX VLM only; no FAISS/Torch imports.
    from ui_rag.rag.providers.mlx_vlm_provider import generate_mlx_vlm

    prompt = Path(args.prompt_file).read_text(encoding="utf-8")
    payload = generate_mlx_vlm(
        prompt=prompt,
        image_paths=[str(Path(x)) for x in args.image],
        model_name=args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        max_image_pixels=args.max_image_pixels,
    )
    if args.json:
        print(json.dumps(payload, ensure_ascii=False))
    else:
        print(payload["text"])




def _vlm_ground_command(args: argparse.Namespace) -> None:
    # Dedicated child: owns MLX only. Selection is categorical; JSON is constructed by code.
    from ui_rag.rag.providers.mlx_vlm_provider import MLXVLMEngine
    from ui_rag.rag.schema import EvidenceItem

    raw = json.loads(Path(args.evidence_file).read_text(encoding="utf-8"))
    evidence = [EvidenceItem.from_mapping(item) for item in raw]
    engine = MLXVLMEngine(args.model)
    if args.evidence_policy == "rank1":
        if not evidence:
            raise ValueError("rank1 evidence policy requires at least one evidence item")
        selected_id = evidence[0].evidence_id
        selection = {
            "policy": "rank1",
            "selected_evidence_id": selected_id,
            "abstain": False,
            "raw_text": None,
        }
        grounding_architecture = "deterministic-rank1 -> single-image grounded VLM answer"
        selector_parse_valid = None
    else:
        selection = engine.select_evidence(
            query=args.query,
            evidence=evidence,
            max_tokens=args.selector_max_tokens,
            max_image_pixels=args.selector_max_image_pixels,
        )
        selected_id = selection["selected_evidence_id"]
        grounding_architecture = "categorical-selector -> deterministic citations -> selected-image answer"
        selector_parse_valid = True
    if selected_id is None:
        payload = {
            "selector": selection,
            "grounding_architecture": grounding_architecture,
            "selector_parse_valid": selector_parse_valid,
            "answer": {
                "answer": "The retrieved screenshots do not provide sufficient visual evidence for this request.",
                "selected_evidence_id": None,
                "evidence_ids": [],
                "observations": [],
                "confidence": None,
                "abstain": True,
                "raw_text": selection["raw_text"],
            },
        }
    else:
        selected = next(x for x in evidence if x.evidence_id == selected_id)
        answer_generation = engine.answer_selected_evidence(
            query=args.query,
            evidence_item=selected,
            max_tokens=args.answer_max_tokens,
            max_image_pixels=args.answer_max_image_pixels,
        )
        answer = {
            "answer": answer_generation["text"],
            "selected_evidence_id": selected_id,
            "evidence_ids": [selected_id],
            "observations": [
                {"evidence_id": selected_id, "claim": answer_generation["text"]}
            ],
            "confidence": None,
            "abstain": False,
            "raw_text": answer_generation["text"],
        }
        payload = {
            "selector": selection,
            "grounding_architecture": grounding_architecture,
            "selector_parse_valid": selector_parse_valid,
            "answer_generation": answer_generation["generation"],
            "answer": answer,
        }
    print(json.dumps(payload, ensure_ascii=False))

def _rag_command(args: argparse.Namespace) -> None:
    # Parent orchestrates isolated retrieval and VLM children.
    from ui_rag.rag.orchestrator import run_grounded_rag

    payload = run_grounded_rag(
        query=args.query,
        base_manifest_path=Path(args.base_manifest),
        rerank_stage_path=Path(args.rerank_stage),
        evidence_k=args.evidence_k,
        candidate_k=args.candidate_k,
        model_name=args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        device=args.device,
        max_image_pixels=args.max_image_pixels,
        evidence_policy=args.evidence_policy,
    )
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return
    answer = payload["answer"]
    print(answer["answer"])
    print(f"selected_evidence_id={answer['selected_evidence_id']} confidence={answer['confidence']} abstain={answer['abstain']}")
    for item in payload["evidence"]:
        print(f"{item['evidence_id']}: rank={item['rank']} file={item['filename']} path={item['path']}")

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ui-rag")
    subparsers = parser.add_subparsers(dest="command", required=True)

    index_parser = subparsers.add_parser(
        "index", help="Build embeddings and FAISS in isolated child processes"
    )
    index_parser.add_argument("--image-dir", default=str(RICO_DIR))
    index_parser.add_argument("--mapping", default=str(LEGACY_IMAGE_MAP))
    index_parser.add_argument("--artifact-name", default=DEFAULT_ARTIFACT_NAME)
    index_parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    index_parser.add_argument("--pretrained", default=DEFAULT_PRETRAINED)
    index_parser.add_argument(
        "--encoder-backend", choices=["open_clip", "transformers_siglip2"], default=None
    )
    index_parser.add_argument("--max-num-patches", type=int, default=None)
    index_parser.add_argument("--batch-size", type=int, default=16)
    index_parser.add_argument("--add-batch-size", type=int, default=4096)
    index_parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default=None)
    index_parser.add_argument("--limit", type=int, default=None)
    index_parser.add_argument(
        "--restart",
        action="store_true",
        help="discard partial/complete embedding state and rebuild",
    )
    index_parser.set_defaults(func=_index_command)

    embed_parser = subparsers.add_parser("embed", help="Encode corpus without importing FAISS")
    embed_parser.add_argument("--image-dir", default=str(RICO_DIR))
    embed_parser.add_argument("--mapping", default=str(LEGACY_IMAGE_MAP))
    embed_parser.add_argument("--artifact-name", default=DEFAULT_ARTIFACT_NAME)
    embed_parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    embed_parser.add_argument("--pretrained", default=DEFAULT_PRETRAINED)
    embed_parser.add_argument(
        "--encoder-backend", choices=["open_clip", "transformers_siglip2"], default=None
    )
    embed_parser.add_argument("--max-num-patches", type=int, default=None)
    embed_parser.add_argument("--batch-size", type=int, default=16)
    embed_parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default=None)
    embed_parser.add_argument("--limit", type=int, default=None)
    embed_parser.add_argument(
        "--restart",
        action="store_true",
        help="discard partial/complete embedding state and rebuild",
    )
    embed_parser.set_defaults(func=_embed_command)

    faiss_parser = subparsers.add_parser(
        "build-faiss", help="Build FAISS from persisted embeddings without importing torch"
    )
    faiss_parser.add_argument("--artifact-name", default=DEFAULT_ARTIFACT_NAME)
    faiss_parser.add_argument("--add-batch-size", type=int, default=4096)
    faiss_parser.set_defaults(func=_build_faiss_command)


    hnsw_parser = subparsers.add_parser(
        "build-hnsw", help="Build HNSW from frozen embeddings in a faiss-only process"
    )
    hnsw_parser.add_argument("--source-manifest", required=True)
    hnsw_parser.add_argument("--artifact-name", required=True)
    hnsw_parser.add_argument("--m", type=int, required=True)
    hnsw_parser.add_argument("--ef-construction", type=int, default=200)
    hnsw_parser.add_argument("--add-batch-size", type=int, default=4096)
    hnsw_parser.add_argument(
        "--threads",
        type=int,
        default=0,
        help="FAISS build threads; 0 keeps the library default",
    )
    hnsw_parser.set_defaults(func=_build_hnsw_command)

    encode_parser = subparsers.add_parser(
        "encode-query", help="Encode one retrieval query in a torch/selected-encoder-only process"
    )
    encode_parser.add_argument("--manifest", default=str(METADATA_DIR / "latest-manifest.json"))
    encode_parser.add_argument("--query", default=None)
    encode_parser.add_argument("--image", default=None)
    encode_parser.add_argument("--text-weight", type=float, default=0.5)
    encode_parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default=None)
    encode_parser.add_argument("--output", required=True)
    encode_parser.add_argument("--json", action="store_true")
    encode_parser.set_defaults(func=_encode_query_command)

    raw_search_parser = subparsers.add_parser(
        "faiss-search", help="Search a persisted query vector in a faiss-only process"
    )
    raw_search_parser.add_argument("--manifest", default=str(METADATA_DIR / "latest-manifest.json"))
    raw_search_parser.add_argument("--query-vector", required=True)
    raw_search_parser.add_argument("--top-k", type=int, default=5)
    raw_search_parser.add_argument("--skip-hash-check", action="store_true")
    raw_search_parser.add_argument("--json", action="store_true")
    raw_search_parser.set_defaults(func=_faiss_search_command)

    search_parser = subparsers.add_parser(
        "search", help="Search via isolated query-encoder and FAISS child processes"
    )
    search_parser.add_argument("--query", default=None)
    search_parser.add_argument("--image", default=None)
    search_parser.add_argument("--top-k", type=int, default=5)
    search_parser.add_argument("--text-weight", type=float, default=0.5)
    search_parser.add_argument("--manifest", default=str(METADATA_DIR / "latest-manifest.json"))
    search_parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default=None)
    search_parser.add_argument("--skip-hash-check", action="store_true")
    search_parser.add_argument("--json", action="store_true")
    search_parser.set_defaults(func=_search_command)

    benchmark_embed_parser = subparsers.add_parser(
        "benchmark-embed", help="Encode benchmark text in a torch/selected-encoder-only process"
    )
    benchmark_embed_parser.add_argument("--manifest", required=True)
    benchmark_embed_parser.add_argument("--queries", required=True)
    benchmark_embed_parser.add_argument("--output", required=True)
    benchmark_embed_parser.add_argument("--stage-output", required=True)
    benchmark_embed_parser.add_argument("--batch-size", type=int, default=128)
    benchmark_embed_parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default=None)
    benchmark_embed_parser.set_defaults(func=_benchmark_embed_command)

    benchmark_eval_parser = subparsers.add_parser(
        "benchmark-eval", help="Evaluate persisted query embeddings in a faiss-only process"
    )
    benchmark_eval_parser.add_argument("--manifest", required=True)
    benchmark_eval_parser.add_argument("--queries", required=True)
    benchmark_eval_parser.add_argument("--query-embeddings", required=True)
    benchmark_eval_parser.add_argument("--query-stage", required=True)
    benchmark_eval_parser.add_argument("--output-dir", required=True)
    benchmark_eval_parser.add_argument("--search-k", type=int, default=100)
    benchmark_eval_parser.add_argument("--batch-size", type=int, default=256)
    benchmark_eval_parser.add_argument("--skip-hash-check", action="store_true")
    benchmark_eval_parser.set_defaults(func=_benchmark_eval_command)

    benchmark_parser = subparsers.add_parser(
        "benchmark", help="Run Screen2Words benchmark with macOS-safe native process isolation"
    )
    benchmark_parser.add_argument("--manifest", default=str(METADATA_DIR / "latest-manifest.json"))
    benchmark_parser.add_argument("--summaries", default=str(SCREEN2WORDS_SUMMARIES))
    benchmark_parser.add_argument("--test-screens", default=str(SCREEN2WORDS_TEST_SCREENS))
    benchmark_parser.add_argument("--split-name", default="test")
    benchmark_parser.add_argument("--mapping", default=str(LEGACY_IMAGE_MAP))
    benchmark_parser.add_argument("--run-name", required=True)
    benchmark_parser.add_argument("--variant", choices=["primary", "all"], default="primary")
    benchmark_parser.add_argument("--limit-screens", type=int, default=None)
    benchmark_parser.add_argument("--embedding-batch-size", type=int, default=128)
    benchmark_parser.add_argument("--search-batch-size", type=int, default=256)
    benchmark_parser.add_argument("--search-k", type=int, default=100)
    benchmark_parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default=None)
    benchmark_parser.add_argument("--skip-hash-check", action="store_true")
    benchmark_parser.set_defaults(func=_benchmark_command)

    vlm_parser = subparsers.add_parser(
        "vlm-generate", help="Generate grounded VLM output in an MLX-only child process"
    )
    vlm_parser.add_argument("--prompt-file", required=True)
    vlm_parser.add_argument("--image", action="append", default=[])
    vlm_parser.add_argument("--model", default=M6_VLM_MODEL)
    vlm_parser.add_argument("--max-tokens", type=int, default=300)
    vlm_parser.add_argument("--temperature", type=float, default=0.0)
    vlm_parser.add_argument("--max-image-pixels", type=int, default=M6_VLM_MAX_IMAGE_PIXELS)
    vlm_parser.add_argument("--json", action="store_true")
    vlm_parser.set_defaults(func=_vlm_generate_command)


    vlm_ground_parser = subparsers.add_parser(
        "vlm-ground", help="Run deterministic/diagnostic evidence policy and single-image grounded answering in one MLX child"
    )
    vlm_ground_parser.add_argument("--query", required=True)
    vlm_ground_parser.add_argument("--evidence-file", required=True)
    vlm_ground_parser.add_argument("--model", default=M6_VLM_MODEL)
    vlm_ground_parser.add_argument(
        "--evidence-policy", choices=["rank1", "vlm-selector"], default="rank1"
    )
    vlm_ground_parser.add_argument("--selector-max-tokens", type=int, default=8)
    vlm_ground_parser.add_argument("--selector-max-image-pixels", type=int, default=M6_VLM_MAX_IMAGE_PIXELS)
    vlm_ground_parser.add_argument("--answer-max-tokens", type=int, default=96)
    vlm_ground_parser.add_argument("--answer-max-image-pixels", type=int, default=M6_VLM_MAX_IMAGE_PIXELS)
    vlm_ground_parser.set_defaults(func=_vlm_ground_command)

    rag_parser = subparsers.add_parser(
        "rag", help="Run frozen M5 retrieval then grounded local VLM evidence analysis"
    )
    rag_parser.add_argument("--query", required=True)
    rag_parser.add_argument(
        "--base-manifest",
        default=str(METADATA_DIR / "siglip2-base-p16-naflex-p256-manifest.json"),
    )
    rag_parser.add_argument(
        "--rerank-stage",
        default=str(METADATA_DIR / "siglip2-base-p16-naflex-p512-rerank-embedding-stage.json"),
    )
    rag_parser.add_argument("--candidate-k", type=int, default=M6_CANDIDATE_K)
    rag_parser.add_argument("--evidence-k", type=int, default=M6_EVIDENCE_K)
    rag_parser.add_argument("--model", default=M6_VLM_MODEL)
    rag_parser.add_argument(
        "--evidence-policy", choices=["rank1", "vlm-selector"], default="rank1"
    )
    rag_parser.add_argument("--max-tokens", type=int, default=300)
    rag_parser.add_argument("--temperature", type=float, default=0.0)
    rag_parser.add_argument("--max-image-pixels", type=int, default=M6_VLM_MAX_IMAGE_PIXELS)
    rag_parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default=None)
    rag_parser.add_argument("--json", action="store_true")
    rag_parser.set_defaults(func=_rag_command)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
