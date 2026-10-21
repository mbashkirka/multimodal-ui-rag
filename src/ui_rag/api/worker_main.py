from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any


def _emit(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def _serve(kind: str, handler: Callable[[dict[str, Any]], dict[str, Any]]) -> None:
    _emit({"status": "ready", "worker": kind})
    for line in sys.stdin:
        message: dict[str, Any] | None = None
        request_id = None
        try:
            message = json.loads(line)
            request_id = message.get("request_id") if isinstance(message, dict) else None
            if not isinstance(message, dict):
                raise TypeError("worker request must be a JSON object")
            if message.get("op") == "shutdown":
                return
            result = handler(message)
            _emit({"request_id": request_id, "ok": True, "result": result})
        except Exception as exc:  # noqa: BLE001 - process boundary serializes worker failures
            _emit(
                {
                    "request_id": request_id,
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )


def _encoder_handler(manifest_path: Path) -> Callable[[dict[str, Any]], dict[str, Any]]:
    # torch/transformers are imported only in this process.
    import numpy as np

    from ui_rag.embeddings.factory import create_encoder, encoder_kwargs_from_manifest
    from ui_rag.indexing.manifest import read_json

    manifest = read_json(manifest_path)
    cfg = manifest["embedding"]
    encoder = create_encoder(**encoder_kwargs_from_manifest(cfg), device=None)
    dimension = int(cfg["dimension"])

    def handle(message: dict[str, Any]) -> dict[str, Any]:
        query = str(message["query"])
        started = time.perf_counter()
        vector = np.asarray(encoder.encode_text(query), dtype=np.float32)
        if vector.shape != (1, dimension):
            raise RuntimeError(f"query embedding shape mismatch: {vector.shape}")
        return {
            "vector": vector.reshape(-1).tolist(),
            "dimension": dimension,
            "device": encoder.device,
            "elapsed_ms": (time.perf_counter() - started) * 1000.0,
        }

    return handle


def _faiss_handler(manifest_path: Path) -> Callable[[dict[str, Any]], dict[str, Any]]:
    # faiss/numpy are imported only in this process; torch is never imported here.
    import faiss
    import numpy as np

    from ui_rag.config import PROJECT_ROOT
    from ui_rag.indexing.manifest import read_json

    manifest = read_json(manifest_path)

    def resolve(raw: str) -> Path:
        path = Path(raw)
        return path if path.is_absolute() else PROJECT_ROOT / path

    index = faiss.read_index(str(resolve(manifest["index"]["path"])))
    with resolve(manifest["mapping"]["artifact_path"]).open("r", encoding="utf-8") as handle:
        image_map = json.load(handle)
    dimension = int(manifest["embedding"]["dimension"])
    if index.d != dimension or index.ntotal != len(image_map):
        raise RuntimeError("FAISS index/map/manifest mismatch")

    def handle(message: dict[str, Any]) -> dict[str, Any]:
        vector = np.asarray(message["vector"], dtype=np.float32).reshape(1, -1)
        if vector.shape != (1, dimension):
            raise ValueError(f"invalid query vector shape: {vector.shape}")
        top_k = min(int(message["top_k"]), int(index.ntotal))
        started = time.perf_counter()
        scores, indices = index.search(np.ascontiguousarray(vector), top_k)
        rows = []
        for rank, (score, faiss_id) in enumerate(zip(scores[0], indices[0]), start=1):
            if int(faiss_id) < 0:
                continue
            rows.append({
                "rank": rank,
                "faiss_id": int(faiss_id),
                "filename": image_map[str(int(faiss_id))],
                "score": float(score),
            })
        return {
            "results": rows,
            "elapsed_ms": (time.perf_counter() - started) * 1000.0,
        }

    return handle


def _vlm_handler(model_name: str) -> Callable[[dict[str, Any]], dict[str, Any]]:
    # MLX is imported only in this process.
    from ui_rag.rag.providers.mlx_vlm_provider import MLXVLMEngine
    from ui_rag.rag.schema import EvidenceItem

    engine = MLXVLMEngine(model_name)

    def handle(message: dict[str, Any]) -> dict[str, Any]:
        evidence = EvidenceItem.from_mapping(message["evidence"])
        started = time.perf_counter()
        answer = engine.answer_selected_evidence(
            query=str(message["query"]),
            evidence_item=evidence,
            max_tokens=int(message["max_tokens"]),
            max_image_pixels=int(message["max_image_pixels"]),
        )
        return {
            "text": answer["text"],
            "generation": answer["generation"],
            "elapsed_ms": (time.perf_counter() - started) * 1000.0,
        }

    return handle


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="kind", required=True)
    encoder = sub.add_parser("encoder")
    encoder.add_argument("--manifest", required=True)
    faiss_parser = sub.add_parser("faiss")
    faiss_parser.add_argument("--manifest", required=True)
    vlm = sub.add_parser("vlm")
    vlm.add_argument("--model", required=True)
    args = parser.parse_args()

    if args.kind == "encoder":
        _serve("encoder", _encoder_handler(Path(args.manifest)))
    elif args.kind == "faiss":
        _serve("faiss", _faiss_handler(Path(args.manifest)))
    elif args.kind == "vlm":
        _serve("vlm", _vlm_handler(args.model))
    else:  # pragma: no cover
        raise AssertionError(args.kind)


if __name__ == "__main__":
    main()
