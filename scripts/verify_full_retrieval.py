from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from ui_rag.config import PROJECT_ROOT
from ui_rag.indexing.manifest import read_json, sha256_file

DEFAULT_MANIFEST = Path("artifacts/metadata/clip-vit-b32-quickgelu-openai-manifest.json")
EXPECTED_VECTORS = 66_261
EXPECTED_DIMENSION = 512


def run_search(
    *,
    manifest: Path,
    image: Path | None = None,
    query: str | None = None,
    device: str | None = None,
    top_k: int = 5,
) -> dict:
    command = [
        sys.executable,
        "-m",
        "ui_rag",
        "search",
        "--manifest",
        str(manifest),
        "--top-k",
        str(top_k),
        "--json",
    ]
    if image is not None:
        command += ["--image", str(image)]
    if query:
        command += ["--query", query]
    if device:
        command += ["--device", device]
    proc = subprocess.run(command, check=True, text=True, capture_output=True)
    return json.loads(proc.stdout)


def verify_hash(manifest_value: str, relative_path: str, label: str) -> Path:
    path = PROJECT_ROOT / relative_path
    if not path.is_file():
        raise RuntimeError(f"{label} artifact is missing: {path}")
    actual = sha256_file(path)
    if actual != manifest_value:
        raise RuntimeError(f"{label} SHA-256 mismatch: expected={manifest_value} got={actual}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify the complete RICO CLIP + FAISS artifact.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default=None)
    parser.add_argument("--query", default="mobile application screen")
    parser.add_argument("--min-self-score", type=float, default=0.999)
    args = parser.parse_args()

    manifest = read_json(args.manifest)
    num_vectors = int(manifest["index"]["num_vectors"])
    dimension = int(manifest["embedding"]["dimension"])
    architecture = manifest.get("architecture", {})

    if num_vectors != EXPECTED_VECTORS:
        raise RuntimeError(f"expected {EXPECTED_VECTORS} vectors, got {num_vectors}")
    if dimension != EXPECTED_DIMENSION:
        raise RuntimeError(f"expected {EXPECTED_DIMENSION}-d embeddings, got {dimension}")
    if manifest["mapping"]["canonical_rows_indexed"] != EXPECTED_VECTORS:
        raise RuntimeError("canonical mapping count does not match complete RICO corpus")
    if manifest["mapping"]["legacy_alias_records_detected"] != 10:
        raise RuntimeError("expected exactly 10 canonicalized legacy alias records")
    if manifest["dataset"].get("limit") is not None:
        raise RuntimeError("full artifact unexpectedly has a dataset limit")
    if not architecture.get("macos_native_isolation"):
        raise RuntimeError("artifact was not built with macOS native-stack isolation")
    if not architecture.get("resumable_embedding_stage"):
        raise RuntimeError("artifact was not built with resumable corpus embedding support")

    embedding_path = verify_hash(
        manifest["embedding"]["artifact_sha256"], manifest["embedding"]["path"], "embedding"
    )
    index_path = verify_hash(manifest["index"]["sha256"], manifest["index"]["path"], "FAISS")
    mapping_path = verify_hash(
        manifest["mapping"]["artifact_sha256"],
        manifest["mapping"]["artifact_path"],
        "mapping",
    )

    mapping = read_json(mapping_path)
    if len(mapping) != EXPECTED_VECTORS:
        raise RuntimeError(f"artifact mapping must contain {EXPECTED_VECTORS} rows")

    first_filename = mapping["0"]
    first_image = PROJECT_ROOT / "data_storage" / "RICO_dataset" / first_filename
    if not first_image.is_file():
        raise RuntimeError(f"self-retrieval image is missing: {first_image}")

    image_payload = run_search(manifest=args.manifest, image=first_image, device=args.device)
    top1 = image_payload["results"][0]
    if top1["filename"] != first_filename:
        raise RuntimeError(
            f"self-retrieval failed: expected {first_filename}, got {top1['filename']}"
        )
    if float(top1["score"]) < args.min_self_score:
        raise RuntimeError(
            f"self-retrieval score too low: {top1['score']:.6f} < {args.min_self_score:.6f}"
        )

    text_payload = run_search(manifest=args.manifest, query=args.query, device=args.device)

    print("Full RICO FAISS artifact verification PASSED")
    print(f"manifest={args.manifest}")
    print(f"vectors={num_vectors}")
    print(f"dimension={dimension}")
    print(f"index_type={manifest['index']['type']}")
    print(f"native_process_isolation={architecture['macos_native_isolation']}")
    print(f"resumable_embedding_stage={architecture['resumable_embedding_stage']}")
    print(f"embedding_size_mib={embedding_path.stat().st_size / (1024 ** 2):.2f}")
    print(f"faiss_index_size_mib={index_path.stat().st_size / (1024 ** 2):.2f}")
    print(f"query_encoder_device={image_payload['encoder']['device']}")
    print()
    print("Image self-retrieval:")
    for item in image_payload["results"]:
        print(
            f"  {item['rank']:>2}. score={item['score']:.6f} "
            f"faiss_id={item['faiss_id']} file={item['filename']}"
        )
    print()
    print(f'Text retrieval for "{args.query}":')
    for item in text_payload["results"]:
        print(
            f"  {item['rank']:>2}. score={item['score']:.6f} "
            f"faiss_id={item['faiss_id']} file={item['filename']}"
        )


if __name__ == "__main__":
    main()
