from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from ui_rag.indexing.manifest import read_json

DEFAULT_MANIFEST = Path(
    "artifacts/metadata/clip-vit-b32-quickgelu-openai-smoke100-manifest.json"
)
DEFAULT_IMAGE = Path("data_storage/RICO_dataset/71018.jpg")


def run_search(
    *,
    manifest: Path,
    image: Path | None = None,
    query: str | None = None,
    device: str | None = None,
) -> dict:
    command = [
        sys.executable,
        "-m",
        "ui_rag",
        "search",
        "--manifest",
        str(manifest),
        "--top-k",
        "5",
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify the 100-screen CLIP + FAISS smoke artifact end-to-end."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--expected-filename", default="71018.jpg")
    parser.add_argument("--min-self-score", type=float, default=0.999)
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default=None)
    args = parser.parse_args()

    manifest = read_json(args.manifest)
    actual_vectors = int(manifest["index"]["num_vectors"])
    dimension = int(manifest["embedding"]["dimension"])
    if actual_vectors != 100:
        raise RuntimeError(f"smoke artifact must contain 100 vectors, got {actual_vectors}")
    if dimension != 512:
        raise RuntimeError(f"expected 512-d CLIP embeddings, got {dimension}")
    if not manifest.get("architecture", {}).get("macos_native_isolation"):
        raise RuntimeError("smoke artifact was not built with macOS native-stack isolation")

    image_payload = run_search(manifest=args.manifest, image=args.image, device=args.device)
    image_results = image_payload["results"]
    if not image_results:
        raise RuntimeError("image self-retrieval returned no results")
    top1 = image_results[0]
    if top1["filename"] != args.expected_filename:
        raise RuntimeError(
            f"self-retrieval failed: expected {args.expected_filename} at rank 1, "
            f"got {top1['filename']}"
        )
    if float(top1["score"]) < args.min_self_score:
        raise RuntimeError(
            f"self-retrieval score too low: {float(top1['score']):.6f} < {args.min_self_score:.6f}"
        )

    text_payload = run_search(
        manifest=args.manifest,
        query="mobile application screen",
        device=args.device,
    )

    print("Smoke FAISS artifact verification PASSED")
    print(f"manifest={args.manifest}")
    print(f"vectors={actual_vectors}")
    print(f"dimension={dimension}")
    print(f"index_type={manifest['index']['type']}")
    print(f"native_process_isolation={manifest['architecture']['macos_native_isolation']}")
    print(f"query_encoder_device={image_payload['encoder']['device']}")
    print()
    print("Image self-retrieval:")
    for item in image_results:
        print(
            f"  {item['rank']:>2}. score={item['score']:.6f} "
            f"faiss_id={item['faiss_id']} file={item['filename']}"
        )
    print()
    print('Text retrieval for "mobile application screen":')
    for item in text_payload["results"]:
        print(
            f"  {item['rank']:>2}. score={item['score']:.6f} "
            f"faiss_id={item['faiss_id']} file={item['filename']}"
        )


if __name__ == "__main__":
    main()
