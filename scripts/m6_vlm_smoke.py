from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from ui_rag.rag.parse import parse_grounded_answer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--model", default="mlx-community/Qwen2.5-VL-3B-Instruct-4bit")
    parser.add_argument("--max-image-pixels", type=int, default=1280 * 28 * 28)
    args = parser.parse_args()

    image = Path(args.image)
    if not image.exists():
        raise FileNotFoundError(image)
    prompt = """You are given one mobile UI screenshot, evidence E1.
Describe only what is visibly supported. Return ONLY JSON:
{"answer":"short description","selected_evidence_id":"E1","evidence_ids":["E1"],"observations":[{"evidence_id":"E1","claim":"visible fact"}],"confidence":0.0,"abstain":false}
"""
    with tempfile.TemporaryDirectory(prefix="ui-rag-m6-smoke-") as tmp:
        prompt_path = Path(tmp) / "prompt.txt"
        prompt_path.write_text(prompt, encoding="utf-8")
        command = [
            sys.executable, "-m", "ui_rag", "vlm-generate",
            "--prompt-file", str(prompt_path),
            "--image", str(image),
            "--model", args.model,
            "--max-tokens", "180",
            "--temperature", "0.0",
            "--max-image-pixels", str(args.max_image_pixels),
            "--json",
        ]
        try:
            proc = subprocess.run(
                command,
                check=True,
                text=True,
                capture_output=True,
            )
        except subprocess.CalledProcessError as exc:
            print("M6 local VLM smoke FAILED", file=sys.stderr)
            print(f"returncode={exc.returncode}", file=sys.stderr)
            if exc.stderr:
                print("--- child stderr ---", file=sys.stderr)
                print(exc.stderr.rstrip(), file=sys.stderr)
            if exc.stdout:
                print("--- child stdout ---", file=sys.stderr)
                print(exc.stdout.rstrip(), file=sys.stderr)
            raise

    wrapper = json.loads(proc.stdout)
    answer = parse_grounded_answer(wrapper["text"], allowed_evidence_ids={"E1"})
    prep = wrapper.get("image_preprocessing", {})
    items = prep.get("items", [])
    print("M6 local VLM smoke PASSED")
    print(f"model={wrapper['model']}")
    print(f"max_image_pixels={prep.get('max_pixels')}")
    if items:
        item = items[0]
        print(
            "image_preprocessing="
            f"{item['original_width']}x{item['original_height']}->"
            f"{item['prepared_width']}x{item['prepared_height']} resized={item['resized']}"
        )
    print(f"selected={answer.selected_evidence_id}")
    print(f"confidence={answer.confidence:.3f}")
    print(f"abstain={answer.abstain}")
    print(f"answer={answer.answer}")


if __name__ == "__main__":
    main()
