from __future__ import annotations

import argparse

import numpy as np

from ui_rag.embeddings.clip import CLIPEncoder


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--text", default="mobile application screen")
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default=None)
    args = parser.parse_args()
    encoder = CLIPEncoder(device=args.device)
    text_embedding = encoder.encode_text(args.text)
    image_embedding = encoder.encode_images(args.image)
    print(f"device={encoder.device}")
    print(f"model={encoder.model_name}")
    print(f"pretrained={encoder.pretrained}")
    print(f"text_shape={text_embedding.shape}")
    print(f"image_shape={image_embedding.shape}")
    print(f"text_norm={np.linalg.norm(text_embedding[0]):.6f}")
    print(f"image_norm={np.linalg.norm(image_embedding[0]):.6f}")
    print(f"cosine_similarity={float(text_embedding[0] @ image_embedding[0]):.6f}")
    assert text_embedding.shape == image_embedding.shape
    assert np.isclose(np.linalg.norm(text_embedding[0]), 1.0, atol=1e-4)
    assert np.isclose(np.linalg.norm(image_embedding[0]), 1.0, atol=1e-4)


if __name__ == "__main__":
    main()
