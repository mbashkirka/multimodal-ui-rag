from __future__ import annotations

from ui_rag.embeddings.base import RetrievalEncoder


def infer_backend(model_name: str, backend: str | None = None) -> str:
    if backend:
        return backend
    if model_name.startswith("google/siglip2-"):
        return "transformers_siglip2"
    return "open_clip"


def create_encoder(
    *,
    model_name: str,
    pretrained: str,
    device: str | None = None,
    backend: str | None = None,
    max_num_patches: int | None = None,
) -> RetrievalEncoder:
    resolved = infer_backend(model_name, backend)
    if resolved == "open_clip":
        from ui_rag.embeddings.clip import CLIPEncoder

        return CLIPEncoder(
            model_name=model_name,
            pretrained=pretrained,
            device=device,
        )
    if resolved == "transformers_siglip2":
        from ui_rag.embeddings.siglip2 import SigLIP2Encoder

        return SigLIP2Encoder(
            model_name=model_name,
            pretrained=pretrained,
            device=device,
            max_num_patches=max_num_patches,
        )
    raise ValueError(f"unsupported encoder backend: {resolved}")


def encoder_kwargs_from_manifest(embedding_cfg: dict) -> dict:
    return {
        "model_name": embedding_cfg["model_name"],
        "pretrained": embedding_cfg.get("pretrained", "huggingface"),
        "backend": embedding_cfg.get("backend"),
        "max_num_patches": embedding_cfg.get("max_num_patches"),
    }
