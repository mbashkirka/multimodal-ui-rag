# M6.1 changelog

## 0.6.1 — MLX VLM stability + bounded visual tokens

- Fixed the MLX-VLM provider to follow the documented `load_config(model_path)` contract instead of passing `model.config` into `apply_chat_template`.
- Added deterministic Qwen2.5-VL evidence preprocessing with a default cap of **1,003,520 pixels per image** (1280 merged visual tokens), matching the typical-use range recommended by Qwen.
- Oversized screenshots are resized with aspect ratio preserved and dimensions aligned to 28-pixel merged vision patches; small images are never upscaled.
- Added flushed native stage markers around MLX-VLM import, config load, model load, image preparation, prompt templating and generation so SIGSEGV failures expose the last successful stage.
- Updated the VLM smoke wrapper to surface child stdout/stderr on native crashes instead of hiding the useful diagnostic context inside `CalledProcessError`.
- Added `make m6-diagnose` for a minimal MLX import/GPU-matmul sanity check.
- Pinned the stable MLX runtime used by the M6 setup to `mlx==0.32.0` alongside `mlx-vlm==0.6.8`.
- Preserved the existing Torch / FAISS / MLX native-process isolation contract.
