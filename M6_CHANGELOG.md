# M6 changelog

## 0.6.0 — Grounded multimodal RAG

- Added frozen M5 production retrieval as a reusable online retrieval stage.
- Added Top-4 evidence packaging with stable evidence IDs.
- Added strict grounded JSON answer schema and citation validator.
- Added local Apple-Silicon VLM provider using `mlx-vlm`.
- Default VLM: `mlx-community/Qwen2.5-VL-3B-Instruct-4bit`.
- Added `ui-rag rag` interactive grounded RAG command.
- Added low-level isolated `ui-rag vlm-generate` command.
- Added local VLM smoke test and full end-to-end RAG smoke target.
- Added balanced Screen2Words evidence-selection diagnostic and evaluator.
- Preserved native-process isolation between Torch, FAISS and MLX.
