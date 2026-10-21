# M6.1 Diagnostic Results

Reference run: Apple Silicon / MLX 0.32.0 / mlx-vlm 0.6.8 / `mlx-community/Qwen2.5-VL-3B-Instruct-4bit`.

The end-to-end smoke succeeded for query `dark banking login screen`: M5 retrieval produced E1–E4, Qwen2.5-VL selected E1, the response was valid JSON, and the cited screenshot visibly matched a mobile-banking login screen.

The subsequent balanced 40-case Screen2Words **test diagnostic** exposed a failure of the original one-pass grounding contract:

| Metric | M6.1 diagnostic |
|---|---:|
| Cases | 40 (20 retrieval-hit / 20 miss) |
| Structured JSON + citation valid | 12.5% |
| Exact evidence selection on retrieval-hit cases | 10.0% |
| End-to-end exact selection | 5.0% |
| Abstention on misses (diagnostic) | 0.0% |
| Mean VLM generation time | 43.67 s/case |
| Total generation time | 1746.61 s |

These numbers are **not** accepted as the final M6 result. The original prompt overloaded a 3B VLM with four high-resolution images, implicit image-order → evidence-ID correspondence, semantic selection, explanation generation, confidence estimation, citation construction, and strict JSON serialization in one decode.

M6.2 therefore separates selection from answer generation and makes citation construction deterministic in the application layer. Since this M6.1 diagnostic already observed Screen2Words test behavior, M6.2 prompt/protocol development is performed on the official dev split before any new reporting holdout is used.
