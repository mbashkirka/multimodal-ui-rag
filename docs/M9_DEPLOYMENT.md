# M9 deployment and containerization policy

## Validated production target

The validated data-plane target remains **native macOS on Apple Silicon**. The frozen
runtime uses separate persistent processes for PyTorch/SigLIP2, FAISS, and MLX/Qwen.
This preserves the native-library isolation established in M1 while giving M8/M9
steady-state serving performance.

## Docker scope

`Dockerfile.ci` is deliberately a **quality/contract container**, not a claim that the
frozen Metal data plane is production-equivalent inside Docker Desktop. It runs lint,
unit/contract tests, and release checks without RICO artifacts or the VLM weights.

MLX itself now has non-macOS backends, but this project has not validated the frozen
`mlx-vlm==0.6.8` + Qwen checkpoint + artifact stack on Linux CUDA/CPU. A future Linux
serving port must therefore be benchmarked as a separate deployment profile rather
than silently treated as equivalent to the proven Apple-Silicon profile.

## CI split

- Ubuntu job: lint, unit/contract tests, OpenAPI/release checks, compileall.
- macOS ARM job: native-import isolation and operational worker contracts.
- Full RICO + Qwen end-to-end inference remains a local/release acceptance test because
  the corpus artifacts and model cache are intentionally not stored in Git.
