from __future__ import annotations

import importlib.metadata
import platform
import sys


def version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def main() -> None:
    print("M6 MLX diagnostics")
    print(f"python={sys.version.split()[0]}")
    print(f"platform={platform.platform()}")
    print(f"machine={platform.machine()}")
    for package in ("mlx", "mlx-vlm", "mlx-lm", "transformers"):
        print(f"{package}={version(package)}")

    print("stage=mlx_import start", flush=True)
    import mlx.core as mx
    print("stage=mlx_import done", flush=True)

    print("stage=mlx_gpu_matmul start", flush=True)
    a = mx.ones((128, 128), dtype=mx.float32)
    b = a @ a
    mx.eval(b)
    value = float(b[0, 0].item())
    print(f"stage=mlx_gpu_matmul done value={value:.1f}", flush=True)
    print("M6 MLX diagnostics PASSED")


if __name__ == "__main__":
    main()
