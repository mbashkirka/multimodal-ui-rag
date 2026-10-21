from __future__ import annotations

import argparse
import json
from pathlib import Path

from ui_rag.api.workers import PersistentJsonWorker
from ui_rag.indexing.manifest import read_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        default="artifacts/metadata/siglip2-base-p16-naflex-p256-manifest.json",
    )
    args = parser.parse_args()
    manifest_path = Path(args.manifest).resolve()
    manifest = read_json(manifest_path)
    dimension = int(manifest["embedding"]["dimension"])

    worker = PersistentJsonWorker(
        kind="faiss",
        args=["--manifest", str(manifest_path)],
        request_timeout_seconds=10.0,
    )
    try:
        worker.start()
        before = worker.diagnostics()
        first = worker.request({"op": "search", "vector": [0.0] * dimension, "top_k": 1})
        worker.terminate_for_diagnostic()
        second = worker.request({"op": "search", "vector": [0.0] * dimension, "top_k": 1})
        after = worker.diagnostics()
        if not after["alive"] or int(after["restarts"]) < 1:
            raise SystemExit(f"worker did not recover: {after}")
        if not first["results"] or not second["results"]:
            raise SystemExit("FAISS recovery smoke returned no rows")
        print(json.dumps({
            "milestone": "M9",
            "stage": "worker_recovery_smoke",
            "worker": "faiss",
            "pid_before": before["pid"],
            "pid_after": after["pid"],
            "restart_count": after["restarts"],
            "timeout_count": after["timeouts"],
            "alive": after["alive"],
        }, indent=2))
        print("M9 worker recovery smoke PASSED")
    finally:
        worker.close(force=True)


if __name__ == "__main__":
    main()
