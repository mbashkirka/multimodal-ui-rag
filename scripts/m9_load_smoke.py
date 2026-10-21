from __future__ import annotations

import argparse
import json
import re
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed


def _post(url: str, body: dict, request_id: str, timeout: float) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", "X-Request-ID": request_id},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
            return {"status": response.status, "request_id": request_id, "latency": payload.get("latency")}
    except urllib.error.HTTPError as exc:
        payload = json.loads(exc.read().decode("utf-8"))
        return {"status": exc.code, "request_id": request_id, "error": payload.get("error")}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--requests", type=int, default=6)
    parser.add_argument("--concurrency", type=int, default=6)
    parser.add_argument("--timeout", type=float, default=90.0)
    args = parser.parse_args()

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [
            pool.submit(
                _post,
                f"{args.base_url}/v1/search",
                {"query": "dark banking login screen", "top_k": 4},
                f"m9-load-{i}",
                args.timeout,
            )
            for i in range(args.requests)
        ]
        rows = [future.result() for future in as_completed(futures)]

    counts = Counter(int(row["status"]) for row in rows)
    if any(code not in {200, 429} for code in counts):
        raise SystemExit(f"unexpected status under load: {dict(counts)}")
    if counts[200] < 1:
        raise SystemExit("no successful request under load")
    if counts[429] < 1:
        raise SystemExit("backpressure gate was not exercised; expected at least one 429")

    with urllib.request.urlopen(f"{args.base_url}/metrics", timeout=10) as response:
        metrics = response.read().decode("utf-8")
    match = re.search(r"^ui_rag_inference_busy_total ([0-9]+)$", metrics, flags=re.MULTILINE)
    busy_total = int(match.group(1)) if match else -1
    if busy_total < 1:
        raise SystemExit("Prometheus metrics did not record inference_busy_total")

    print(json.dumps({
        "milestone": "M9",
        "stage": "concurrency_backpressure_smoke",
        "requests": args.requests,
        "concurrency": args.concurrency,
        "status_counts": dict(sorted(counts.items())),
        "inference_busy_total": busy_total,
        "sample_success_latency": next((r.get("latency") for r in rows if r["status"] == 200), None),
    }, indent=2))
    print("M9 load/backpressure smoke PASSED")


if __name__ == "__main__":
    main()
