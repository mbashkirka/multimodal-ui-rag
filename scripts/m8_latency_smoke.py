from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request


def post_json(url: str, payload: dict, request_id: str) -> dict:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "X-Request-ID": request_id,
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise RuntimeError(exc.read().decode("utf-8")) from exc


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure M8.1 warm-worker API latency")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--query", default="dark banking login screen")
    args = parser.parse_args()

    runs = []
    for i in range(1, 3):
        search = post_json(
            f"{args.base_url}/v1/search",
            {"query": args.query, "top_k": 4},
            f"m81-search-{i}",
        )
        runs.append({"endpoint": "search", "run": i, "latency": search["latency"]})
    for i in range(1, 3):
        rag = post_json(
            f"{args.base_url}/v1/rag",
            {"query": args.query, "max_tokens": 128},
            f"m81-rag-{i}",
        )
        runs.append({"endpoint": "rag", "run": i, "latency": rag["latency"]})

    print(json.dumps({"milestone": "M8.1", "serving_mode": "persistent-isolated-workers", "runs": runs}, indent=2))


if __name__ == "__main__":
    main()
