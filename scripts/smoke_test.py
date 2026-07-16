"""Simple HTTP smoke test for a running API service."""

from __future__ import annotations

import argparse
import json
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import urlopen


def fetch_json(url: str) -> dict[str, Any]:
    """Fetch a JSON response from a URL."""
    try:
        with urlopen(url, timeout=5) as response:
            payload = response.read().decode("utf-8")
    except (HTTPError, URLError, TimeoutError) as exc:
        raise SystemExit(f"Smoke test request failed: {url}: {exc}") from exc

    return json.loads(payload)


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test a running Multilingual RAG API.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    args = parser.parse_args()

    health = fetch_json(f"{args.base_url.rstrip('/')}/healthz")
    if health.get("status") != "ok":
        raise SystemExit(f"Unexpected health response: {health}")

    readiness = fetch_json(f"{args.base_url.rstrip('/')}/readyz")
    if readiness.get("status") != "ok":
        raise SystemExit(f"Unexpected readiness response: {readiness}")

    print("Smoke test passed")


if __name__ == "__main__":
    main()

