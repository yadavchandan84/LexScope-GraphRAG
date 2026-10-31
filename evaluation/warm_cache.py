"""Pre-populate the answer cache from the golden set.

Run once after (re)indexing so every golden-set question is served instantly
from Redis (or the in-process fallback) instead of running the full pipeline:

    python -m evaluation.warm_cache

Add ``--force`` to recompute even queries that are already cached.
"""

from __future__ import annotations

import argparse
import json
import time

from app.service import answer_query
from caching import ping as cache_ping

GOLDEN_SET_PATH = "evaluation/golden_set.json"


def load_queries(path: str) -> list[str]:
    with open(path, encoding="utf-8") as f:
        items = json.load(f)
    # Preserve order but drop duplicate queries (the golden set has a few).
    seen: set[str] = set()
    queries: list[str] = []
    for item in items:
        q = item.get("query", "").strip()
        if q and q not in seen:
            seen.add(q)
            queries.append(q)
    return queries


def main() -> None:
    parser = argparse.ArgumentParser(description="Warm the answer cache from the golden set.")
    parser.add_argument("--force", action="store_true", help="Recompute even cached answers.")
    parser.add_argument("--path", default=GOLDEN_SET_PATH, help="Path to the golden set JSON.")
    args = parser.parse_args()

    queries = load_queries(args.path)
    print(f"Cache backend: {cache_ping()}")
    print(f"Warming {len(queries)} unique golden-set queries...\n")

    for i, query in enumerate(queries, start=1):
        start = time.perf_counter()
        result = answer_query(query, use_cache=not args.force)
        elapsed = time.perf_counter() - start
        status = "cached" if result.get("cached") else "computed"
        print(f"[{i}/{len(queries)}] {status} in {elapsed:.2f}s  |  {query[:70]}")

    print("\nDone. Golden-set queries will now be served from cache.")


if __name__ == "__main__":
    main()
