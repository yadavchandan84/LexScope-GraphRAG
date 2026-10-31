"""Query service: retrieval + generation with a cache in front.

The API endpoint and the eval harness both go through :func:`answer_query`
so caching behaves identically everywhere. An identical question (same query,
doc-type filter and top_k) is served from the cache and skips the entire
retrieval + LLM pipeline — this is what makes the golden set answer instantly.
"""

from __future__ import annotations

from caching import cache_get_json, cache_set_json, make_key
from config import settings
from generation.generate import generate_answer
from retrieval.cli import run_retrieval_pipeline


def _normalize_query(query: str) -> str:
    # Collapse whitespace and lowercase so trivially different phrasings of the
    # same question share a cache entry.
    return " ".join(query.split()).lower()


def answer_query(
    query: str,
    doc_type_filter: str | None = None,
    top_k: int = 8,
    use_cache: bool = True,
) -> dict:
    """Return a response dict for ``query``. Cached by (query, filter, top_k)."""
    cache_key = make_key("answer", _normalize_query(query), doc_type_filter, top_k)

    if use_cache:
        cached = cache_get_json(cache_key)
        if cached is not None:
            cached["cached"] = True
            return cached

    chunks = run_retrieval_pipeline(
        query, doc_type_filter=doc_type_filter, rerank_top_k=top_k
    )
    result = generate_answer(query, chunks)

    payload = {
        "answer": result.answer,
        "citations": result.citations,
        "confidence": result.confidence,
        "insufficient_context": result.insufficient_context,
        "retrieved_chunk_ids": [c.chunk_id for c in chunks],
        "cached": False,
    }

    if use_cache:
        cache_set_json(cache_key, payload, ttl=settings.cache_answer_ttl)

    return payload
