"""Tests for the cache layer and the cached answer service.

These run without Redis, the heavy models, or any external service: the cache
falls back to its in-process backend and the pipeline is monkeypatched.
"""

from dataclasses import dataclass

import caching.cache as cache
from caching import cache_get_json, cache_set_json, make_key, ping


def test_make_key_is_stable_and_scoped():
    k1 = make_key("answer", "hello world", None, 8)
    k2 = make_key("answer", "hello world", None, 8)
    k3 = make_key("answer", "hello world", None, 5)
    assert k1 == k2
    assert k1 != k3
    assert k1.startswith("lexscope:answer:")


def test_memory_roundtrip():
    key = make_key("test", "roundtrip")
    assert cache_get_json(key) is None
    cache_set_json(key, {"value": 42})
    assert cache_get_json(key) == {"value": 42}


def test_ping_reports_a_valid_backend():
    # "redis" when REDIS_URL points at a reachable server, otherwise the
    # in-process "memory" fallback (or "disabled" if caching is turned off).
    assert ping() in ("redis", "memory", "disabled")


@dataclass
class _Chunk:
    chunk_id: str
    text: str = "text"


@dataclass
class _Result:
    answer: str = "cached answer"
    citations: list = None
    confidence: str = "high"
    insufficient_context: bool = False

    def __post_init__(self):
        if self.citations is None:
            self.citations = []


def test_answer_query_uses_cache(monkeypatch):
    import app.service as service

    calls = {"pipeline": 0, "generate": 0}

    def fake_pipeline(query, doc_type_filter=None, rerank_top_k=8):
        calls["pipeline"] += 1
        return [_Chunk("c1"), _Chunk("c2")]

    def fake_generate(query, chunks):
        calls["generate"] += 1
        return _Result()

    monkeypatch.setattr(service, "run_retrieval_pipeline", fake_pipeline)
    monkeypatch.setattr(service, "generate_answer", fake_generate)

    q = "a unique test question for caching"

    first = service.answer_query(q, top_k=8)
    assert first["cached"] is False
    assert first["answer"] == "cached answer"
    assert first["retrieved_chunk_ids"] == ["c1", "c2"]
    assert calls == {"pipeline": 1, "generate": 1}

    # Second identical call must be served from cache — no extra pipeline/LLM work.
    second = service.answer_query(q, top_k=8)
    assert second["cached"] is True
    assert second["answer"] == "cached answer"
    assert calls == {"pipeline": 1, "generate": 1}

    # use_cache=False forces recomputation.
    third = service.answer_query(q, top_k=8, use_cache=False)
    assert third["cached"] is False
    assert calls == {"pipeline": 2, "generate": 2}
