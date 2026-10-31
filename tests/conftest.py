"""Test isolation for the cache layer.

Unit tests should be deterministic and must not depend on (or pollute) a real
Redis instance. This fixture forces the in-process memory backend and clears it
before every test, so results don't leak between runs regardless of REDIS_URL.
"""

import pytest

import caching.cache as cache


@pytest.fixture(autouse=True)
def _isolated_memory_cache(monkeypatch):
    # Pretend Redis is unavailable so cache_* use the in-process dict...
    monkeypatch.setattr(cache, "_redis_client", None)
    monkeypatch.setattr(cache, "_redis_checked", True)
    monkeypatch.setattr(cache, "_get_redis", lambda: None)
    # ...and start each test from an empty cache.
    with cache._LOCK:
        cache._MEMORY.clear()
    yield
    with cache._LOCK:
        cache._MEMORY.clear()
