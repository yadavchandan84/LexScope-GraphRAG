"""Small cache abstraction used across the RAG pipeline.

Backends, in order of preference:
  1. Redis, when ``REDIS_URL`` is configured and reachable. Shared across
     processes and survives restarts — this is what makes repeat / golden-set
     queries return instantly.
  2. An in-process dict fallback, so the app still runs (with per-process
     caching only) when Redis is not available.

The module never raises on cache errors: caching is a best-effort speed-up,
never a correctness dependency. Any backend failure degrades to "cache miss".
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from typing import Any

from config import settings

_LOCK = threading.Lock()
_MEMORY: dict[str, tuple[float | None, str]] = {}
_redis_client = None
_redis_checked = False


def _get_redis():
    """Return a connected redis client, or None to use the memory fallback."""
    global _redis_client, _redis_checked
    if _redis_checked:
        return _redis_client
    _redis_checked = True

    if not settings.cache_enabled or not settings.redis_url:
        _redis_client = None
        return None

    try:
        import redis  # imported lazily so redis stays an optional dependency

        client = redis.Redis.from_url(
            settings.redis_url,
            socket_connect_timeout=2,
            socket_timeout=2,
            decode_responses=True,
        )
        client.ping()
        _redis_client = client
    except Exception:
        # Unreachable / not installed — fall back to in-process cache silently.
        _redis_client = None
    return _redis_client


def ping() -> str:
    """Report which backend is active. Handy for /health and warm-up scripts."""
    if not settings.cache_enabled:
        return "disabled"
    return "redis" if _get_redis() is not None else "memory"


def make_key(prefix: str, *parts: Any) -> str:
    """Build a stable, collision-resistant cache key from arbitrary parts."""
    raw = "\x1f".join(json.dumps(p, sort_keys=True, default=str) for p in parts)
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return f"lexscope:{prefix}:{digest}"


def _memory_get(key: str) -> str | None:
    with _LOCK:
        entry = _MEMORY.get(key)
        if entry is None:
            return None
        expires_at, value = entry
        if expires_at is not None and expires_at <= time.time():
            _MEMORY.pop(key, None)
            return None
        return value


def _memory_set(key: str, value: str, ttl: int | None) -> None:
    expires_at = time.time() + ttl if ttl else None
    with _LOCK:
        _MEMORY[key] = (expires_at, value)


def cache_get_json(key: str) -> Any | None:
    """Return the decoded JSON value for ``key``, or None on miss/error."""
    if not settings.cache_enabled:
        return None
    client = _get_redis()
    try:
        raw = client.get(key) if client is not None else _memory_get(key)
    except Exception:
        return None
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except (ValueError, TypeError):
        return None


def cache_set_json(key: str, value: Any, ttl: int | None = None) -> None:
    """Store ``value`` (JSON-serializable) under ``key``. Best effort."""
    if not settings.cache_enabled:
        return
    try:
        raw = json.dumps(value, default=str)
    except (TypeError, ValueError):
        return
    client = _get_redis()
    try:
        if client is not None:
            if ttl:
                client.set(key, raw, ex=ttl)
            else:
                client.set(key, raw)
        else:
            _memory_set(key, raw, ttl)
    except Exception:
        # Never let a cache write break a request.
        return
