from __future__ import annotations

import time
import functools
from contextvars import ContextVar
from typing import Any

_cache: dict[tuple, tuple[Any, float]] = {}
_bypass: ContextVar[bool] = ContextVar("bypass_cache", default=False)


def cached(ttl_seconds: int):
    """In-memory TTL cache decorator. Key = (fn_name, args, sorted_kwargs).
    When _bypass is True the cache is skipped entirely — no read, no write."""
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            if _bypass.get():
                return fn(*args, **kwargs)
            key = (fn.__name__, args, tuple(sorted(kwargs.items())))
            if key in _cache:
                value, timestamp = _cache[key]
                if time.time() - timestamp < ttl_seconds:
                    return value
            try:
                result = fn(*args, **kwargs)
            except Exception:
                raise
            _cache[key] = (result, time.time())
            return result
        return wrapper
    return decorator


def clear_cache() -> None:
    _cache.clear()
