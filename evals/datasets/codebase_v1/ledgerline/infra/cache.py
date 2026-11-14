"""Process-local TTL cache, the backend protocol shared with Redis, and ``@memoize``."""
from __future__ import annotations

import functools
import hashlib
import threading
import time
from collections import OrderedDict
from decimal import Decimal
from typing import Any, Callable, Protocol, TypeVar

T = TypeVar("T")


class CacheBackend(Protocol):
    def get(self, key: str) -> Any | None: ...
    def set(self, key: str, value: Any, ttl_seconds: float) -> None: ...
    def incr(self, key: str, amount: int = 1, ttl_seconds: float | None = None) -> int: ...


class TTLCache:
    """Bounded LRU cache with per-entry expiry; safe to share between threads."""

    def __init__(self, max_entries: int = 10_000) -> None:
        self._data: OrderedDict[str, tuple[Any, float]] = OrderedDict()
        self._max_entries = max_entries
        self._lock = threading.Lock()

    def get(self, key: str) -> Any | None:
        with self._lock:
            entry = self._data.get(key)
            if entry is None:
                return None
            if entry[1] <= time.monotonic():
                del self._data[key]
                return None
            self._data.move_to_end(key)
            return entry[0]

    def set(self, key: str, value: Any, ttl_seconds: float) -> None:
        with self._lock:
            self._data[key] = (value, time.monotonic() + ttl_seconds)
            self._data.move_to_end(key)
            while len(self._data) > self._max_entries:
                self._data.popitem(last=False)

    def incr(self, key: str, amount: int = 1, ttl_seconds: float | None = None) -> int:
        """Atomically increment a counter; ``ttl_seconds`` applies only when the key is created."""
        with self._lock:
            now = time.monotonic()
            value, expires = self._data.get(key, (0, 0.0))
            if expires <= now:
                value, expires = 0, now + ttl_seconds if ttl_seconds is not None else float("inf")
            self._data[key] = (value + amount, expires)
            return value + amount


def _stable_repr(value: Any) -> str:
    if isinstance(value, (str, int, bool, Decimal)) or value is None:
        return repr(value)
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(_stable_repr(v) for v in value) + "]"
    raise TypeError(f"unhashable cache argument of type {type(value).__name__}")


def make_key(namespace: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> str:
    """Deterministic cache key for a call, independent of keyword-argument order."""
    parts = [namespace]
    parts.extend(_stable_repr(arg) for arg in args)
    parts.extend(sorted(kwargs))
    digest = hashlib.blake2b("|".join(parts).encode(), digest_size=16).hexdigest()
    return f"{namespace}:{digest}"


_default_backend: CacheBackend = TTLCache()


def memoize(
    namespace: str, ttl_seconds: float = 60, *, skip_first_arg: bool = False
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Cache a function's non-None results. ``skip_first_arg`` keeps ``self``/clients out of the key."""

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            key = make_key(namespace, args[1:] if skip_first_arg else args, kwargs)
            cached = _default_backend.get(key)
            if cached is not None:
                return cached
            result = func(*args, **kwargs)
            if result is not None:
                _default_backend.set(key, result, ttl_seconds)
            return result

        return wrapper

    return decorator
