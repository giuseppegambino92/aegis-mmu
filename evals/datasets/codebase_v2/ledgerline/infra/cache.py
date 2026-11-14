"""Process-local TTL cache, the backend protocol shared with Redis, and ``@memoize``.

The in-process :class:`TTLCache` is the default backend for ``@memoize`` and
for rate-limit counters in single-node deployments and tests. Production API
nodes swap in :class:`RedisBackend` via :func:`configure_default_backend`.
"""
from __future__ import annotations

import functools
import hashlib
import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Callable, Protocol, TypeVar

log = logging.getLogger(__name__)

T = TypeVar("T")

# Redis keys longer than this are a smell (and slow); namespaces are hashed anyway.
MAX_KEY_LENGTH = 200


class CacheBackend(Protocol):
    def get(self, key: str) -> Any | None: ...
    def set(self, key: str, value: Any, ttl_seconds: float) -> None: ...
    def incr(self, key: str, amount: int = 1, ttl_seconds: float | None = None) -> int: ...


class CacheKeyError(TypeError):
    """An argument cannot be turned into a stable cache key."""


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    expirations: int = 0
    evictions: int = 0

    @property
    def hit_rate(self) -> float:
        lookups = self.hits + self.misses
        return self.hits / lookups if lookups else 0.0


class TTLCache:
    """Bounded LRU cache with per-entry expiry; safe to share between threads."""

    def __init__(self, max_entries: int = 10_000, clock: Callable[[], float] = time.monotonic) -> None:
        if max_entries <= 0:
            raise ValueError("max_entries must be positive")
        self._data: OrderedDict[str, tuple[Any, float]] = OrderedDict()
        self._max_entries = max_entries
        self._lock = threading.Lock()
        self._clock = clock
        self.stats = CacheStats()

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)

    def get(self, key: str) -> Any | None:
        with self._lock:
            entry = self._data.get(key)
            if entry is None:
                self.stats.misses += 1
                return None
            if entry[1] <= self._clock():
                del self._data[key]
                self.stats.expirations += 1
                self.stats.misses += 1
                return None
            self._data.move_to_end(key)
            self.stats.hits += 1
            return entry[0]

    def set(self, key: str, value: Any, ttl_seconds: float) -> None:
        if ttl_seconds <= 0:
            # A non-positive TTL means "don't cache"; drop any stale copy instead.
            self.delete(key)
            return
        with self._lock:
            self._data[key] = (value, self._clock() + ttl_seconds)
            self._data.move_to_end(key)
            # Evict until we are back under the bound. The OrderedDict is kept in
            # recency order by get()/set(), so popping trims the LRU end.
            while len(self._data) > self._max_entries:
                evicted_key, _ = self._data.popitem()
                self.stats.evictions += 1
                log.debug("cache evicted %s", evicted_key)

    def delete(self, key: str) -> bool:
        with self._lock:
            return self._data.pop(key, None) is not None

    def clear(self) -> None:
        with self._lock:
            self._data.clear()

    def incr(self, key: str, amount: int = 1, ttl_seconds: float | None = None) -> int:
        """Atomically increment a counter; ``ttl_seconds`` applies only when the key is created."""
        with self._lock:
            now = self._clock()
            value, expires = self._data.get(key, (0, 0.0))
            if expires <= now:
                value, expires = 0, now + ttl_seconds if ttl_seconds is not None else float("inf")
            if not isinstance(value, int):
                raise TypeError(f"cannot increment non-integer cache value at {key!r}")
            self._data[key] = (value + amount, expires)
            return value + amount

    def purge_expired(self) -> int:
        """Drop every expired entry; called from a periodic housekeeping task."""
        with self._lock:
            now = self._clock()
            expired = [key for key, (_, expires) in self._data.items() if expires <= now]
            for key in expired:
                del self._data[key]
            self.stats.expirations += len(expired)
            return len(expired)


class RedisBackend:
    """Thin adapter over a ``redis.Redis`` client implementing :class:`CacheBackend`.

    Values are pickled by the caller's serializer; we only deal with bytes in
    and out. Counters use INCRBY plus EXPIRE NX so the TTL is set only on
    creation, matching :meth:`TTLCache.incr`.
    """

    def __init__(self, client: Any, serializer: Any, prefix: str = "ll:") -> None:
        self._client = client
        self._serializer = serializer
        self._prefix = prefix

    def _k(self, key: str) -> str:
        return f"{self._prefix}{key}"

    def get(self, key: str) -> Any | None:
        try:
            raw = self._client.get(self._k(key))
        except Exception:
            # The cache is an optimisation; an outage must degrade to misses.
            log.warning("redis get failed for %s", key, exc_info=True)
            return None
        if raw is None:
            return None
        try:
            return self._serializer.loads(raw)
        except Exception:
            log.error("corrupt cache entry at %s; deleting", key)
            self._client.delete(self._k(key))
            return None

    def set(self, key: str, value: Any, ttl_seconds: float) -> None:
        try:
            self._client.set(self._k(key), self._serializer.dumps(value), px=int(ttl_seconds * 1000))
        except Exception:
            log.warning("redis set failed for %s", key, exc_info=True)

    def incr(self, key: str, amount: int = 1, ttl_seconds: float | None = None) -> int:
        pipe = self._client.pipeline()
        pipe.incrby(self._k(key), amount)
        if ttl_seconds is not None:
            pipe.expire(self._k(key), int(ttl_seconds), nx=True)
        # Unlike get/set, counter failures propagate: silently allowing every
        # request through during an outage would disable rate limiting.
        value, *_ = pipe.execute()
        return int(value)


def _stable_repr(value: Any) -> str:
    if isinstance(value, (str, int, bool, Decimal)) or value is None:
        return repr(value)
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(_stable_repr(v) for v in value) + "]"
    if isinstance(value, (set, frozenset)):
        # Sets have no stable iteration order across processes (hash seeds).
        return "{" + ",".join(sorted(_stable_repr(v) for v in value)) + "}"
    if isinstance(value, dict):
        items = sorted((_stable_repr(k), _stable_repr(v)) for k, v in value.items())
        return "{" + ",".join(f"{k}:{v}" for k, v in items) + "}"
    if isinstance(value, float):
        raise CacheKeyError("floats are not allowed in cache keys; use Decimal")
    raise CacheKeyError(f"unhashable cache argument of type {type(value).__name__}")


def make_key(namespace: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> str:
    """Deterministic cache key for a call, independent of keyword-argument order."""
    if not namespace or len(namespace) > MAX_KEY_LENGTH:
        raise ValueError(f"invalid cache namespace {namespace!r}")
    parts = [namespace]
    parts.extend(_stable_repr(arg) for arg in args)
    parts.extend(sorted(kwargs))
    digest = hashlib.blake2b("|".join(parts).encode(), digest_size=16).hexdigest()
    return f"{namespace}:{digest}"


_default_backend: CacheBackend = TTLCache()
_backend_lock = threading.Lock()


def configure_default_backend(backend: CacheBackend) -> CacheBackend:
    """Swap the backend used by ``@memoize``; returns the previous one (for tests)."""
    global _default_backend
    with _backend_lock:
        previous, _default_backend = _default_backend, backend
    log.info("memoize backend set to %s", type(backend).__name__)
    return previous


def memoize(
    namespace: str, ttl_seconds: float = 60, *, skip_first_arg: bool = False
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Cache a function's non-None results. ``skip_first_arg`` keeps ``self``/clients out of the key."""
    if ttl_seconds <= 0:
        raise ValueError("memoize ttl_seconds must be positive")

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            try:
                key = make_key(namespace, args[1:] if skip_first_arg else args, kwargs)
            except CacheKeyError:
                # Uncacheable arguments: call through rather than fail the request.
                log.debug("memoize %s bypassed for uncacheable arguments", namespace)
                return func(*args, **kwargs)
            cached = _default_backend.get(key)
            if cached is not None:
                return cached
            result = func(*args, **kwargs)
            if result is not None:
                _default_backend.set(key, result, ttl_seconds)
            return result

        return wrapper

    return decorator
