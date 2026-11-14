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
# Above this many evictions in a single set() something is badly mis-sized;
# log loudly so it shows up on the cache dashboard.
EVICTION_STORM_THRESHOLD = 50


class CacheBackend(Protocol):
    def get(self, key: str) -> Any | None:
        """Return the cached value for ``key``, or None on a miss or an expired entry.

        Implementations must never raise for an unavailable store: a cache
        outage degrades to misses so callers fall through to the source of truth.
        """
        ...

    def set(self, key: str, value: Any, ttl_seconds: float) -> None:
        """Store ``value`` for ``ttl_seconds``; a non-positive TTL means "do not cache".

        Fractional TTLs are allowed (the rate limiter uses sub-second TTLs to
        expire a counter immediately). Failures are logged, not raised.
        """
        ...

    def incr(self, key: str, amount: int = 1, ttl_seconds: float | None = None) -> int:
        """Atomically add ``amount`` to an integer counter and return the new value.

        ``ttl_seconds`` applies only when the counter is created. Unlike get/set,
        failures propagate: callers decide whether to fail open or closed.
        """
        ...


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
        """Fraction of lookups served from cache since the stats were last reset.

        Expired entries count as misses (they are, from the caller's point of
        view), so a cache with a TTL that is too short shows a low hit rate
        even when most keys are being requested repeatedly.
        """
        lookups = self.hits + self.misses + self.expirations
        if lookups <= 0:
            # Avoid a ZeroDivisionError on a freshly started node; the metrics
            # exporter treats 0.0 as "no data yet".
            return 0.0
        return self.hits / lookups


class TTLCache:
    """Bounded LRU cache with per-entry expiry; safe to share between threads."""

    def __init__(self, max_entries: int = 10_000, clock: Callable[[], float] = time.monotonic) -> None:
        if not isinstance(max_entries, int) or isinstance(max_entries, bool):
            raise TypeError(f"max_entries must be an int, got {type(max_entries).__name__}")
        if max_entries <= 0:
            raise ValueError("max_entries must be positive")
        if not callable(clock):
            raise TypeError("clock must be a zero-argument callable returning seconds")
        # Entries are (value, absolute expiry on ``clock``). The OrderedDict
        # doubles as the recency list: the most recently used key is last.
        self._data: OrderedDict[str, tuple[Any, float]] = OrderedDict()
        self._max_entries = max_entries
        # A plain Lock (not RLock): no method calls another while holding it,
        # except set() -> delete(), which happens before the lock is taken.
        self._lock = threading.Lock()
        self._clock = clock
        self.stats = CacheStats()
        log.debug("TTLCache created max_entries=%d clock=%s", max_entries, getattr(clock, "__name__", clock))

    def __len__(self) -> int:
        """Number of stored entries, including expired ones not yet purged."""
        with self._lock:
            return len(self._data)

    def get(self, key: str) -> Any | None:
        """Return the live value for ``key`` and mark it most recently used.

        Expired entries are deleted lazily here as well as by
        :meth:`purge_expired`, so a hot key never serves stale data even when
        the housekeeping task is behind.
        """
        if not isinstance(key, str):
            raise TypeError(f"cache keys must be str, got {type(key).__name__}")
        with self._lock:
            entry = self._data.get(key)
            if entry is None:
                self.stats.misses += 1
                return None
            value, expires_at = entry
            if expires_at <= self._clock():
                # Expiry is exclusive of the deadline: at exactly expires_at
                # the entry is already gone.
                del self._data[key]
                self.stats.expirations += 1
                self.stats.misses += 1
                return None
            self._data.move_to_end(key)
            self.stats.hits += 1
            return value

    def set(self, key: str, value: Any, ttl_seconds: float) -> None:
        if not isinstance(key, str):
            raise TypeError(f"cache keys must be str, got {type(key).__name__}")
        if len(key) > MAX_KEY_LENGTH:
            log.warning("cache key longer than %d chars: %s...", MAX_KEY_LENGTH, key[:40])
        if ttl_seconds <= 0:
            # A non-positive TTL means "don't cache"; drop any stale copy instead.
            self.delete(key)
            return
        evicted = 0
        with self._lock:
            self._data[key] = (value, self._clock() + ttl_seconds)
            self._data.move_to_end(key)
            # Evict until we are back under the bound. The OrderedDict is kept in
            # recency order by get()/set(), so popping trims the LRU end.
            while len(self._data) > self._max_entries:
                evicted_key, _ = self._data.popitem()
                self.stats.evictions += 1
                evicted += 1
                log.debug("cache evicted %s", evicted_key)
        if evicted >= EVICTION_STORM_THRESHOLD:
            log.warning("cache evicted %d entries in one set(); max_entries=%d may be too small",
                        evicted, self._max_entries)

    def delete(self, key: str) -> bool:
        """Remove ``key``; returns True if an entry (live or expired) was present.

        Deleting an expired entry still counts as an expiration in the stats so
        that the expiration rate matches what Redis reports for the same load.
        """
        with self._lock:
            entry = self._data.pop(key, None)
            if entry is None:
                return False
            _, expires_at = entry
            if expires_at <= self._clock():
                self.stats.expirations += 1
                log.debug("deleted already-expired cache entry %s", key)
            return True

    def clear(self) -> None:
        """Drop every entry and reset the counters (used between tests and on config reload).

        The stats are replaced rather than zeroed in place so that anyone
        holding a reference to the old object (the metrics exporter) keeps the
        final values for the last scrape.
        """
        with self._lock:
            dropped = len(self._data)
            self._data.clear()
            self.stats = CacheStats()
        if dropped:
            log.info("cache cleared, %d entries dropped", dropped)

    def incr(self, key: str, amount: int = 1, ttl_seconds: float | None = None) -> int:
        """Atomically increment a counter; ``ttl_seconds`` applies only when the key is created."""
        if not isinstance(amount, int) or isinstance(amount, bool):
            raise TypeError("increment amount must be an int")
        if ttl_seconds is not None and ttl_seconds <= 0:
            raise ValueError("counter ttl_seconds must be positive when given")
        with self._lock:
            now = self._clock()
            value, expires = self._data.get(key, (0, 0.0))
            if expires <= now:
                # Missing or expired: start a fresh counter. Without a TTL the
                # counter lives until evicted, like a plain Redis INCR.
                value, expires = 0, now + ttl_seconds if ttl_seconds is not None else float("inf")
            if not isinstance(value, int):
                raise TypeError(f"cannot increment non-integer cache value at {key!r}")
            self._data[key] = (value + amount, expires)
            self._data.move_to_end(key)
            return value + amount

    def purge_expired(self) -> int:
        """Drop every expired entry; called from a periodic housekeeping task."""
        started = time.perf_counter()
        with self._lock:
            now = self._clock()
            expired = [key for key, (_, expires) in self._data.items() if expires <= now]
            for key in expired:
                del self._data[key]
            self.stats.expirations += len(expired)
            remaining = len(self._data)
        elapsed_ms = (time.perf_counter() - started) * 1000
        # The scan holds the lock; if it gets slow, request latency suffers.
        if elapsed_ms > 50:
            log.warning("purge_expired held the cache lock for %.1f ms (%d entries)", elapsed_ms, remaining)
        else:
            log.debug("purged %d expired entries, %d remain", len(expired), remaining)
        return len(expired)


class RedisBackend:
    """Thin adapter over a ``redis.Redis`` client implementing :class:`CacheBackend`.

    Values are pickled by the caller's serializer; we only deal with bytes in
    and out. Counters use INCRBY plus EXPIRE NX so the TTL is set only on
    creation, matching :meth:`TTLCache.incr`.
    """

    def __init__(self, client: Any, serializer: Any, prefix: str = "ll:") -> None:
        if client is None:
            raise ValueError("RedisBackend requires a redis client")
        for attr in ("dumps", "loads"):
            if not callable(getattr(serializer, attr, None)):
                raise TypeError(f"serializer must provide a callable {attr}()")
        if not prefix or not prefix.endswith(":"):
            # Keeps our keys visually separate from other tenants of the same
            # Redis cluster and lets ops SCAN by prefix.
            raise ValueError(f"redis key prefix must be non-empty and end with ':' (got {prefix!r})")
        self._client = client
        self._serializer = serializer
        self._prefix = prefix
        log.info("redis cache backend configured with prefix %s", prefix)

    def _k(self, key: str) -> str:
        """Namespace ``key`` with this backend's prefix."""
        if not key:
            raise ValueError("empty cache key")
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
            # Usually a value written by an older deploy with a different
            # pickle protocol. Drop it so the next caller repopulates it.
            log.error("corrupt cache entry at %s; deleting", key)
            try:
                self._client.delete(self._k(key))
            except Exception:
                log.warning("redis delete failed for corrupt entry %s", key, exc_info=True)
            return None

    def set(self, key: str, value: Any, ttl_seconds: float) -> None:
        """Store ``value`` under ``key``; failures are logged and swallowed like :meth:`get`."""
        if ttl_seconds <= 0:
            # Mirror TTLCache: a non-positive TTL means "don't cache".
            try:
                self._client.delete(self._k(key))
            except Exception:
                log.warning("redis delete failed for %s", key, exc_info=True)
            return
        try:
            payload = self._serializer.dumps(value)
        except Exception:
            log.error("cannot serialize %s for cache key %s; not caching", type(value).__name__, key)
            return
        # Redis expiries are integral; send whole seconds rather than a float
        # the server would reject.
        expire_seconds = int(ttl_seconds)
        try:
            self._client.set(self._k(key), payload, ex=expire_seconds)
        except Exception:
            log.warning("redis set failed for %s", key, exc_info=True)

    def incr(self, key: str, amount: int = 1, ttl_seconds: float | None = None) -> int:
        if not isinstance(amount, int) or isinstance(amount, bool):
            raise TypeError("increment amount must be an int")
        redis_key = self._k(key)
        pipe = self._client.pipeline()
        pipe.incrby(redis_key, amount)
        if ttl_seconds is not None:
            # EXPIRE takes whole seconds; round up so a counter never expires
            # before the window it is counting.
            pipe.expire(redis_key, max(int(-(-ttl_seconds // 1)), 1), xx=True)
        # Unlike get/set, counter failures propagate: silently allowing every
        # request through during an outage would disable rate limiting.
        value, *_ = pipe.execute()
        if value is None:
            raise RuntimeError(f"redis INCRBY returned no value for {redis_key}")
        return int(value)


def _stable_repr(value: Any) -> str:
    """Render ``value`` the same way in every process, for use inside cache keys.

    ``repr`` alone is not enough: set and dict ordering depends on the hash
    seed, and floats are banned because 0.1 + 0.2 style noise would split one
    logical key into many.
    """
    if isinstance(value, (str, int, bool, Decimal)) or value is None:
        return repr(value)
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(_stable_repr(v) for v in value) + "]"
    if isinstance(value, (set, frozenset)):
        # Sets have no stable iteration order across processes (hash seeds).
        return "{" + ",".join(_stable_repr(v) for v in value) + "}"
    if isinstance(value, dict):
        items = sorted((_stable_repr(k), _stable_repr(v)) for k, v in value.items())
        return "{" + ",".join(f"{k}:{v}" for k, v in items) + "}"
    if isinstance(value, float):
        raise CacheKeyError("floats are not allowed in cache keys; use Decimal")
    cache_key = getattr(value, "cache_key", None)
    if callable(cache_key):
        # Domain objects (Money, Sku...) can opt in by exposing cache_key().
        return f"{type(value).__name__}({cache_key()})"
    raise CacheKeyError(f"unhashable cache argument of type {type(value).__name__}")


def make_key(namespace: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> str:
    """Deterministic cache key for a call, independent of keyword-argument order."""
    if not namespace or len(namespace) > MAX_KEY_LENGTH:
        raise ValueError(f"invalid cache namespace {namespace!r}")
    if any(ch.isspace() for ch in namespace) or "|" in namespace:
        # "|" is our part separator and whitespace breaks redis-cli debugging.
        raise ValueError(f"cache namespace {namespace!r} contains reserved characters")
    non_str = [name for name in kwargs if not isinstance(name, str)]
    if non_str:
        raise CacheKeyError(f"keyword argument names must be strings: {non_str!r}")
    parts = [namespace]
    parts.extend(_stable_repr(arg) for arg in args)
    parts.extend(sorted(kwargs))
    # Hash rather than concatenate: arguments can be long (SKU lists) and the
    # digest keeps every key well under MAX_KEY_LENGTH.
    digest = hashlib.blake2b("|".join(parts).encode(), digest_size=16).hexdigest()
    key = f"{namespace}:{digest}"
    log.debug("cache key %s for %d args / %d kwargs", key, len(args), len(kwargs))
    return key


_default_backend: CacheBackend = TTLCache()
_backend_lock = threading.Lock()


def configure_default_backend(backend: CacheBackend) -> CacheBackend:
    """Swap the backend used by ``@memoize``; returns the previous one (for tests)."""
    global _default_backend
    missing = [name for name in ("get", "set", "incr") if not callable(getattr(backend, name, None))]
    if missing:
        raise TypeError(f"{type(backend).__name__} does not implement CacheBackend: missing {missing}")
    with _backend_lock:
        previous, _default_backend = _default_backend, backend
    if previous is backend:
        log.debug("memoize backend unchanged (%s)", type(backend).__name__)
    else:
        # Entries in the previous backend are not migrated; the first requests
        # after a swap are cold misses, which is expected during deploys.
        log.info("memoize backend set to %s (was %s)", type(backend).__name__, type(previous).__name__)
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
            if skip_first_arg and not args:
                raise TypeError(f"{func.__qualname__} memoized with skip_first_arg but called without arguments")
            try:
                key = make_key(namespace, args[1:] if skip_first_arg else args, kwargs)
            except CacheKeyError:
                # Uncacheable arguments: call through rather than fail the request.
                log.debug("memoize %s bypassed for uncacheable arguments", namespace)
                return func(*args, **kwargs)
            # Read the module global once so a concurrent backend swap cannot
            # make us read from one backend and write to another.
            backend = _default_backend
            cached = backend.get(key)
            if cached is not None:
                return cached
            result = func(*args, **kwargs)
            # None is never cached: it is how lookups signal "not found", and
            # caching it would hide newly created records for a whole TTL.
            if result is not None:
                backend.set(key, result, ttl_seconds)
            return result

        wrapper.cache_namespace = namespace  # type: ignore[attr-defined]
        return wrapper

    return decorator
