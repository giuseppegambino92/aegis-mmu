"""Fixed-window rate limiting per tenant, principal and route class.

Each request is classified into a route class (``api.read``, ``api.write``,
``reports.export``...). The limiter counts requests per caller in fixed,
epoch-aligned windows using the shared cache backend, so every API node sees
the same counters without coordination.

Enterprise tenants can buy higher limits; those are loaded as per-tenant
rule overrides from the billing service and take precedence over defaults.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, replace
from typing import Callable, Mapping

from ledgerline.auth.tokens import Claims
from ledgerline.infra.cache import CacheBackend

log = logging.getLogger(__name__)

WRITE_SUFFIXES = (".create", ".update", ".delete")
# Hard ceiling on the cost of a single request so a buggy caller cannot burn a
# whole window with one call (bulk endpoints charge per item).
MAX_REQUEST_COST = 100
# Purchased limits above this are almost certainly a unit mix-up in billing
# (per-hour figures loaded as per-minute) and are capped.
MAX_OVERRIDE_LIMIT = 100_000
# Extra lifetime for counters beyond their window so a slow node that
# increments right at the boundary still finds the key.
COUNTER_TTL_SLACK_SECONDS = 5


class RateLimitConfigError(ValueError):
    pass


@dataclass(frozen=True)
class RateRule:
    name: str
    limit: int
    window_seconds: int
    burst: int = 0

    def __post_init__(self) -> None:
        if not self.name or " " in self.name:
            raise RateLimitConfigError(f"invalid rule name {self.name!r}")
        if self.limit <= 0 or self.window_seconds <= 0:
            raise RateLimitConfigError(f"rule {self.name}: limit and window must be positive")
        if self.burst < 0:
            raise RateLimitConfigError(f"rule {self.name}: burst must be non-negative")
        if self.burst > self.limit:
            # Burst is headroom on top of the steady limit; a burst larger than
            # the limit itself usually means the two fields were swapped.
            raise RateLimitConfigError(f"rule {self.name}: burst {self.burst} exceeds limit {self.limit}")
        if 86_400 % self.window_seconds and self.window_seconds < 86_400:
            # Windows that do not divide a day drift against the daily usage
            # reports; allowed, but flag it at startup.
            log.warning("rule %s: window %ds does not divide a day evenly", self.name, self.window_seconds)

    @property
    def capacity(self) -> int:
        """Total requests allowed per window: the steady limit plus burst headroom."""
        return self.limit + self.burst


@dataclass(frozen=True)
class Decision:
    allowed: bool
    remaining: int
    reset_after: int

    def headers(self) -> dict[str, str]:
        """Standard rate-limit response headers for this decision.

        ``remaining`` can go negative internally (the counter keeps counting
        rejected requests) but is never shown below zero to clients.
        ``Retry-After`` is only sent on rejections, per RFC 6585.
        """
        reset_after = max(int(self.reset_after), 0)
        out = {
            "X-RateLimit-Remaining": str(max(self.remaining, 0)),
            "X-RateLimit-Reset": str(reset_after),
        }
        if not self.allowed:
            # Never tell a client to retry immediately; a zero here causes
            # tight retry loops in several popular HTTP client libraries.
            out["Retry-After"] = str(max(reset_after, 1))
        return out


DEFAULT_RULES: Mapping[str, RateRule] = {
    "auth.password_reset": RateRule("auth.password_reset", limit=5, window_seconds=900),
    "api.read": RateRule("api.read", limit=600, window_seconds=60, burst=100),
    "api.write": RateRule("api.write", limit=120, window_seconds=60, burst=20),
    "reports.export": RateRule("reports.export", limit=10, window_seconds=3600),
}


class RateLimiter:
    def __init__(
        self, store: CacheBackend, rules: Mapping[str, RateRule] = DEFAULT_RULES,
        clock: Callable[[], float] = time.time,
    ) -> None:
        missing = {"api.read", "api.write"} - set(rules)
        if missing:
            raise RateLimitConfigError(f"rule set must define fallback rules: {sorted(missing)}")
        mismatched = [key for key, rule in rules.items() if rule.name != key]
        if mismatched:
            # The rule name ends up in logs and metrics; a mismatch makes
            # dashboards attribute throttling to the wrong route class.
            raise RateLimitConfigError(f"rule keys do not match rule names: {sorted(mismatched)}")
        self._store = store
        self._rules = dict(rules)
        self._clock = clock
        # tenant_id -> route class -> rule
        self._overrides: dict[str, dict[str, RateRule]] = {}
        log.info("rate limiter configured with rules %s", ", ".join(sorted(self._rules)))

    def load_overrides(self, overrides: Mapping[str, Mapping[str, int]]) -> None:
        """Load purchased limits, e.g. ``{"t_acme": {"api.read": 5000}}``.

        Only the limit is overridden; window and burst come from the default
        rule so that bucket boundaries stay aligned across tenants.
        """
        parsed: dict[str, dict[str, RateRule]] = {}
        for tenant_id, limits in overrides.items():
            if not tenant_id:
                raise RateLimitConfigError("override with an empty tenant id")
            for route_class, limit in limits.items():
                base = self._rules.get(route_class)
                if base is None:
                    raise RateLimitConfigError(f"override for unknown rule {route_class!r} (tenant {tenant_id})")
                if limit < base.limit:
                    # Overrides are a paid upgrade; a lower number is a billing-sync bug.
                    log.warning("ignoring override below default for %s/%s: %d", tenant_id, route_class, limit)
                    continue
                if limit > MAX_OVERRIDE_LIMIT:
                    log.error("capping override for %s/%s from %d to %d", tenant_id, route_class,
                              limit, MAX_OVERRIDE_LIMIT)
                    limit = MAX_OVERRIDE_LIMIT
                parsed.setdefault(tenant_id, {})[route_class] = replace(base, limit=limit)
        # Swap in one assignment so concurrent checks see either the old or
        # the new set of overrides, never a half-loaded mix.
        self._overrides = parsed
        log.info("loaded rate-limit overrides for %d tenants", len(parsed))

    def rule_for(self, route_class: str, tenant_id: str | None = None) -> RateRule:
        """Resolve the rule for a route class, honouring the tenant's purchased overrides."""
        if not route_class:
            raise ValueError("route_class is required")
        if route_class in self._rules:
            return self._rules[route_class]
        # Unconfigured classes fall back to the generic read/write rule based
        # on the action suffix produced by classify_route().
        is_write = route_class.endswith(WRITE_SUFFIXES)
        fallback = "api.write" if is_write else "api.read"
        tenant_rules = self._overrides.get(tenant_id or "", {})
        if route_class in tenant_rules:
            return tenant_rules[route_class]
        if fallback in tenant_rules:
            return tenant_rules[fallback]
        return self._rules[fallback]

    def _bucket_key(self, claims: Claims, rule: RateRule, window_start: int) -> str:
        """Cache key for the caller's counter in one window.

        Keys are scoped by tenant and principal so one noisy API key cannot
        exhaust its tenant's other integrations. The window start is part of
        the key, so each window gets a fresh counter and old ones simply age
        out through the store TTL.
        """
        if not claims.tenant_id or not claims.subject:
            # Anonymous traffic is limited at the edge proxy, never here.
            raise ValueError("rate limiting requires an authenticated tenant and subject")
        if window_start < 0:
            raise ValueError(f"negative window start {window_start}")
        return f"rl:{claims.tenant_id}:{claims.subject}:{window_start}"

    def _window(self, rule: RateRule, now: float) -> tuple[int, int]:
        """Return ``(window_start, reset_after)`` for the window containing ``now``.

        Windows are aligned to the Unix epoch so that every API node computes
        the same boundaries from its own clock. ``reset_after`` is the number
        of whole seconds until the next window opens, never less than one.
        """
        if now < 0:
            raise ValueError(f"clock returned a negative timestamp: {now}")
        width = rule.window_seconds
        # Integer window index; the counter key embeds the start second, so the
        # result must be identical across nodes for the same instant.
        window_start = int(round(now / width)) * width
        reset_after = max(int(window_start + width - now), 1)
        return window_start, reset_after

    def check(self, claims: Claims, route_class: str, cost: int = 1) -> Decision:
        """Count ``cost`` units against the caller's bucket for this route class.

        Windows are aligned to the epoch so every API node agrees on bucket
        boundaries without coordination.
        """
        if cost <= 0:
            raise ValueError("cost must be positive")
        if cost > MAX_REQUEST_COST:
            log.warning("clamping request cost %d to %d for %s", cost, MAX_REQUEST_COST, route_class)
            cost = MAX_REQUEST_COST
        rule = self.rule_for(route_class, claims.tenant_id)
        now = self._clock()
        window_start, reset_after = self._window(rule, now)

        key = self._bucket_key(claims, rule, window_start)
        try:
            used = self._store.incr(key, cost, ttl_seconds=rule.window_seconds + COUNTER_TTL_SLACK_SECONDS)
        except Exception:
            # Fail open for reads (availability matters more), closed for
            # writes and auth flows (abuse protection matters more).
            fail_open = rule.name == "api.read"
            log.error("rate-limit store unavailable for %s; failing %s", rule.name,
                      "open" if fail_open else "closed", exc_info=True)
            return Decision(fail_open, 0, reset_after)
        remaining = rule.limit + rule.burst - used
        if remaining < 0:
            log.info("rate limited tenant=%s sub=%s rule=%s", claims.tenant_id, claims.subject, rule.name)
        return Decision(remaining > 0, remaining, reset_after)

    def peek(self, claims: Claims, route_class: str) -> Decision:
        """Report the caller's current standing without consuming quota (used by /v1/limits)."""
        rule = self.rule_for(route_class, claims.tenant_id)
        now = self._clock()
        window_start, reset_after = self._window(rule, now)
        key = self._bucket_key(claims, rule, window_start)
        try:
            raw = self._store.get(key)
        except Exception:
            # /v1/limits is informational; report a full bucket rather than 500.
            log.warning("rate-limit store unavailable for peek on %s", rule.name, exc_info=True)
            raw = None
        try:
            used = int(raw or 0)
        except (TypeError, ValueError):
            log.error("non-integer rate-limit counter at %s: %r", key, raw)
            used = 0
        remaining = rule.limit - used
        return Decision(remaining >= 0, remaining, reset_after)

    def reset(self, claims: Claims, route_class: str, *, actor: str) -> None:
        """Clear the caller's current window, e.g. when support unlocks a password-reset lockout.

        Only the current window is touched; counters for past windows expire
        on their own via the store TTL.
        """
        if not actor:
            raise ValueError("reset requires the acting support agent for the audit trail")
        rule = self.rule_for(route_class, claims.tenant_id)
        window_start, _ = self._window(rule, self._clock())
        key = self._bucket_key(claims, rule, window_start)
        # The backend protocol has no delete; an immediately-expiring write is
        # equivalent and works for both TTLCache and Redis.
        self._store.set(key, 0, ttl_seconds=0.001)
        log.warning("rate limit for tenant=%s sub=%s rule=%s reset by %s",
                    claims.tenant_id, claims.subject, rule.name, actor)


_METHOD_ACTIONS = {"POST": "create", "PUT": "update", "PATCH": "update", "DELETE": "delete"}


def classify_route(method: str, path: str) -> str:
    """Map an HTTP request to a route class, e.g. ``POST /v1/orders`` -> ``orders.create``.

    Explicitly configured classes (password reset, exports) are matched on
    path first because they have much tighter limits than generic writes.
    """
    method = (method or "GET").upper()
    # Ignore the query string and any trailing slash; both vary between SDKs.
    path = path.split("?", 1)[0]
    segments = [segment for segment in path.split("/") if segment]
    if segments and segments[0].startswith("v") and segments[0][1:].isdigit():
        segments = segments[1:]
    if not segments:
        return "api.read"
    if segments[:2] == ["auth", "password-reset"]:
        return "auth.password_reset"
    if segments[:2] == ["reports", "export"]:
        return "reports.export"
    resource = segments[0].replace("-", "_").lower()
    action = _METHOD_ACTIONS.get(method)
    if action is None:
        # GET, HEAD and OPTIONS are all reads.
        return f"{resource}.read"
    return f"{resource}.{action}"


def rate_limit_response(decision: Decision) -> tuple[int, dict[str, str], dict[str, str]]:
    """Status, headers and JSON body for a rejected request."""
    if decision.allowed:
        # Callers should only build a 429 for rejected decisions; returning one
        # for an allowed request would throttle a caller who has quota left.
        raise ValueError("rate_limit_response called for an allowed decision")
    retry_in = max(int(decision.reset_after), 1)
    unit = "second" if retry_in == 1 else "seconds"
    body = {
        "code": "rate_limited",
        "message": f"Too many requests. Retry in {retry_in} {unit}.",
    }
    log.debug("429 issued, retry in %d %s", retry_in, unit)
    return 429, decision.headers(), body
