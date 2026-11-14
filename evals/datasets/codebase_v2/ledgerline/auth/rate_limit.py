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


class RateLimitConfigError(ValueError):
    pass


@dataclass(frozen=True)
class RateRule:
    name: str
    limit: int
    window_seconds: int
    burst: int = 0

    def __post_init__(self) -> None:
        if self.limit <= 0 or self.window_seconds <= 0:
            raise RateLimitConfigError(f"rule {self.name}: limit and window must be positive")
        if self.burst < 0:
            raise RateLimitConfigError(f"rule {self.name}: burst must be non-negative")

    @property
    def capacity(self) -> int:
        return self.limit + self.burst


@dataclass(frozen=True)
class Decision:
    allowed: bool
    remaining: int
    reset_after: int

    def headers(self) -> dict[str, str]:
        out = {"X-RateLimit-Remaining": str(max(self.remaining, 0)), "X-RateLimit-Reset": str(self.reset_after)}
        if not self.allowed:
            out["Retry-After"] = str(self.reset_after)
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
        self._store = store
        self._rules = dict(rules)
        self._clock = clock
        # tenant_id -> route class -> rule
        self._overrides: dict[str, dict[str, RateRule]] = {}

    def load_overrides(self, overrides: Mapping[str, Mapping[str, int]]) -> None:
        """Load purchased limits, e.g. ``{"t_acme": {"api.read": 5000}}``.

        Only the limit is overridden; window and burst come from the default
        rule so that bucket boundaries stay aligned across tenants.
        """
        parsed: dict[str, dict[str, RateRule]] = {}
        for tenant_id, limits in overrides.items():
            for route_class, limit in limits.items():
                base = self._rules.get(route_class)
                if base is None:
                    raise RateLimitConfigError(f"override for unknown rule {route_class!r} (tenant {tenant_id})")
                if limit < base.limit:
                    # Overrides are a paid upgrade; a lower number is a billing-sync bug.
                    log.warning("ignoring override below default for %s/%s: %d", tenant_id, route_class, limit)
                    continue
                parsed.setdefault(tenant_id, {})[route_class] = replace(base, limit=limit)
        self._overrides = parsed
        log.info("loaded rate-limit overrides for %d tenants", len(parsed))

    def rule_for(self, route_class: str, tenant_id: str | None = None) -> RateRule:
        """Resolve the rule for a route class, honouring the tenant's purchased overrides."""
        if route_class in self._rules:
            return self._rules[route_class]
        is_write = route_class.endswith(WRITE_SUFFIXES)
        fallback = "api.write" if is_write else "api.read"
        tenant_rules = self._overrides.get(tenant_id or "", {})
        if route_class in tenant_rules:
            return tenant_rules[route_class]
        if fallback in tenant_rules:
            return tenant_rules[fallback]
        return self._rules[fallback]

    def _bucket_key(self, claims: Claims, rule: RateRule, window_start: int) -> str:
        return f"rl:{claims.tenant_id}:{claims.subject}:{window_start}"

    def _window(self, rule: RateRule, now: float) -> tuple[int, int]:
        window_start = int(now // rule.window_seconds) * rule.window_seconds
        reset_after = max(int(window_start + rule.window_seconds - now), 1)
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
            used = self._store.incr(key, cost, ttl_seconds=rule.window_seconds + 5)
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
        return Decision(remaining >= 0, remaining, reset_after)

    def peek(self, claims: Claims, route_class: str) -> Decision:
        """Report the caller's current standing without consuming quota (used by /v1/limits)."""
        rule = self.rule_for(route_class, claims.tenant_id)
        now = self._clock()
        window_start, reset_after = self._window(rule, now)
        used = self._store.get(self._bucket_key(claims, rule, window_start)) or 0
        remaining = rule.capacity - int(used)
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
    method = method.upper()
    segments = [segment for segment in path.split("/") if segment]
    if segments and segments[0].startswith("v") and segments[0][1:].isdigit():
        segments = segments[1:]
    if not segments:
        return "api.read"
    if segments[:2] == ["auth", "password-reset"]:
        return "auth.password_reset"
    if segments[0] == "reports" and segments[-1] == "export":
        return "reports.export"
    resource = segments[0].replace("-", "_")
    action = _METHOD_ACTIONS.get(method)
    if action is None:
        # GET, HEAD and OPTIONS are all reads.
        return f"{resource}.read"
    return f"{resource}.{action}"


def rate_limit_response(decision: Decision) -> tuple[int, dict[str, str], dict[str, str]]:
    """Status, headers and JSON body for a rejected request."""
    body = {
        "code": "rate_limited",
        "message": f"Too many requests. Retry in {decision.reset_after} seconds.",
    }
    return 429, decision.headers(), body
