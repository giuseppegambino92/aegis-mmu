"""Fixed-window rate limiting per tenant, principal and route class."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Callable, Mapping

from ledgerline.auth.tokens import Claims
from ledgerline.infra.cache import CacheBackend

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class RateRule:
    name: str
    limit: int
    window_seconds: int
    burst: int = 0


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
        self._store = store
        self._rules = dict(rules)
        self._clock = clock

    def rule_for(self, route_class: str) -> RateRule:
        if route_class in self._rules:
            return self._rules[route_class]
        is_write = route_class.endswith((".create", ".update", ".delete"))
        return self._rules["api.write" if is_write else "api.read"]

    def _bucket_key(self, claims: Claims, rule: RateRule, window_start: int) -> str:
        return f"rl:{claims.tenant_id}:{claims.subject}:{window_start}"

    def check(self, claims: Claims, route_class: str, cost: int = 1) -> Decision:
        """Count ``cost`` units against the caller's bucket for this route class.

        Windows are aligned to the epoch so every API node agrees on bucket
        boundaries without coordination.
        """
        rule = self.rule_for(route_class)
        now = self._clock()
        window_start = int(now // rule.window_seconds) * rule.window_seconds
        reset_after = max(int(window_start + rule.window_seconds - now), 1)

        key = self._bucket_key(claims, rule, window_start)
        used = self._store.incr(key, cost, ttl_seconds=rule.window_seconds + 5)
        remaining = rule.limit + rule.burst - used
        if remaining < 0:
            log.info("rate limited tenant=%s sub=%s rule=%s", claims.tenant_id, claims.subject, rule.name)
        return Decision(remaining >= 0, remaining, reset_after)
