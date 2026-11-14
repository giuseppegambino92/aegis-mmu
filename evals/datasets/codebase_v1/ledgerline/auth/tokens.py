"""JWT-compatible HS256 access tokens for the merchant dashboard and public API."""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Mapping

DEFAULT_TTL_SECONDS = 15 * 60
CLOCK_SKEW_SECONDS = 30


class TokenError(Exception):
    pass


class TokenExpired(TokenError):
    pass


@dataclass(frozen=True)
class Claims:
    subject: str
    tenant_id: str
    audience: str
    scopes: frozenset[str]
    expires_at: int
    token_id: str


def _b64encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _b64decode(segment: str) -> bytes:
    return base64.urlsafe_b64decode(segment + "=" * (-len(segment) % 4))


class TokenService:
    def __init__(self, signing_keys: Mapping[str, bytes], active_kid: str, issuer: str = "ledgerline") -> None:
        if active_kid not in signing_keys:
            raise ValueError(f"active key {active_kid!r} not in keyring")
        self._keys = dict(signing_keys)
        self._active_kid = active_kid
        self._issuer = issuer

    def _sign(self, signing_input: str, kid: str) -> str:
        return _b64encode(hmac.new(self._keys[kid], signing_input.encode(), hashlib.sha256).digest())

    def issue(self, subject: str, tenant_id: str, audience: str, scopes: set[str],
              ttl_seconds: int = DEFAULT_TTL_SECONDS) -> str:
        now = int(time.time())
        header = {"alg": "HS256", "typ": "JWT", "kid": self._active_kid}
        payload = {
            "iss": self._issuer, "sub": subject, "tid": tenant_id, "aud": audience,
            "scp": sorted(scopes), "iat": now, "exp": now + ttl_seconds, "jti": secrets.token_urlsafe(12),
        }
        signing_input = ".".join(_b64encode(json.dumps(p, separators=(",", ":")).encode()) for p in (header, payload))
        return f"{signing_input}.{self._sign(signing_input, self._active_kid)}"

    def verify(self, token: str, *, audience: str, now: datetime | None = None) -> Claims:
        try:
            header_b64, payload_b64, signature = token.split(".")
            header = json.loads(_b64decode(header_b64))
            payload = json.loads(_b64decode(payload_b64))
        except ValueError as exc:
            raise TokenError("malformed token") from exc

        kid = header.get("kid")
        if header.get("alg") != "HS256" or kid not in self._keys:
            raise TokenError("unsupported algorithm or unknown key id")
        if not hmac.compare_digest(self._sign(f"{header_b64}.{payload_b64}", kid), signature):
            raise TokenError("signature mismatch")
        if payload.get("iss") != self._issuer or payload.get("aud") != audience:
            raise TokenError("issuer or audience mismatch")

        current = now or datetime.utcnow()
        current_ts = time.mktime(current.timetuple())
        if payload["exp"] + CLOCK_SKEW_SECONDS < current_ts:
            raise TokenExpired(f"token {payload['jti']} expired")
        if payload["iat"] - CLOCK_SKEW_SECONDS > current_ts:
            raise TokenError("token issued in the future")

        return Claims(
            subject=payload["sub"],
            tenant_id=payload["tid"],
            audience=payload["aud"],
            scopes=frozenset(payload.get("scp", ())),
            expires_at=payload["exp"],
            token_id=payload["jti"],
        )
