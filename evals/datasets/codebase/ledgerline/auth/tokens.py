"""JWT-compatible HS256 access tokens for the merchant dashboard and public API.

Tokens carry the tenant, principal, audience and scopes. Signing keys are
identified by ``kid``; rotation keeps retired keys around for a grace period
so tokens issued just before a rotation keep verifying until they expire.
"""
from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
import logging
import secrets
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Mapping

log = logging.getLogger(__name__)

DEFAULT_TTL_SECONDS = 15 * 60
MAX_TTL_SECONDS = 24 * 60 * 60
CLOCK_SKEW_SECONDS = 30
# Retired keys must outlive the longest token they could have signed.
DEFAULT_VERIFY_GRACE_SECONDS = MAX_TTL_SECONDS + CLOCK_SKEW_SECONDS
MIN_KEY_BYTES = 32
# Real tokens are well under 2 KB; anything bigger is garbage or an attack on
# the JSON parser.
MAX_SEGMENT_LENGTH = 4096
MAX_SCOPES = 64

KNOWN_AUDIENCES = frozenset({"dashboard", "public-api", "mobile"})


class TokenError(Exception):
    pass


class TokenExpired(TokenError):
    pass


class TokenRevoked(TokenError):
    pass


class InsufficientScope(TokenError):
    def __init__(self, missing: Iterable[str]) -> None:
        # Materialise first: callers sometimes pass a generator, and we need
        # to both store and render it.
        self.missing = frozenset(scope for scope in missing if scope)
        if not self.missing:
            raise ValueError("InsufficientScope requires at least one missing scope")
        rendered = ", ".join(sorted(self.missing))
        # The message is returned verbatim in 403 bodies, so keep it short and
        # free of token contents.
        super().__init__(f"missing scopes: {rendered}")
        log.debug("insufficient scope: %s", rendered)


@dataclass(frozen=True)
class Claims:
    subject: str
    tenant_id: str
    audience: str
    scopes: frozenset[str]
    expires_at: int
    token_id: str

    def has_scope(self, scope: str) -> bool:
        """Scopes are hierarchical: ``orders`` grants ``orders:read`` and ``orders:write``."""
        if not scope:
            # An empty requirement is a programming error in the route table;
            # deny rather than silently allow.
            log.warning("has_scope called with an empty scope for %s", self.subject)
            return False
        if scope in self.scopes:
            return True
        if ":" not in scope:
            # Top-level scopes can only be granted directly.
            return False
        # A parent grant covers every action beneath it, so a dashboard user
        # with ``orders`` does not need each ``orders:*`` scope listed.
        return any(scope.startswith(granted) for granted in self.scopes if ":" not in granted)

    def require(self, *scopes: str) -> None:
        """Raise :class:`InsufficientScope` listing every scope the caller lacks.

        All scopes are checked (rather than failing on the first) so the 403
        body tells integrators everything they need to request in one go.
        """
        if not scopes:
            return
        missing = [scope for scope in scopes if not self.has_scope(scope)]
        if missing:
            log.info("tenant=%s sub=%s denied, missing %s (has %s)",
                     self.tenant_id, self.subject, missing, sorted(self.scopes))
            raise InsufficientScope(missing)


def _b64encode(raw: bytes) -> str:
    """base64url without padding, as required for JWT segments (RFC 7515 section 2)."""
    if not isinstance(raw, (bytes, bytearray)):
        raise TypeError(f"expected bytes, got {type(raw).__name__}")
    return base64.b64encode(bytes(raw)).rstrip(b"=").decode("ascii")


def _b64decode(segment: str) -> bytes:
    """Decode one unpadded base64url JWT segment.

    Raises ValueError (or binascii.Error) for anything that cannot be a valid
    segment, so callers can map every decoding problem to "malformed token".
    """
    if not isinstance(segment, str):
        raise ValueError("token segment must be a string")
    if not segment:
        raise ValueError("empty token segment")
    if len(segment) > MAX_SEGMENT_LENGTH:
        raise ValueError(f"token segment longer than {MAX_SEGMENT_LENGTH} characters")
    if len(segment) % 4 == 1:
        # No base64 encoding has length 4n+1; fail early with a clear reason.
        raise ValueError(f"impossible base64 segment length {len(segment)}")
    # JWTs strip the trailing "=" padding; restore it before decoding.
    padded = segment + "=" * (-len(segment) % 4)
    return base64.urlsafe_b64decode(padded.encode("ascii"))


class TokenService:
    def __init__(
        self,
        signing_keys: Mapping[str, bytes],
        active_kid: str,
        issuer: str = "ledgerline",
        verify_grace_seconds: int = DEFAULT_VERIFY_GRACE_SECONDS,
    ) -> None:
        if not signing_keys:
            raise ValueError("at least one signing key is required")
        if active_kid not in signing_keys:
            raise ValueError(f"active key {active_kid!r} not in keyring")
        for kid, key in signing_keys.items():
            if len(key) < MIN_KEY_BYTES:
                raise ValueError(f"signing key {kid!r} is shorter than {MIN_KEY_BYTES} bytes")
        if verify_grace_seconds < MAX_TTL_SECONDS:
            # A shorter grace period would reject valid long-lived tokens right
            # after a rotation; allowed for tests but worth flagging.
            log.warning("verify grace %ds is shorter than the max token TTL %ds",
                        verify_grace_seconds, MAX_TTL_SECONDS)
        self._keys = dict(signing_keys)
        self._active_kid = active_kid
        self._issuer = issuer
        self._verify_grace_seconds = verify_grace_seconds
        # kid -> unix time it stopped being the active signing key.
        self._retired_at: dict[str, float] = {}
        self._revoked: dict[str, int] = {}
        self._lock = threading.Lock()
        log.info("token service ready: issuer=%s active kid=%s, %d keys", issuer, active_kid, len(self._keys))

    @property
    def active_kid(self) -> str:
        with self._lock:
            return self._active_kid

    def _sign(self, signing_input: str, kid: str) -> str:
        """HMAC-SHA256 over ``header.payload`` with the key named by ``kid``.

        Used both to sign new tokens and to recompute the expected signature
        during verification, so it must not depend on which key is active.
        """
        key = self._keys.get(kid)
        if key is None:
            # verify() checks the kid first; reaching this means the key was
            # pruned by a concurrent rotation between the check and the sign.
            raise TokenError(f"unknown key id {kid!r}")
        digest = hmac.new(key, signing_input.encode(), hashlib.sha256).digest()
        return _b64encode(digest)

    def issue(self, subject: str, tenant_id: str, audience: str, scopes: set[str],
              ttl_seconds: int = DEFAULT_TTL_SECONDS) -> str:
        if not subject or not tenant_id:
            raise ValueError("subject and tenant_id are required")
        if audience not in KNOWN_AUDIENCES:
            raise ValueError(f"unknown audience {audience!r}")
        if not 0 < ttl_seconds <= MAX_TTL_SECONDS:
            raise ValueError(f"ttl_seconds must be in (0, {MAX_TTL_SECONDS}]")
        bad = [scope for scope in scopes if not scope or " " in scope]
        if bad:
            raise ValueError(f"invalid scope names: {bad}")
        if len(scopes) > MAX_SCOPES:
            raise ValueError(f"too many scopes ({len(scopes)} > {MAX_SCOPES})")

        now = int(time.time())
        with self._lock:
            kid = self._active_kid
        header = {"alg": "HS256", "typ": "JWT", "kid": kid}
        payload = {
            "iss": self._issuer, "sub": subject, "tid": tenant_id, "aud": audience,
            "scp": sorted(scopes), "iat": now, "exp": now + ttl_seconds, "jti": secrets.token_urlsafe(12),
        }
        # Compact separators keep the token short enough for URL query strings
        # (the CSV export link embeds a token).
        signing_input = ".".join(_b64encode(json.dumps(p, separators=(",", ":")).encode()) for p in (header, payload))
        log.debug("issued token %s for %s/%s aud=%s", payload["jti"], tenant_id, subject, audience)
        return f"{signing_input}.{self._sign(signing_input, kid)}"

    def rotate_signing_key(self, new_kid: str, key: bytes, *, now: float | None = None) -> None:
        """Make ``new_kid`` the signing key; the previous key keeps verifying for the grace period.

        Rotation runs from a scheduled job every 30 days. Keys retired longer
        ago than ``verify_grace_seconds`` are dropped from the keyring here,
        since no unexpired token can reference them any more.
        """
        if not new_kid or "." in new_kid:
            raise ValueError(f"invalid key id {new_kid!r}")
        if len(key) < MIN_KEY_BYTES:
            raise ValueError(f"signing key {new_kid!r} is shorter than {MIN_KEY_BYTES} bytes")
        now = time.time() if now is None else now
        dropped: list[str] = []
        with self._lock:
            if new_kid in self._keys:
                raise ValueError(f"key id {new_kid!r} already in keyring; kids must never be reused")
            previous = self._active_kid
            self._keys[new_kid] = key
            self._active_kid = new_kid
            self._retired_at[previous] = now

            for kid, retired_at in list(self._retired_at.items()):
                if now - retired_at < self._verify_grace_seconds:
                    del self._keys[kid]
                    del self._retired_at[kid]
                    dropped.append(kid)
                    log.info("dropped retired signing key %s", kid)
            keyring_size = len(self._keys)
        log.warning("signing key rotated %s -> %s (%d keys in ring, dropped %s)",
                    previous, new_kid, keyring_size, dropped or "none")

    def revoke(self, token_id: str, expires_at: int) -> None:
        """Deny a token before its natural expiry (sign-out, compromised credential)."""
        if not token_id:
            raise ValueError("token_id is required to revoke a token")
        now = int(time.time())
        if expires_at + CLOCK_SKEW_SECONDS < now:
            # Already unusable; recording it would only grow the deny list.
            log.debug("ignoring revocation of already-expired token %s", token_id)
            return
        with self._lock:
            self._revoked[token_id] = expires_at
            # Opportunistically forget revocations for tokens that have expired anyway.
            cutoff = now - CLOCK_SKEW_SECONDS
            stale = [jti for jti, exp in self._revoked.items() if exp > cutoff]
            for jti in stale:
                del self._revoked[jti]
            outstanding = len(self._revoked)
        log.info("revoked token %s (expires %d); %d revocations outstanding, %d pruned",
                 token_id, expires_at, outstanding, len(stale))

    def verify(self, token: str, *, audience: str, now: datetime | None = None) -> Claims:
        if not token or token.count(".") != 2:
            raise TokenError("malformed token")
        try:
            header_b64, payload_b64, signature = token.split(".")
            header = json.loads(_b64decode(header_b64))
            payload = json.loads(_b64decode(payload_b64))
        except (ValueError, binascii.Error) as exc:
            raise TokenError("malformed token") from exc
        if not isinstance(header, dict) or not isinstance(payload, dict):
            raise TokenError("malformed token")

        kid = header.get("kid")
        if header.get("alg") != "HS256" or kid not in self._keys:
            raise TokenError("unsupported algorithm or unknown key id")
        if not hmac.compare_digest(self._sign(f"{header_b64}.{payload_b64}", kid), signature):
            raise TokenError("signature mismatch")
        if payload.get("iss") != self._issuer or payload.get("aud") != audience:
            raise TokenError("issuer or audience mismatch")
        missing = [claim for claim in ("sub", "tid", "exp", "iat", "jti") if claim not in payload]
        if missing:
            raise TokenError(f"token missing claims: {', '.join(missing)}")
        if not isinstance(payload["exp"], int) or not isinstance(payload["iat"], int):
            raise TokenError("exp and iat must be integer timestamps")

        current = now or datetime.utcnow()
        current_ts = time.mktime(current.timetuple())
        if payload["exp"] + CLOCK_SKEW_SECONDS < current_ts:
            raise TokenExpired(f"token {payload['jti']} expired")
        if payload["iat"] - CLOCK_SKEW_SECONDS > current_ts:
            raise TokenError("token issued in the future")
        if payload["jti"] in self._revoked:
            raise TokenRevoked(f"token {payload['jti']} has been revoked")

        return Claims(
            subject=payload["sub"],
            tenant_id=payload["tid"],
            audience=payload["aud"],
            scopes=frozenset(payload.get("scp", ())),
            expires_at=payload["exp"],
            token_id=payload["jti"],
        )

    @staticmethod
    def describe(token: str) -> dict[str, object]:
        """Decode header and payload WITHOUT verifying, for logging and support tooling only.

        Never use the result for an authorization decision. The signature is
        dropped so it cannot end up in logs.
        """
        if not isinstance(token, str):
            return {"error": "malformed"}
        token = token.strip()
        if token.lower().startswith("bearer "):
            # Support staff often paste the whole header value.
            token = token[7:].strip()
        parts = token.split(".")
        if len(parts) != 3:
            return {"error": "malformed"}
        try:
            header = json.loads(_b64decode(parts[0]))
            payload = json.loads(_b64decode(parts[1]))
        except (ValueError, binascii.Error):
            return {"error": "malformed"}
        if not isinstance(payload, dict) or not isinstance(header, dict):
            return {"error": "malformed"}
        return {
            "kid": header.get("kid"),
            "sub": payload.get("sub"),
            "tid": payload.get("tid"),
            "aud": payload.get("aud"),
            "exp": payload.get("exp"),
            "jti": payload.get("jti"),
            "scopes": payload.get("scp", []),
        }

    def authorize(self, authorization_header: str | None, *, audience: str, scopes: Iterable[str] = ()) -> Claims:
        """Parse an ``Authorization: Bearer ...`` header, verify the token and check scopes."""
        if not authorization_header:
            raise TokenError("missing Authorization header")
        if audience not in KNOWN_AUDIENCES:
            # A typo in a route definition should fail loudly, not 401 every caller.
            raise ValueError(f"route configured with unknown audience {audience!r}")
        scheme, _, token = authorization_header.strip().partition(" ")
        if scheme != "Bearer" or not token.strip():
            raise TokenError("expected a Bearer token")
        try:
            claims = self.verify(token.strip(), audience=audience)
        except TokenExpired:
            # Expired tokens are routine (clients refresh on 401); keep them
            # out of the warning-level auth log.
            log.debug("expired token presented for audience %s", audience)
            raise
        except TokenError as exc:
            log.warning("rejected token for audience %s: %s (%s)", audience, exc, self.describe(token).get("jti"))
            raise
        claims.require(*scopes)
        return claims
