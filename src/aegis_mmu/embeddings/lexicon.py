"""Identifier splitting, light stemming, a field-synonym lexicon and a software-engineering thesaurus.

Two generic priors feed the offline engine (no benchmark-specific identifiers):

* ``CONCEPT_GROUPS`` - synonyms seen in code and JSON APIs (``price``/``amount``/``fare``,
  ``origin``/``from``/``departure``...). Used for embeddings and by the schema healer.
* ``SE_THESAURUS`` - symptom <-> mechanism bridges an engineer applies when reading a bug report
  (``429``/``throttle`` <-> ``bucket``/``window``/``backoff``; ``401``/``403`` <-> ``jwt``/``bearer``/
  ``session``; ``504``/``timeout`` <-> ``circuit``/``pool``/``deadline``; ``oom``/``leak`` <->
  ``cache``/``evict``/``buffer``; ``race``/``deadlock`` <-> ``lock``/``mutex``/``atomic``...).

A word may belong to several concepts (``token`` is both auth and rate-limiting). Every concept a
word belongs to becomes a shared hashed feature; BM25 additionally runs a concept-expanded query.
"""

from __future__ import annotations

import re
from functools import lru_cache

_IDENT = re.compile(r"[A-Za-z][A-Za-z0-9]*|[0-9]+")
_CAMEL = re.compile(r"[A-Z]+(?=[A-Z][a-z]|[0-9]|\b)|[A-Z]?[a-z]+|[A-Z]+|[0-9]+")

CONCEPT_GROUPS: dict[str, tuple[str, ...]] = {
    "price": ("price", "amount", "cost", "fare", "total", "charge", "fee", "subtotal", "sum"),
    "identifier": ("id", "uid", "uuid", "identifier", "key", "ref", "reference", "pk", "guid"),
    "code": ("number", "num", "no", "code", "nr"),
    "person": ("user", "account", "customer", "member", "profile", "owner", "client"),
    "source": ("from", "origin", "source", "src", "departure", "depart", "departs"),
    "destination": ("to", "destination", "dest", "dst", "arrival", "arrive", "arrives"),
    "time": ("time", "timestamp", "datetime", "date", "at", "ts", "when"),
    "count": ("count", "qty", "quantity", "remaining", "left", "available", "availability", "capacity"),
    "name": ("name", "title", "label", "display"),
    "carrier": ("carrier", "airline", "operator", "vendor", "provider", "supplier", "merchant"),
    "duration": ("duration", "elapsed", "span", "minutes", "min", "mins", "seconds", "secs"),
    "currency": ("currency", "ccy", "iso"),
    "error": ("error", "err", "exception", "failure", "fail", "fault", "crash", "bug", "broken"),
    "delete": ("delete", "remove", "del", "drop", "purge", "evict", "destroy"),
    "create": ("create", "insert", "add", "new", "make", "register"),
    "fetch": ("get", "fetch", "load", "read", "retrieve", "find", "lookup"),
    "update": ("update", "modify", "patch", "edit", "write", "save", "store", "persist", "put"),
    "auth": ("auth", "authentication", "login", "signin", "session", "token", "jwt", "credential", "bearer", "csrf"),
    "secret": ("password", "passwd", "secret", "apikey", "pin", "otp"),
    "refund": ("refund", "reimburse", "reimbursement", "chargeback", "reversal"),
    "payment": ("payment", "pay", "invoice", "billing", "bill", "checkout", "transaction", "capture"),
    "message": ("email", "mail", "smtp", "notification", "notify", "message", "template"),
    "cache": ("cache", "memo", "memoize", "lru", "ttl"),
    "throttle": ("rate", "throttle", "limit", "limiter", "quota", "bucket", "burst"),
    "concurrency": ("lock", "mutex", "race", "thread", "concurrent", "atomic", "semaphore", "parallel"),
    "discount": ("discount", "coupon", "promo", "promotion", "voucher", "rebate"),
    "stock": ("stock", "inventory", "warehouse", "reserve", "reservation", "sku", "restock"),
    "order": ("order", "purchase", "cart", "basket"),
    "rounding": ("round", "rounding", "cent", "cents", "decimal", "precision", "quantize", "minor"),
    "conversion": ("convert", "conversion", "exchange", "fx"),
    "timezone": ("timezone", "tz", "utc", "zone", "offset", "local"),
    "retry": ("retry", "retries", "backoff", "attempt", "attempts", "jitter"),
    "expiry": ("expire", "expiry", "expiration", "expired", "stale", "ttl"),
    "leak": ("leak", "unbounded", "grow", "growth", "retain", "oom"),
    "sql": ("sql", "query", "select", "cursor", "execute", "injection", "where"),
    "connection": ("connection", "conn", "pool", "socket", "http"),
    "seat": ("seat", "cabin"),
    "aircraft": ("aircraft", "plane", "equipment"),
    "tax": ("tax", "taxes", "vat", "duty"),
}

SE_THESAURUS: dict[str, tuple[str, ...]] = {
    "se_rate_limiting": ("429", "rate", "ratelimit", "throttle", "throttled", "throttling", "quota", "quotas", "limit",
                         "limits", "limiter", "bucket", "buckets", "leaky", "window", "windows", "burst", "backoff", "token"),
    "se_auth_failure": ("401", "403", "unauthorized", "unauthenticated", "forbidden", "auth", "authenticate",
                        "authentication", "authorization", "authorize", "login", "logout", "signin", "session", "token",
                        "jwt", "bearer", "credential", "credentials", "permission", "permissions", "scope", "scopes",
                        "oauth", "csrf", "claims", "signature"),
    "se_upstream_failure": ("500", "502", "503", "504", "timeout", "timeouts", "deadline", "circuit", "breaker",
                            "unavailable", "gateway", "upstream", "pool", "socket", "connection", "retry", "retries",
                            "backoff", "hang", "hangs", "latency"),
    "se_memory_pressure": ("memory", "leak", "leaks", "oom", "heap", "cache", "evict", "eviction", "buffer", "gc", "ref",
                           "refs", "weakref", "retain", "unbounded", "growth", "rss"),
    "se_concurrency": ("race", "deadlock", "lock", "locks", "mutex", "atomic", "semaphore", "async", "await",
                       "concurrent", "concurrency", "thread", "threads", "parallel", "coroutine", "coroutines",
                       "simultaneous", "interleaving"),
    "se_not_found": ("404", "missing", "notfound", "absent", "lookup", "exists"),
    "se_conflict": ("409", "conflict", "duplicate", "duplicated", "idempotent", "idempotency", "already", "twice", "double"),
    "se_bad_input": ("400", "422", "invalid", "validate", "validation", "malformed", "reject", "rejected", "required"),
    "se_cache_key": ("cache", "memoize", "memo", "key", "keys", "hit", "stale", "collision", "collide", "hash", "digest",
                     "fingerprint"),
    "se_clock": ("timezone", "utc", "tz", "offset", "naive", "aware", "dst", "epoch", "timestamp", "clock", "now",
                 "midnight", "mktime", "localtime"),
    "se_expiry": ("expire", "expired", "expiry", "expiration", "ttl", "stale", "fresh", "refresh", "lifetime", "validity"),
    "se_money_precision": ("rounding", "round", "cent", "cents", "penny", "decimal", "precision", "quantize", "minor",
                           "fractional", "yen"),
    "se_persistence": ("sql", "query", "queries", "select", "insert", "cursor", "transaction", "commit", "rollback",
                       "repository", "repo", "db", "database", "row", "rows"),
    "se_notification": ("email", "mail", "smtp", "notification", "notify", "message", "template", "send", "sent"),
}

ALL_CONCEPTS: dict[str, tuple[str, ...]] = {**CONCEPT_GROUPS, **SE_THESAURUS}
CONCEPT_OF: dict[str, str] = {w: c for c, words in CONCEPT_GROUPS.items() for w in words}
CONCEPTS_OF: dict[str, tuple[str, ...]] = {}
for _c, _words in ALL_CONCEPTS.items():
    for _w in _words:
        CONCEPTS_OF[_w] = CONCEPTS_OF.get(_w, ()) + ((_c,) if _c not in CONCEPTS_OF.get(_w, ()) else ())

# Very frequent code tokens that carry little topical signal.
STOPWORDS: frozenset[str] = frozenset(
    """self cls this def return import from as if else elif for while in is not and or none null nil
    true false the a an of to be with on by it its that var let val const fn func fun function
    public private protected static final override async await try catch except finally raise throw
    new class struct interface type int str string float bool void any dict list of args kwargs pass
    i j k x y n v s e t""".split()
)


def stem(w: str) -> str:
    """Light suffix stripper (ing/ed/ies/es/s and a final silent e) so cache/cached/caching meet."""
    if len(w) > 4 and w.endswith("e") and not w.endswith(("ee", "ue")):
        return w[:-1]
    if len(w) > 5 and w.endswith("ing"):
        return w[:-3]
    if len(w) > 4 and w.endswith("ied"):
        return w[:-3] + "y"
    if len(w) > 4 and w.endswith("ed"):
        return w[:-2]
    if len(w) > 4 and w.endswith("ies"):
        return w[:-3] + "y"
    if len(w) > 4 and w.endswith("es") and w[-3] in "sxz":
        return w[:-2]
    if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
        return w[:-1]
    return w


def concept_of(w: str) -> str | None:
    c = CONCEPT_OF.get(w)
    if c is None:
        c = _CONCEPT_OF_STEM.get(stem(w))
    return c


def concepts_of(w: str) -> tuple[str, ...]:
    """Every concept (field-synonym group and SE-thesaurus group) a word belongs to."""
    c = CONCEPTS_OF.get(w)
    if c is None:
        c = _CONCEPTS_OF_STEM.get(stem(w), ())
    return c


def expand_term(w: str, limit: int = 24) -> tuple[str, ...]:
    """Sibling words of every concept of ``w`` (sparse query expansion), one per stem, excluding ``w``."""
    out: dict[str, None] = {}
    stems = {stem(w)}
    for c in concepts_of(w):
        for x in ALL_CONCEPTS[c]:
            st = stem(x)
            if x != w and len(x) >= 2 and st not in stems:
                stems.add(st)
                out.setdefault(x, None)
    return tuple(out)[:limit]


def split_identifier(ident: str) -> list[str]:
    out: list[str] = []
    for part in ident.split("_"):
        if not part:
            continue
        for m in _CAMEL.findall(part):
            out.append(m.lower())
    return out


@lru_cache(maxsize=16384)
def subword_tokens(text: str, min_len: int = 2) -> tuple[str, ...]:
    """Split text into lower-cased subwords (snake_case and camelCase aware). Memoised (pure)."""
    toks: list[str] = []
    for m in _IDENT.finditer(text):
        for s in split_identifier(m.group()):
            if len(s) >= min_len or s.isdigit():
                toks.append(s)
    return tuple(toks)


_CONCEPT_OF_STEM: dict[str, str] = {}
for _c, _words in CONCEPT_GROUPS.items():
    for _w in _words:
        _CONCEPT_OF_STEM.setdefault(stem(_w), _c)

_CONCEPTS_OF_STEM: dict[str, tuple[str, ...]] = {}
for _w, _cs in CONCEPTS_OF.items():
    _st = stem(_w)
    _CONCEPTS_OF_STEM[_st] = _CONCEPTS_OF_STEM.get(_st, ()) + tuple(c for c in _cs if c not in _CONCEPTS_OF_STEM.get(_st, ()))
