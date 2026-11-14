"""HAR / HTTP-trace ingestion, Shannon-entropy token detection and backward taint slicing.

Data-flow edges ``R_a -> R_b`` are added whenever a *dynamic* value produced by the response of
``R_a`` (JSON leaf, response header or ``Set-Cookie``) is consumed by the request of ``R_b`` (URL
path/query, header such as ``Authorization``/``X-CSRF-Token``, cookie, or body). A value is
considered dynamic when it looks like a token/nonce/identifier:

    H(X) = - sum_x P(x) log2 P(x)   (per-character Shannon entropy)

strings with ``len >= 8`` and ``H >= 3.0`` bits (or ``len >= 16`` and ``H >= 2.5``) and numeric
values of id-like keys. Low-entropy values (airport codes, dates, booleans) never create edges,
which is what separates the CSRF/JWT chain from coincidental matches.

Given a target output field, the backward slice from the response containing it yields the
minimal execution chain; everything else (analytics, ads, assets, preflights, decoy API calls,
downstream consumers) is pruned.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter, deque
from dataclasses import dataclass, field
from http.cookies import SimpleCookie
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import parse_qsl, unquote, urlsplit

TRACKER_RE = re.compile(r"(google-analytics|googletagmanager|doubleclick|googlesyndication|facebook|fbcdn|hotjar|segment|"
                        r"sentry|mixpanel|amplitude|newrelic|datadoghq|clarity\.ms|tiktok|linkedin|bing\.com/|/collect\b|/pixel|/tr\b)", re.I)
STATIC_EXT_RE = re.compile(r"\.(js|mjs|css|png|jpe?g|gif|svg|webp|ico|woff2?|ttf|otf|map|webmanifest)(\?|$)", re.I)
ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}([T ]\d{2}:\d{2}(:\d{2}(\.\d+)?)?(Z|[+-]\d{2}:?\d{2})?)?$")
ID_KEY_RE = re.compile(r"(^id$|_id$|Id$|^uid$|^uuid$|_uuid$|^pk$)")
SKIP_RESP_HEADERS = {"content-type", "content-length", "date", "server", "vary", "cache-control", "etag", "expires",
                     "access-control-allow-origin", "access-control-allow-headers", "access-control-allow-credentials",
                     "access-control-allow-methods", "connection", "content-encoding", "age", "last-modified",
                     "strict-transport-security", "x-content-type-options", "x-frame-options", "alt-svc", "via"}


def shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    n = len(s)
    return -sum((c / n) * math.log2(c / n) for c in Counter(s).values())


def flatten_json(obj: Any, prefix: str = "$") -> list[tuple[str, Any]]:
    """Leaf (JSONPath, value) pairs with concrete indices, e.g. ``$.items[0].price``."""
    out: list[tuple[str, Any]] = []
    stack: list[tuple[str, Any]] = [(prefix, obj)]
    while stack:
        p, v = stack.pop()
        if isinstance(v, dict):
            for k in reversed(list(v.keys())):
                key = k if re.match(r"^[A-Za-z_][\w$-]*$", str(k)) else json.dumps(str(k))
                stack.append((f"{p}.{key}", v[k]))
        elif isinstance(v, list):
            for i in range(len(v) - 1, -1, -1):
                stack.append((f"{p}[{i}]", v[i]))
        else:
            out.append((p, v))
    return out


def canonical_path(path: str) -> str:
    return re.sub(r"\[\d+\]", "[*]", path)


def last_key(path: str) -> str:
    m = re.search(r"\.([^.\[\]]+)(?:\[[\d*]+\])*$", path)
    return m.group(1).strip('"') if m else path


@dataclass
class Txn:
    index: int
    method: str
    url: str
    origin: str
    path: str
    query: list[tuple[str, str]]
    req_headers: dict[str, str]
    req_cookies: dict[str, str]
    req_body: str
    req_json: Any
    req_mime: str
    status: int
    resp_headers: dict[str, str]
    set_cookies: dict[str, str]
    resp_body: str
    resp_json: Any
    mime: str
    started: str = ""
    category: str = "api"

    @property
    def short(self) -> str:
        return f"#{self.index} {self.method} {self.origin}{self.path}"


@dataclass
class ValueRef:
    value: str
    txn: int
    location: str  # body:$.a.b | header:name | cookie:name
    key: str
    entropy: float


@dataclass
class Edge:
    src: int
    dst: int
    value: str
    src_loc: str
    dst_loc: str
    entropy: float

    def to_dict(self) -> dict:
        v = self.value
        return {"src": self.src, "dst": self.dst, "value_preview": (v[:10] + "…" + v[-4:]) if len(v) > 18 else v,
                "src_loc": self.src_loc, "dst_loc": self.dst_loc, "entropy": round(self.entropy, 3)}


@dataclass
class MinimalChain:
    target: str
    target_index: int
    target_path: str
    chain: list[int]
    edges: list[Edge]
    txns: list[Txn]
    categories: dict[str, int] = field(default_factory=dict)

    @property
    def pruned(self) -> list[int]:
        keep = set(self.chain)
        return [t.index for t in self.txns if t.index not in keep]

    @property
    def noise_reduction(self) -> float:
        return 1.0 - len(self.chain) / max(1, len(self.txns))

    def steps(self) -> list[Txn]:
        return [self.txns[i] for i in self.chain]

    def to_dict(self) -> dict:
        return {
            "target": self.target, "target_index": self.target_index, "target_path": self.target_path,
            "chain": [{"index": t.index, "method": t.method, "url": t.origin + t.path, "status": t.status} for t in self.steps()],
            "edges": [e.to_dict() for e in self.edges if e.src in self.chain and e.dst in self.chain],
            "n_requests": len(self.txns), "n_pruned": len(self.pruned),
            "noise_reduction_pct": round(100 * self.noise_reduction, 2), "categories": self.categories,
        }


# ----------------------------------------------------------------------------- parsing


def _hdict(items: Iterable[dict] | dict | None) -> dict[str, str]:
    if items is None:
        return {}
    if isinstance(items, dict):
        return {str(k).lower(): str(v) for k, v in items.items()}
    out: dict[str, str] = {}
    for h in items:
        name = str(h.get("name", "")).lower()
        if name.startswith(":"):
            continue
        val = str(h.get("value", ""))
        out[name] = f"{out[name]}\n{val}" if name in out and name == "set-cookie" else val
    return out


def _try_json(text: str, mime: str = "") -> Any:
    if not text:
        return None
    t = text.lstrip()
    if not t or t[0] not in "[{":
        return None
    try:
        return json.loads(t)
    except ValueError:
        return None


def _parse_set_cookie(raw: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in raw.split("\n"):
        if not line.strip():
            continue
        name_val = line.split(";", 1)[0]
        if "=" in name_val:
            k, v = name_val.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def _parse_cookie_header(raw: str) -> dict[str, str]:
    c = SimpleCookie()
    try:
        c.load(raw)
    except Exception:  # noqa: BLE001 - malformed cookies are common in captures
        return dict(p.strip().split("=", 1) for p in raw.split(";") if "=" in p)
    return {k: m.value for k, m in c.items()}


def _categorise(t: Txn) -> str:
    if t.method == "OPTIONS":
        return "preflight"
    if TRACKER_RE.search(t.origin + t.path):
        return "tracker"
    if STATIC_EXT_RE.search(t.path) or t.mime.split(";")[0] in ("text/css", "application/javascript", "font/woff2") \
            or t.mime.startswith(("image/", "font/")):
        return "static"
    if t.mime.startswith("text/html"):
        return "document"
    return "api"


def parse_trace(data: dict | list | str | Path) -> list[Txn]:
    """Parse a HAR 1.2 document or a JSON list of ``{method,url,request_headers,...}`` transactions."""
    if isinstance(data, (str, Path)) and not (isinstance(data, str) and data.lstrip().startswith(("{", "["))):
        data = json.loads(Path(data).read_text())
    elif isinstance(data, str):
        data = json.loads(data)
    txns: list[Txn] = []
    if isinstance(data, dict) and "log" in data:
        for i, e in enumerate(data["log"].get("entries", [])):
            rq, rs = e.get("request", {}), e.get("response", {})
            url = rq.get("url", "")
            sp = urlsplit(url)
            req_h = _hdict(rq.get("headers"))
            resp_h = _hdict(rs.get("headers"))
            cookies = {c["name"]: c.get("value", "") for c in rq.get("cookies", []) if "name" in c}
            if "cookie" in req_h:
                cookies.update(_parse_cookie_header(req_h["cookie"]))
            set_c = {c["name"]: c.get("value", "") for c in rs.get("cookies", []) if "name" in c}
            if "set-cookie" in resp_h:
                set_c.update(_parse_set_cookie(resp_h["set-cookie"]))
            post = rq.get("postData") or {}
            body = post.get("text") or ""
            if not body and post.get("params"):
                body = "&".join(f"{p['name']}={p.get('value', '')}" for p in post["params"])
            content = rs.get("content") or {}
            rbody = content.get("text") or ""
            if content.get("encoding") == "base64":
                import base64

                try:
                    rbody = base64.b64decode(rbody).decode("utf-8", "replace")
                except Exception:  # noqa: BLE001
                    rbody = ""
            mime = content.get("mimeType") or resp_h.get("content-type", "")
            t = Txn(i, rq.get("method", "GET").upper(), url, f"{sp.scheme}://{sp.netloc}", sp.path or "/",
                    parse_qsl(sp.query, keep_blank_values=True), req_h, cookies, body, _try_json(body),
                    post.get("mimeType", req_h.get("content-type", "")), int(rs.get("status", 0) or 0), resp_h, set_c,
                    rbody, _try_json(rbody), mime, e.get("startedDateTime", ""))
            txns.append(t)
    elif isinstance(data, list):
        for i, e in enumerate(data):
            url = e.get("url", "")
            sp = urlsplit(url)
            req_h = _hdict(e.get("request_headers") or e.get("headers"))
            resp_h = _hdict(e.get("response_headers"))
            body = e.get("request_body") or ""
            body = body if isinstance(body, str) else json.dumps(body)
            rbody = e.get("response_body") or ""
            rbody = rbody if isinstance(rbody, str) else json.dumps(rbody)
            cookies = _parse_cookie_header(req_h["cookie"]) if "cookie" in req_h else {}
            set_c = _parse_set_cookie(resp_h["set-cookie"]) if "set-cookie" in resp_h else {}
            t = Txn(i, str(e.get("method", "GET")).upper(), url, f"{sp.scheme}://{sp.netloc}", sp.path or "/",
                    parse_qsl(sp.query, keep_blank_values=True), req_h, cookies, body, _try_json(body),
                    req_h.get("content-type", ""), int(e.get("status", 200)), resp_h, set_c, rbody, _try_json(rbody),
                    resp_h.get("content-type", "application/json"))
            txns.append(t)
    else:
        raise ValueError("unsupported trace format: expected HAR {'log': ...} or a list of transactions")
    for t in txns:
        t.category = _categorise(t)
    return txns


# ----------------------------------------------------------------------------- taint graph


def is_dynamic(value: Any, key: str = "") -> tuple[bool, float]:
    if isinstance(value, bool) or value is None:
        return False, 0.0
    if isinstance(value, (int, float)):
        s = str(value)
        return (bool(ID_KEY_RE.search(key)) and isinstance(value, int) and len(s) >= 3), shannon_entropy(s)
    s = str(value)
    if " " in s.strip() or ISO_DATE_RE.match(s):
        return False, shannon_entropy(s)
    h = shannon_entropy(s)
    if s.isdigit():
        return (bool(ID_KEY_RE.search(key)) and len(s) >= 3) or len(s) >= 12, h
    ok = (len(s) >= 8 and h >= 3.0) or (len(s) >= 16 and h >= 2.5)
    if ok and re.fullmatch(r"https?://\S+", s) and STATIC_EXT_RE.search(s):
        ok = False
    return ok, h


class TaintGraph:
    def __init__(self, txns: list[Txn]):
        self.txns = txns
        self.produced: list[list[ValueRef]] = [self._produced(t) for t in txns]
        self.edges: list[Edge] = []
        self._build()

    @staticmethod
    def _produced(t: Txn) -> list[ValueRef]:
        out: list[ValueRef] = []
        seen: set[str] = set()

        def add(v: Any, loc: str, key: str) -> None:
            ok, h = is_dynamic(v, key)
            sv = str(v)
            if ok and sv not in seen:
                seen.add(sv)
                out.append(ValueRef(sv, t.index, loc, key, h))

        if t.resp_json is not None:
            for p, v in flatten_json(t.resp_json):
                add(v, "body:" + p, last_key(p))
        for k, v in t.set_cookies.items():
            add(v, "cookie:" + k, k)
        for k, v in t.resp_headers.items():
            if k not in SKIP_RESP_HEADERS and k != "set-cookie":
                add(v, "header:" + k, k)
        return out

    @staticmethod
    def _consumers(t: Txn) -> list[tuple[str, str]]:
        hay: list[tuple[str, str]] = [("path", unquote(t.path))]
        hay += [(f"query:{k}", v) for k, v in t.query]
        hay += [(f"header:{k}", v) for k, v in t.req_headers.items() if k not in ("cookie",)]
        hay += [(f"cookie:{k}", v) for k, v in t.req_cookies.items()]
        if t.req_json is not None:
            hay += [("body:" + p, str(v)) for p, v in flatten_json(t.req_json)]
        elif t.req_body:
            hay.append(("body", unquote(t.req_body)))
        return hay

    def _build(self) -> None:
        latest: dict[str, ValueRef] = {}  # value -> most recent producer before current txn
        for t in self.txns:
            hay = self._consumers(t)
            if latest:
                for value, ref in latest.items():
                    numeric = value.isdigit()
                    for loc, text in hay:
                        if value not in text:
                            continue
                        if numeric and not re.search(rf"(?<![0-9]){re.escape(value)}(?![0-9])", text):
                            continue
                        self.edges.append(Edge(ref.txn, t.index, value, ref.location, loc, ref.entropy))
                        break
            for ref in self.produced[t.index]:
                latest[ref.value] = ref

    def parents(self, idx: int) -> set[int]:
        return {e.src for e in self.edges if e.dst == idx}

    def ancestors(self, idx: int) -> set[int]:
        seen = {idx}
        q = deque([idx])
        while q:
            n = q.popleft()
            for p in self.parents(n):
                if p not in seen:
                    seen.add(p)
                    q.append(p)
        return seen

    # ------------------------------------------------------------------ target search
    def find_target(self, target: str) -> tuple[int, str]:
        tl = target.lower().lstrip("$.")
        best: tuple[float, int, str] | None = None
        for t in self.txns:
            if t.resp_json is None or t.category != "api" or not (200 <= t.status < 300):
                continue
            for p, v in flatten_json(t.resp_json):
                cp = canonical_path(p)
                k = last_key(p).lower()
                score = 0.0
                if k == tl or cp.lower().lstrip("$.") == tl:
                    score = 3.0
                elif cp.lower().endswith("." + tl):
                    score = 2.5
                elif str(v) == target:
                    score = 2.0
                elif tl in k or (k in tl and len(k) >= 4):
                    score = 1.0
                if score and (best is None or (score, -t.index) > (best[0], -best[1])):
                    best = (score, t.index, cp)
        if best is None:
            raise LookupError(f"target {target!r} not found in any JSON API response")
        return best[1], best[2]


def extract_minimal_chain(har_data: dict | list | str | Path, target_field_or_value: str) -> MinimalChain:
    """Backward slice from the response carrying ``target`` to its root seed requests."""
    txns = parse_trace(har_data)
    g = TaintGraph(txns)
    idx, path = g.find_target(target_field_or_value)
    chain = sorted(g.ancestors(idx))
    cats = Counter(t.category for t in txns)
    return MinimalChain(target_field_or_value, idx, path, chain, g.edges, txns, dict(cats))
