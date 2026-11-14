"""Deterministically generate the Web-to-MCP fixtures.

* ``skyfare.har``         – 40-request noisy browser capture (analytics, ads, static assets, CORS
                             preflights, CSRF handshake, login, decoy API calls, target flight search)
* ``flight_v1.json``      – the target API response as captured (contract source)
* ``flight_v2_drift.json``– a breaking "v2" response: 8 renamed + relocated fields, a renamed array
                             container, new distractor fields and *fresh* record values
* ``har_truth.json``      – ground truth: minimal chain entry indices + target field
* ``drift_truth.json``    – ground truth: expected v1 JSONPath -> v2 JSONPath

Run: ``python -m evals.datasets.generate_fixtures``
"""

from __future__ import annotations

import base64
import hashlib
import json
import random
from pathlib import Path

HERE = Path(__file__).parent
SEED = 1729
WWW = "https://www.skyfare.example"
API = "https://api.skyfare.example"
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_6) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.1 Safari/605.1.15"


def _rng_hex(rng: random.Random, n: int) -> str:
    return "".join(rng.choice("0123456789abcdef") for _ in range(n))


def _b64url(d: dict) -> str:
    return base64.urlsafe_b64encode(json.dumps(d, separators=(",", ":")).encode()).decode().rstrip("=")


def _jwt(rng: random.Random, sub: int) -> str:
    head = _b64url({"alg": "HS256", "typ": "JWT"})
    body = _b64url({"sub": str(sub), "iat": 1795000000, "exp": 1795003600, "jti": _rng_hex(rng, 16)})
    sig = base64.urlsafe_b64encode(hashlib.sha256(f"{head}.{body}{rng.random()}".encode()).digest()).decode().rstrip("=")
    return f"{head}.{body}.{sig}"


def _h(d: dict) -> list[dict]:
    return [{"name": k, "value": v} for k, v in d.items()]


def _entry(i: int, method: str, url: str, *, req_headers: dict | None = None, query: dict | None = None,
           post: tuple[str, str] | None = None, status: int = 200, resp_headers: dict | None = None,
           mime: str = "application/json", body: str = "", cookies: list[dict] | None = None) -> dict:
    from urllib.parse import urlencode

    full = url + (("?" + urlencode(query)) if query else "")
    rh = {"User-Agent": UA, "Accept": "*/*", "Accept-Language": "en-US,en;q=0.9"}
    rh.update(req_headers or {})
    e = {
        "startedDateTime": f"2026-11-13T09:{14 + i // 60:02d}:{i % 60:02d}.{(i * 37) % 1000:03d}Z",
        "time": 20 + (i * 13) % 180,
        "request": {
            "method": method, "url": full, "httpVersion": "HTTP/2",
            "headers": _h(rh), "queryString": _h(query or {}), "cookies": cookies or [],
            "headersSize": -1, "bodySize": len(post[1]) if post else 0,
        },
        "response": {
            "status": status, "statusText": {200: "OK", 204: "No Content", 201: "Created"}.get(status, ""),
            "httpVersion": "HTTP/2", "headers": _h({"Content-Type": mime, **(resp_headers or {})}), "cookies": [],
            "content": {"size": len(body), "mimeType": mime, "text": body}, "redirectURL": "", "headersSize": -1,
            "bodySize": len(body),
        },
        "cache": {}, "timings": {"send": 1, "wait": 15, "receive": 4},
    }
    if post:
        e["request"]["postData"] = {"mimeType": post[0], "text": post[1]}
    return e


FLIGHTS_V1 = [
    {"flight_number": "AZ610", "carrier_name": "ITA Airways", "origin": "FCO", "destination": "JFK",
     "departure_time": "2026-12-03T10:15:00Z", "arrival_time": "2026-12-03T19:05:00Z",
     "duration_minutes": 590, "price_eur": 412.5, "seats_available": 7},
    {"flight_number": "DL241", "carrier_name": "Delta Air Lines", "origin": "FCO", "destination": "JFK",
     "departure_time": "2026-12-03T12:40:00Z", "arrival_time": "2026-12-03T21:20:00Z",
     "duration_minutes": 580, "price_eur": 389.0, "seats_available": 3},
    {"flight_number": "UA915", "carrier_name": "United Airlines", "origin": "FCO", "destination": "JFK",
     "departure_time": "2026-12-03T15:05:00Z", "arrival_time": "2026-12-04T00:10:00Z",
     "duration_minutes": 605, "price_eur": 455.9, "seats_available": 12},
]

# v2: same route on a different day -> fresh values (no record overlap with v1 samples)
FLIGHTS_V2_SOURCE = [
    {"flight_number": "AZ608", "carrier_name": "ITA Airways", "origin": "FCO", "destination": "JFK",
     "departure_time": "2026-12-10T09:55:00Z", "arrival_time": "2026-12-10T18:40:00Z",
     "duration_minutes": 585, "price_eur": 398.4, "seats_available": 5, "taxes_eur": 61.2, "aircraft": "A330"},
    {"flight_number": "AA199", "carrier_name": "American Airlines", "origin": "FCO", "destination": "JFK",
     "departure_time": "2026-12-10T11:30:00Z", "arrival_time": "2026-12-10T20:25:00Z",
     "duration_minutes": 595, "price_eur": 431.7, "seats_available": 9, "taxes_eur": 58.9, "aircraft": "B772"},
]


def flight_v1(search_id: str) -> dict:
    return {"data": {"search_id": search_id, "currency": "EUR", "flights": FLIGHTS_V1}}


def flight_v2(search_id: str) -> dict:
    itins = []
    for f in FLIGHTS_V2_SOURCE:
        itins.append({
            "segment": {
                "flight_no": f["flight_number"], "airline_name": f["carrier_name"],
                "from_airport": f["origin"], "to_airport": f["destination"],
                "departs_at": f["departure_time"], "arrives_at": f["arrival_time"],
                "aircraft": f["aircraft"],
            },
            "duration_minutes": f["duration_minutes"],
            "fare": {"amount_eur": f["price_eur"], "taxes_eur": f["taxes_eur"], "cabin": "economy"},
            "availability": {"seats_left": f["seats_available"], "waitlist": False},
        })
    return {"data": {"search_id": search_id, "currency": "EUR", "itineraries": itins,
                     "tracking": {"experiment": "v2-rollout", "bucket": 7}}}


DRIFT_TRUTH = {
    "$.data.search_id": "$.data.search_id",
    "$.data.currency": "$.data.currency",
    "$.data.flights[*].flight_number": "$.data.itineraries[*].segment.flight_no",
    "$.data.flights[*].carrier_name": "$.data.itineraries[*].segment.airline_name",
    "$.data.flights[*].origin": "$.data.itineraries[*].segment.from_airport",
    "$.data.flights[*].destination": "$.data.itineraries[*].segment.to_airport",
    "$.data.flights[*].departure_time": "$.data.itineraries[*].segment.departs_at",
    "$.data.flights[*].arrival_time": "$.data.itineraries[*].segment.arrives_at",
    "$.data.flights[*].duration_minutes": "$.data.itineraries[*].duration_minutes",
    "$.data.flights[*].price_eur": "$.data.itineraries[*].fare.amount_eur",
    "$.data.flights[*].seats_available": "$.data.itineraries[*].availability.seats_left",
}


def build_har() -> tuple[dict, dict]:
    rng = random.Random(SEED)
    csrf = _rng_hex(rng, 32)
    visitor = _rng_hex(rng, 24)
    ga_cid = f"{rng.randint(10**8, 10**9)}.{1795000000 + rng.randint(0, 9999)}"
    user_id = 48213
    jwt = _jwt(rng, user_id)
    search_id = "srch_" + _rng_hex(rng, 12)
    refresh = _rng_hex(rng, 40)
    E: list[dict] = []
    add = E.append
    i = lambda: len(E)  # noqa: E731

    add(_entry(i(), "GET", f"{WWW}/", mime="text/html", body="<!doctype html><html><head><title>SkyFare</title></head><body><div id=app></div></body></html>",
               resp_headers={"Set-Cookie": f"_sf_visitor={visitor}; Path=/; Domain=www.skyfare.example; Max-Age=31536000"}))
    for name, mime in (("app.4f9c1e.js", "application/javascript"), ("vendor.91ab2d.js", "application/javascript"),
                       ("styles.7d21aa.css", "text/css"), ("logo.svg", "image/svg+xml")):
        add(_entry(i(), "GET", f"{WWW}/static/{name}", mime=mime, body="/* minified asset */" if "j" in mime or "css" in mime else "<svg/>"))
    add(_entry(i(), "GET", "https://fonts.gstatic.example/s/inter/v13/inter-latin.woff2", mime="font/woff2", body=""))
    add(_entry(i(), "GET", "https://www.googletagmanager.com/gtm.js", query={"id": "GTM-K9X2PQ"}, mime="application/javascript", body="/* gtm */"))
    add(_entry(i(), "POST", "https://www.google-analytics.com/g/collect", query={"v": "2", "tid": "G-7XQ2M1", "cid": ga_cid, "en": "page_view"},
               status=204, mime="text/plain", body=""))
    add(_entry(i(), "GET", "https://securepubads.g.doubleclick.net/gampad/ads", query={"iu": "/2231/skyfare/home", "sz": "728x90", "correlator": str(rng.randint(10**12, 10**13))},
               mime="application/javascript", body="googletag.cmd.push(function(){});"))
    # CSRF handshake (chain step 1)
    add(_entry(i(), "OPTIONS", f"{API}/v1/auth/csrf", status=204, mime="text/plain",
               resp_headers={"Access-Control-Allow-Origin": WWW, "Access-Control-Allow-Credentials": "true"}))
    idx_csrf = i()
    add(_entry(idx_csrf, "GET", f"{API}/v1/auth/csrf", req_headers={"Origin": WWW, "Accept": "application/json"},
               resp_headers={"Set-Cookie": f"sf_csrf={csrf}; Path=/; Secure; SameSite=Strict"},
               body=json.dumps({"csrf_token": csrf, "expires_in": 7200})))
    add(_entry(i(), "GET", f"{API}/v1/config/public", req_headers={"Origin": WWW},
               body=json.dumps({"build": "2026.11.3", "features": {"price_alerts": True, "v2_search": False},
                                "request_id": _rng_hex(rng, 20)})))
    add(_entry(i(), "POST", "https://api.segment.example/v1/t", post=("application/json", json.dumps({
        "anonymousId": visitor, "event": "Login Viewed", "properties": {"path": "/login"}})), mime="application/json",
        body=json.dumps({"success": True})))
    add(_entry(i(), "OPTIONS", f"{API}/v1/auth/session", status=204, mime="text/plain",
               resp_headers={"Access-Control-Allow-Origin": WWW, "Access-Control-Allow-Headers": "content-type,x-csrf-token"}))
    # Login (chain step 2)
    idx_login = i()
    add(_entry(idx_login, "POST", f"{API}/v1/auth/session",
               req_headers={"Origin": WWW, "Content-Type": "application/json", "X-CSRF-Token": csrf, "Cookie": f"sf_csrf={csrf}"},
               post=("application/json", json.dumps({"email": "traveller@example.com", "password": "correct-horse-battery-9"})),
               status=201, body=json.dumps({"session": {"access_token": jwt, "token_type": "Bearer", "expires_in": 3600,
                                                        "refresh_token": refresh, "user_id": user_id}})))
    add(_entry(i(), "GET", "https://script.hotjar.example/modules.8d1f.js", mime="application/javascript", body="/* hj */"))
    # decoy API depending on the session but not on the path to the target
    add(_entry(i(), "GET", f"{API}/v1/users/{user_id}/preferences", req_headers={"Origin": WWW, "Authorization": f"Bearer {jwt}"},
               body=json.dumps({"home_airport": "FCO", "currency": "EUR", "newsletter": False})))
    # decoy: contains the travel date + airport codes (low-entropy values, must NOT create edges)
    add(_entry(i(), "GET", f"{API}/v1/promotions", query={"market": "IT"}, req_headers={"Origin": WWW},
               body=json.dumps({"promos": [{"code": "WINTER26", "route": "FCO-JFK", "valid_until": "2026-12-03", "discount_pct": 10}]})))
    add(_entry(i(), "GET", f"{WWW}/static/img/hero-nyc.webp", mime="image/webp", body=""))
    add(_entry(i(), "OPTIONS", f"{API}/v1/flights/search", status=204, mime="text/plain",
               resp_headers={"Access-Control-Allow-Origin": WWW, "Access-Control-Allow-Headers": "authorization,x-csrf-token"}))
    # Target (chain step 3)
    idx_target = i()
    add(_entry(idx_target, "GET", f"{API}/v1/flights/search",
               query={"origin": "FCO", "destination": "JFK", "date": "2026-12-03", "adults": "1", "cabin": "economy"},
               req_headers={"Origin": WWW, "Accept": "application/json", "Authorization": f"Bearer {jwt}", "X-CSRF-Token": csrf,
                            "Cookie": f"sf_csrf={csrf}"},
               body=json.dumps(flight_v1(search_id))))
    # descendants of the target (must be pruned: they consume, not produce, the target)
    add(_entry(i(), "POST", f"{API}/v1/telemetry/events", req_headers={"Origin": WWW, "X-CSRF-Token": csrf, "Content-Type": "application/json"},
               post=("application/json", json.dumps({"event": "search_results", "search_id": search_id, "count": 3})), status=204, body=""))
    add(_entry(i(), "GET", f"{API}/v1/flights/AZ610/seatmap", query={"search_id": search_id},
               req_headers={"Origin": WWW, "Authorization": f"Bearer {jwt}"},
               body=json.dumps({"rows": 32, "layout": "3-3-3", "available": ["12A", "12B", "30F"]})))
    # remaining noise
    noise = [
        ("POST", "https://www.google-analytics.com/g/collect", {"v": "2", "tid": "G-7XQ2M1", "cid": ga_cid, "en": "search"}, "text/plain", ""),
        ("GET", "https://securepubads.g.doubleclick.net/pcs/view", {"xai": _rng_hex(rng, 30), "sig": _rng_hex(rng, 24)}, "image/gif", ""),
        ("GET", f"{WWW}/static/img/airline-az.png", None, "image/png", ""),
        ("GET", f"{WWW}/static/img/airline-dl.png", None, "image/png", ""),
        ("GET", f"{WWW}/static/img/airline-ua.png", None, "image/png", ""),
        ("POST", "https://o44721.ingest.sentry.example/api/5512/envelope/", {"sentry_key": _rng_hex(rng, 32)}, "application/json", "{}"),
        ("GET", f"{API}/v1/recommendations", {"near": "FCO"}, "application/json", json.dumps({"items": [{"city": "Lisbon", "from_eur": 89}]})),
        ("GET", f"{API}/v1/notifications/poll", {"since": "1795000100"}, "application/json", json.dumps({"items": []})),
        ("GET", "https://www.facebook.example/tr", {"id": "8810021", "ev": "Search"}, "image/gif", ""),
        ("POST", "https://api.segment.example/v1/t", None, "application/json", json.dumps({"success": True})),
        ("GET", "https://fonts.gstatic.example/s/inter/v13/inter-bold.woff2", None, "font/woff2", ""),
        ("GET", f"{WWW}/static/chunk.search.a81c.js", None, "application/javascript", "/* chunk */"),
        ("GET", f"{WWW}/manifest.webmanifest", None, "application/manifest+json", json.dumps({"name": "SkyFare"})),
        ("GET", f"{WWW}/favicon.ico", None, "image/x-icon", ""),
        ("POST", "https://www.google-analytics.com/g/collect", {"v": "2", "tid": "G-7XQ2M1", "cid": ga_cid, "en": "scroll"}, "text/plain", ""),
        ("GET", "https://securepubads.g.doubleclick.net/gampad/ads", {"iu": "/2231/skyfare/results", "sz": "300x250"}, "application/javascript", ""),
        ("GET", f"{API}/v1/currency/rates", {"base": "EUR"}, "application/json", json.dumps({"USD": 1.08, "GBP": 0.86})),
    ]
    for method, url, q, mime, body in noise:
        if len(E) >= 40:
            break
        hdr = {"Origin": WWW, "Authorization": f"Bearer {jwt}"} if "/v1/recommendations" in url else {}
        post = ("application/json", json.dumps({"anonymousId": visitor, "event": "Search"})) if method == "POST" and "segment" in url else None
        add(_entry(i(), method, url, query=q, mime=mime, body=body, req_headers=hdr, post=post,
                   status=204 if method == "POST" and not body else 200))
    assert len(E) == 40, len(E)
    har = {"log": {"version": "1.2", "creator": {"name": "WebInspector", "version": "537.36"},
                   "pages": [{"startedDateTime": E[0]["startedDateTime"], "id": "page_1", "title": WWW + "/", "pageTimings": {}}],
                   "entries": E}}
    truth = {"target_field": "price_eur", "minimal_chain": [idx_csrf, idx_login, idx_target], "n_entries": len(E),
             "secret_inputs": ["password"], "search_id": search_id}
    return har, truth


def main() -> None:
    har, truth = build_har()
    (HERE / "skyfare.har").write_text(json.dumps(har, indent=1))
    (HERE / "har_truth.json").write_text(json.dumps(truth, indent=2))
    (HERE / "flight_v1.json").write_text(json.dumps(flight_v1(truth["search_id"]), indent=2))
    (HERE / "flight_v2_drift.json").write_text(json.dumps(flight_v2(truth["search_id"]), indent=2))
    (HERE / "drift_truth.json").write_text(json.dumps(DRIFT_TRUTH, indent=2))
    print(f"wrote {HERE}/skyfare.har ({len(har['log']['entries'])} entries), chain={truth['minimal_chain']}")


if __name__ == "__main__":
    main()
