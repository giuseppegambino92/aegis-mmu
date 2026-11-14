"""A live re-implementation of the captured SkyFare API used for end-to-end tool replay.

Every call issues *fresh* CSRF and session tokens and enforces them, so a synthesised tool only
passes if it really extracts and threads dynamic values at runtime (replaying captured tokens
fails). ``drift=True`` serves the breaking v2 response schema.
"""

from __future__ import annotations

import secrets

from fastapi import FastAPI, Header, HTTPException, Request, Response

from .generate_fixtures import flight_v1, flight_v2


def create_app(password: str = "correct-horse-battery-9", drift: bool = False) -> FastAPI:
    app = FastAPI(title="skyfare-replay")
    csrf_tokens: set[str] = set()
    sessions: dict[str, int] = {}
    app.state.drift = drift

    @app.get("/v1/auth/csrf")
    def csrf(response: Response) -> dict:
        tok = secrets.token_hex(16)
        csrf_tokens.add(tok)
        response.set_cookie("sf_csrf", tok, secure=True, samesite="strict")
        return {"csrf_token": tok, "expires_in": 7200}

    @app.post("/v1/auth/session", status_code=201)
    async def session(request: Request, x_csrf_token: str = Header(default="")) -> dict:
        if x_csrf_token not in csrf_tokens or request.cookies.get("sf_csrf") != x_csrf_token:
            raise HTTPException(403, "csrf check failed")
        body = await request.json()
        if body.get("password") != password or "@" not in str(body.get("email", "")):
            raise HTTPException(401, "bad credentials")
        tok = "sk_" + secrets.token_urlsafe(32)
        sessions[tok] = 48213
        return {"session": {"access_token": tok, "token_type": "Bearer", "expires_in": 3600,
                            "refresh_token": secrets.token_hex(20), "user_id": 48213}}

    @app.get("/v1/flights/search")
    def search(origin: str, destination: str, date: str, adults: int = 1, cabin: str = "economy",
               authorization: str = Header(default=""), x_csrf_token: str = Header(default="")) -> dict:
        if not authorization.startswith("Bearer ") or authorization[7:] not in sessions:
            raise HTTPException(401, "invalid session")
        if x_csrf_token not in csrf_tokens:
            raise HTTPException(403, "csrf check failed")
        sid = "srch_" + secrets.token_hex(6)
        return flight_v2(sid) if app.state.drift else flight_v1(sid)

    return app
