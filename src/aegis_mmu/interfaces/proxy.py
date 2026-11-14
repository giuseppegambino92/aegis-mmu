"""Transparent Anthropic/OpenAI-compatible reverse proxy with in-transit semantic folding.

Endpoints
---------
* ``POST /v1/messages``          Anthropic Messages API (Claude Code: ``ANTHROPIC_BASE_URL=http://127.0.0.1:8787``)
* ``POST /v1/chat/completions``  OpenAI Chat Completions (Cursor / OpenAI-compatible clients, Ollama)
* ``GET  /v1/aegis/stats``       aggregate telemetry
* ``POST /v1/aegis/fold``        direct folding API (files + query)
* anything else                  forwarded verbatim to the upstream

Modes (``X-Aegis-Mode`` header, default from settings):
* ``upstream-forward`` – fold, then forward to the real API (streaming supported),
* ``passthrough``      – forward unmodified (telemetry only),
* ``fold-local``       – fold and answer locally with a synthetic message carrying the rewritten
                         request (deterministic; no network) – used by tests and dry runs.

Every response carries ``X-Aegis-Tokens-Original``, ``X-Aegis-Tokens-Compressed``,
``X-Aegis-Savings-Pct``, ``X-Aegis-KV-Hit-Rate`` and ``X-Aegis-Latency-Ms``.
"""

from __future__ import annotations

import hashlib
import json
import time
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, Response, StreamingResponse

from .. import __version__
from ..config import AegisSettings, get_settings
from ..mmu.pipeline import RewriteResult, SemanticMMU
from ..mmu.prompt_codec import prompt_text
from ..tokens import count_tokens

HOP_HEADERS = {"host", "content-length", "connection", "keep-alive", "transfer-encoding", "accept-encoding", "te",
               "upgrade", "proxy-authorization", "proxy-connection"}
RESP_DROP = {"content-length", "content-encoding", "transfer-encoding", "connection", "keep-alive"}


def _fwd_headers(request: Request) -> dict[str, str]:
    return {k: v for k, v in request.headers.items() if k.lower() not in HOP_HEADERS and not k.lower().startswith("x-aegis-")}


def create_app(mmu: SemanticMMU | None = None, settings: AegisSettings | None = None,
               upstream_transport: httpx.AsyncBaseTransport | None = None,
               allow_upstream_override: bool = False, tune_gc_on_start: bool = False) -> FastAPI:
    settings = settings or (mmu.settings if mmu else get_settings())

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        await app.state.client.aclose()

    app = FastAPI(title="aegis-mmu proxy", version=__version__, lifespan=lifespan)
    app.state.mmu = mmu or SemanticMMU(settings=settings)
    app.state.settings = settings
    app.state.client = httpx.AsyncClient(transport=upstream_transport, timeout=settings.upstream_timeout_s)
    if tune_gc_on_start:
        from ..runtime import tune_gc

        tune_gc()

    def _mode(request: Request) -> str:
        m = request.headers.get("x-aegis-mode", settings.proxy_mode).strip().lower()
        if m not in ("fold-local", "passthrough", "upstream-forward"):
            m = settings.proxy_mode
        return m

    def _upstream(request: Request, default: str) -> str:
        override = request.headers.get("x-aegis-upstream")
        if override and allow_upstream_override:
            return override.rstrip("/")
        return default.rstrip("/")

    async def _forward(request: Request, base: str, path: str, body: bytes, extra: dict[str, str]) -> Response:
        url = f"{base}/{path.lstrip('/')}"
        if request.url.query:
            url += "?" + request.url.query
        headers = _fwd_headers(request)
        stream = False
        try:
            stream = bool(json.loads(body or b"{}").get("stream"))
        except (ValueError, AttributeError):
            stream = False
        req = app.state.client.build_request(request.method, url, headers=headers, content=body)
        upstream = await app.state.client.send(req, stream=True)
        out_headers = {k: v for k, v in upstream.headers.items() if k.lower() not in RESP_DROP}
        out_headers.update(extra)
        if stream or upstream.headers.get("content-type", "").startswith("text/event-stream"):
            async def gen():
                try:
                    async for chunk in upstream.aiter_raw():
                        yield chunk
                finally:
                    await upstream.aclose()
            return StreamingResponse(gen(), status_code=upstream.status_code, headers=out_headers,
                                     media_type=upstream.headers.get("content-type"))
        content = await upstream.aread()
        await upstream.aclose()
        return Response(content=content, status_code=upstream.status_code, headers=out_headers,
                        media_type=upstream.headers.get("content-type"))

    def _summary(rr: RewriteResult) -> str:
        return (f"[aegis-mmu fold-local] session={rr.session_id} folded_blocks={rr.folded_blocks} "
                f"tokens {rr.tokens_original}->{rr.tokens_rewritten} ({rr.savings_pct:.1f}% saved), "
                f"kv_prefix_hit={rr.kv_hit_rate:.3f}, promotions={len(rr.promotions)}, warnings={len(rr.warnings)}, "
                f"latency={rr.latency_ms:.2f}ms")

    def _telemetry(rr: RewriteResult) -> dict[str, Any]:
        return {"session_id": rr.session_id, "tokens_original": rr.tokens_original, "tokens_compressed": rr.tokens_rewritten,
                "savings_pct": round(rr.savings_pct, 2), "kv_hit_rate": round(rr.kv_hit_rate, 4),
                "latency_ms": round(rr.latency_ms, 3), "folded_blocks": rr.folded_blocks,
                "promotions": rr.promotions, "warnings": rr.warnings, "session_reset": rr.reset}

    async def _handle(request: Request, fmt: str, path: str, base_default: str) -> Response:
        raw = await request.body()
        mode = _mode(request)
        try:
            body = json.loads(raw or b"{}")
        except ValueError:
            return JSONResponse({"type": "error", "error": {"type": "invalid_request_error", "message": "body is not JSON"}}, 400)
        if mode == "passthrough":
            t0 = time.perf_counter()
            n = count_tokens(prompt_text(body, fmt))
            lat = (time.perf_counter() - t0) * 1e3
            app.state.mmu.store.log_event("passthrough", tokens_original=n, tokens_compressed=n, latency_ms=lat)
            hdr = {"X-Aegis-Tokens-Original": str(n), "X-Aegis-Tokens-Compressed": str(n), "X-Aegis-Savings-Pct": "0.00",
                   "X-Aegis-KV-Hit-Rate": "0.0000", "X-Aegis-Latency-Ms": f"{lat:.3f}", "X-Aegis-Mode": mode}
            return await _forward(request, _upstream(request, base_default), path, raw, hdr)

        sid = request.headers.get("x-aegis-session")
        rr = await run_in_threadpool(app.state.mmu.rewrite_request, body, fmt, sid)
        hdr = rr.headers() | {"X-Aegis-Mode": mode}
        if mode == "fold-local":
            msg_id = "aegis_" + hashlib.blake2b(json.dumps(rr.body, sort_keys=True).encode(), digest_size=8).hexdigest()
            telemetry = _telemetry(rr)
            if fmt == "anthropic":
                payload = {"id": "msg_" + msg_id, "type": "message", "role": "assistant", "model": body.get("model", "aegis-mmu"),
                           "content": [{"type": "text", "text": _summary(rr)}], "stop_reason": "end_turn",
                           "stop_sequence": None, "usage": {"input_tokens": rr.tokens_rewritten, "output_tokens": 0},
                           "aegis": telemetry | {"rewritten_request": rr.body}}
            else:
                payload = {"id": "chatcmpl-" + msg_id, "object": "chat.completion", "created": int(time.time()),
                           "model": body.get("model", "aegis-mmu"),
                           "choices": [{"index": 0, "message": {"role": "assistant", "content": _summary(rr)}, "finish_reason": "stop"}],
                           "usage": {"prompt_tokens": rr.tokens_rewritten, "completion_tokens": 0, "total_tokens": rr.tokens_rewritten},
                           "aegis": telemetry | {"rewritten_request": rr.body}}
            return JSONResponse(payload, headers=hdr)
        out = json.dumps(rr.body, ensure_ascii=False).encode("utf-8")
        return await _forward(request, _upstream(request, base_default), path, out, hdr)

    @app.post("/v1/messages")
    async def messages(request: Request) -> Response:
        return await _handle(request, "anthropic", "v1/messages", settings.upstream_anthropic)

    @app.post("/v1/chat/completions")
    async def chat(request: Request) -> Response:
        return await _handle(request, "openai", "v1/chat/completions", settings.upstream_openai)

    @app.get("/v1/aegis/stats")
    async def stats() -> dict:
        s = await run_in_threadpool(app.state.mmu.store.stats)
        return s | {"version": __version__, "engine": app.state.mmu.engine.name, "mode": settings.proxy_mode}

    @app.post("/v1/aegis/fold")
    async def fold(payload: dict) -> dict:
        files = [(f["path"], f["content"], f.get("language")) for f in payload.get("files", [])]
        res = await run_in_threadpool(app.state.mmu.fold, files, payload.get("query", ""), payload.get("budget"))
        return res.to_dict()

    @app.get("/healthz")
    async def health() -> dict:
        return {"status": "ok", "version": __version__}

    @app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"])
    async def catch_all(path: str, request: Request) -> Response:
        is_anthropic = "anthropic-version" in request.headers or "x-api-key" in request.headers or path.startswith("v1/messages")
        base = settings.upstream_anthropic if is_anthropic else settings.upstream_openai
        return await _forward(request, _upstream(request, base), path, await request.body(), {"X-Aegis-Mode": "passthrough"})

    return app
