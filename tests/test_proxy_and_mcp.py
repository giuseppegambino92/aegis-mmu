import io
import json

import httpx
import pytest
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.testclient import TestClient

from aegis_mmu.demo import agent_session_requests
from aegis_mmu.interfaces.mcp_server import MCPServer
from aegis_mmu.interfaces.proxy import create_app
from evals.datasets.skyfare_server import create_app as skyfare_app

from .conftest import DATASETS


def _upstream():
    """A real in-process upstream that records what the proxy forwarded."""
    app = FastAPI()
    app.state.seen = []

    @app.post("/v1/messages")
    async def messages(request: Request):
        body = await request.json()
        app.state.seen.append({"headers": dict(request.headers), "body": body})
        if body.get("stream"):
            async def sse():
                yield b'event: message_start\ndata: {"type":"message_start"}\n\n'
                yield b'event: message_stop\ndata: {"type":"message_stop"}\n\n'
            return StreamingResponse(sse(), media_type="text/event-stream")
        return JSONResponse({"id": "msg_up", "type": "message", "role": "assistant",
                             "content": [{"type": "text", "text": "upstream says hi"}], "usage": {"input_tokens": 1, "output_tokens": 1}})

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        body = await request.json()
        app.state.seen.append({"headers": dict(request.headers), "body": body})
        return JSONResponse({"id": "c1", "object": "chat.completion", "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}}]})

    @app.post("/v1/messages/count_tokens")
    async def count(request: Request):
        return JSONResponse({"input_tokens": 42})

    return app


@pytest.fixture()
def proxy(mmu):
    up = _upstream()
    app = create_app(mmu, upstream_transport=httpx.ASGITransport(app=up))
    with TestClient(app) as c:
        yield c, up


@pytest.fixture(scope="module")
def session_reqs(codebase):
    return agent_session_requests(json.loads((DATASETS / "session.json").read_text())["turns"], dict(codebase))


def test_fold_local_anthropic(proxy, session_reqs):
    c, up = proxy
    r = c.post("/v1/messages", json=session_reqs[0], headers={"X-Aegis-Mode": "fold-local", "X-Aegis-Session": "p1"})
    assert r.status_code == 200 and not up.state.seen
    h = r.headers
    for k in ("x-aegis-tokens-original", "x-aegis-tokens-compressed", "x-aegis-savings-pct", "x-aegis-kv-hit-rate", "x-aegis-latency-ms"):
        assert k in h
    assert int(h["x-aegis-tokens-compressed"]) < int(h["x-aegis-tokens-original"])
    body = r.json()
    assert body["type"] == "message" and body["aegis"]["folded_blocks"] == 1
    assert "[L3 " in json.dumps(body["aegis"]["rewritten_request"])
    r2 = c.post("/v1/messages", json=session_reqs[1], headers={"X-Aegis-Mode": "fold-local", "X-Aegis-Session": "p1"})
    assert float(r2.headers["x-aegis-kv-hit-rate"]) > 0.3


def test_upstream_forward_sends_folded_body_and_auth(proxy, session_reqs):
    c, up = proxy
    r = c.post("/v1/messages", json=session_reqs[0], headers={"X-Aegis-Mode": "upstream-forward", "x-api-key": "sk-test",
                                                              "anthropic-version": "2023-06-01"})
    assert r.status_code == 200 and r.json()["content"][0]["text"] == "upstream says hi"
    seen = up.state.seen[-1]
    assert seen["headers"]["x-api-key"] == "sk-test" and "x-aegis-mode" not in seen["headers"]
    fwd = json.dumps(seen["body"])
    assert "[L3 " in fwd and len(fwd) < len(json.dumps(session_reqs[0]))
    assert float(r.headers["x-aegis-savings-pct"]) > 0


def test_streaming_passthrough(proxy, session_reqs):
    c, _ = proxy
    req = dict(session_reqs[0], stream=True)
    with c.stream("POST", "/v1/messages", json=req, headers={"X-Aegis-Mode": "upstream-forward"}) as r:
        data = b"".join(r.iter_raw())
    assert r.headers["content-type"].startswith("text/event-stream")
    assert b"message_start" in data and b"message_stop" in data


def test_passthrough_mode_is_byte_identical(proxy, session_reqs):
    c, up = proxy
    r = c.post("/v1/messages", json=session_reqs[0], headers={"X-Aegis-Mode": "passthrough"})
    assert r.status_code == 200 and r.headers["x-aegis-savings-pct"] == "0.00"
    assert up.state.seen[-1]["body"] == session_reqs[0]


def test_openai_and_catch_all_and_stats(proxy, codebase):
    c, up = proxy
    path, src = next((p, s) for p, s in codebase if p.endswith("tokens.py"))
    body = {"model": "gpt", "messages": [{"role": "user", "content": f"verify() rejects fresh tokens\n```python {path}\n{src}\n```"}]}
    r = c.post("/v1/chat/completions", json=body, headers={"X-Aegis-Mode": "fold-local"})
    assert r.json()["object"] == "chat.completion" and r.json()["aegis"]["folded_blocks"] == 1
    r = c.post("/v1/chat/completions", json=body)
    assert r.json()["id"] == "c1"
    assert len(up.state.seen[-1]["body"]["messages"][0]["content"]) < len(body["messages"][0]["content"])
    assert c.post("/v1/messages/count_tokens", json={"messages": []}, headers={"anthropic-version": "2023-06-01"}).json() == {"input_tokens": 42}
    s = c.get("/v1/aegis/stats").json()
    assert s["requests"] >= 2 and s["tokens_saved"] > 0 and s["engine"].startswith("fast-local")
    f = c.post("/v1/aegis/fold", json={"query": "token expiry", "files": [{"path": path, "content": src}], "budget": 300}).json()
    assert f["tokens_folded"] < f["tokens_original"]


def _rpc(srv, method, params=None, mid=1):
    return srv.handle({"jsonrpc": "2.0", "id": mid, "method": method, "params": params or {}})


def test_mcp_protocol_and_core_tools(mmu, codebase):
    srv = MCPServer(mmu)
    init = _rpc(srv, "initialize", {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "t", "version": "1"}})
    assert init["result"]["protocolVersion"] == "2025-06-18" and init["result"]["serverInfo"]["name"] == "aegis-mmu"
    assert srv.handle({"jsonrpc": "2.0", "method": "notifications/initialized"}) is None
    names = [t["name"] for t in _rpc(srv, "tools/list")["result"]["tools"]]
    assert names == ["aegis_fold_context", "aegis_expand_node", "aegis_record_failure", "aegis_harvest_har", "aegis_heal_payload"]
    files = [{"path": p, "content": s} for p, s in codebase]
    res = _rpc(srv, "tools/call", {"name": "aegis_fold_context", "arguments": {"query": "ledger balance cutoff", "files": files}})["result"]
    assert not res["isError"] and res["structuredContent"]["savings_pct"] > 70 and "<aegis_file" in res["content"][0]["text"]
    vid = res["structuredContent"]["hits"][0]["vec_id"]
    exp = _rpc(srv, "tools/call", {"name": "aegis_expand_node", "arguments": {"vec_id": vid}})["result"]
    assert exp["structuredContent"]["vec_id"] == vid and "def " in exp["content"][0]["text"]
    paged = _rpc(srv, "tools/call", {"name": "aegis_fold_context", "arguments": {"query": "ledger balance cutoff", "files": files[:4], "session_id": "m1"}})
    assert paged["result"]["content"][0]["text"].startswith('<aegis_context epoch="0">')
    rec = _rpc(srv, "tools/call", {"name": "aegis_record_failure", "arguments": {"task": "use datetime.utcnow() for ledger stamps", "error": "TypeError: can't compare offset-naive and offset-aware datetimes"}})
    assert rec["result"]["structuredContent"]["id"] == 1
    assert _rpc(srv, "nope")["error"]["code"] == -32601
    bad = _rpc(srv, "tools/call", {"name": "aegis_expand_node", "arguments": {"vec_id": "zzz"}})
    assert bad["error"]["code"] == -32602


def test_mcp_harvest_heal_and_dynamic_tool(mmu):
    srv = MCPServer(mmu, http_client_factory=lambda: httpx.AsyncClient(transport=httpx.ASGITransport(app=skyfare_app(drift=True))))
    out = _rpc(srv, "tools/call", {"name": "aegis_harvest_har", "arguments": {"har_path": str(DATASETS / "skyfare.har"), "target": "price_eur", "tool_name": "skyfare_flight_search"}})["result"]
    assert [s["index"] for s in out["structuredContent"]["chain"]] == [10, 14, 20]
    tools = {t["name"]: t for t in _rpc(srv, "tools/list")["result"]["tools"]}
    assert "skyfare_flight_search" in tools and "password" in tools["skyfare_flight_search"]["inputSchema"]["properties"]
    healed = _rpc(srv, "tools/call", {"name": "aegis_heal_payload", "arguments": {"tool_name": "skyfare_flight_search", "payload": json.loads((DATASETS / "flight_v2_drift.json").read_text())}})["result"]
    assert healed["structuredContent"]["report"]["healed"] and healed["structuredContent"]["healed_payload"]["data"]["flights"]
    live = _rpc(srv, "tools/call", {"name": "skyfare_flight_search", "arguments": {"password": "correct-horse-battery-9"}})["result"]
    assert not live["isError"] and live["structuredContent"]["data"]["data"]["flights"][0]["price_eur"] == 398.4
    err = _rpc(srv, "tools/call", {"name": "skyfare_flight_search", "arguments": {"password": "wrong"}})["result"]
    assert err["isError"] and "401" in err["content"][0]["text"]


def test_mcp_stdio_transport(mmu):
    srv = MCPServer(mmu)
    stdin = io.StringIO("\n".join([
        json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2024-11-05"}}),
        json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}),
        "not json",
        json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/list"}),
        json.dumps({"jsonrpc": "2.0", "id": 3, "method": "ping"}),
    ]) + "\n")
    stdout = io.StringIO()
    srv.serve_stdio(stdin, stdout)
    lines = [json.loads(x) for x in stdout.getvalue().splitlines()]
    assert [l.get("id") for l in lines] == [1, None, 2, 3]
    assert lines[0]["result"]["protocolVersion"] == "2024-11-05" and lines[1]["error"]["code"] == -32700
    assert len(lines[2]["result"]["tools"]) == 5 and lines[3]["result"] == {}
