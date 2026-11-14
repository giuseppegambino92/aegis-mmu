"""Model Context Protocol server (JSON-RPC 2.0 over stdio, newline-delimited; also callable in-process).

Core tools: ``aegis_fold_context``, ``aegis_expand_node``, ``aegis_record_failure``,
``aegis_harvest_har``, ``aegis_heal_payload``. Every tool harvested from a HAR is additionally
registered as its own MCP tool (zero-token deterministic execution with self-healing output).
"""

from __future__ import annotations

import asyncio
import json
import sys
import traceback
from pathlib import Path
from typing import Any, Callable, TextIO

from .. import __version__
from ..config import AegisSettings, get_settings
from ..harvester.synthesizer import ToolRuntime, ToolSpec, input_json_schema, synthesize
from ..harvester.taint_graph import extract_minimal_chain
from ..mmu.pipeline import SemanticMMU

SUPPORTED_PROTOCOLS = ("2025-06-18", "2025-03-26", "2024-11-05")
JSONRPC_PARSE_ERROR, JSONRPC_INVALID_REQUEST, JSONRPC_METHOD_NOT_FOUND, JSONRPC_INVALID_PARAMS, JSONRPC_INTERNAL = (
    -32700, -32600, -32601, -32602, -32603)

CORE_TOOLS: list[dict[str, Any]] = [
    {
        "name": "aegis_fold_context",
        "title": "Holographic context folding",
        "description": "Fold source files into a token-budgeted, multi-resolution (L0 full / L1 control-flow skeleton / "
                       "L2 signatures / L3 one-line) context for a task. Relevant symbols stay verbatim; everything "
                       "else is folded with expandable vec_ids. With session_id, pages are appended KV-cache-stably.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The task / bug report driving relevance."},
                "files": {"type": "array", "description": "Files to fold.", "items": {
                    "type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"},
                                                     "language": {"type": "string"}}, "required": ["path", "content"]}},
                "paths": {"type": "array", "items": {"type": "string"}, "description": "Local file paths to read instead of inline content."},
                "budget": {"type": "integer", "description": "Target token budget (default: adaptive)."},
                "session_id": {"type": "string", "description": "Enable append-only KV-stable paging for this session."},
            },
            "required": ["query"],
        },
    },
    {
        "name": "aegis_expand_node",
        "title": "Expand folded node",
        "description": "Return the full L0 source of a folded unit (or whole file) by its vec_id.",
        "inputSchema": {"type": "object", "properties": {"vec_id": {"type": "string"}}, "required": ["vec_id"]},
    },
    {
        "name": "aegis_record_failure",
        "title": "Record failed attempt (Negative RAG)",
        "description": "Store a failed approach (error, rolled-back diff, rejected review) so future similar tasks get "
                       "an <aegis_negative_memory_warning>.",
        "inputSchema": {"type": "object", "properties": {
            "task": {"type": "string"}, "code": {"type": "string"}, "error": {"type": "string"},
            "why": {"type": "string"}, "remedy": {"type": "string"}, "title": {"type": "string"},
            "kind": {"type": "string", "enum": ["exception", "rollback", "diff", "review", "note"]}},
            "required": ["task"]},
    },
    {
        "name": "aegis_harvest_har",
        "title": "Compile HAR into an MCP tool",
        "description": "Backward-taint-slice a HAR/HTTP trace to the minimal request chain producing `target`, "
                       "synthesise a deterministic tool + shadow contract and register it as an MCP tool.",
        "inputSchema": {"type": "object", "properties": {
            "har_path": {"type": "string"}, "har": {"type": "object"}, "target": {"type": "string"},
            "tool_name": {"type": "string"}, "write_client": {"type": "string", "description": "Optional path for the generated Python client."}},
            "required": ["target", "tool_name"]},
    },
    {
        "name": "aegis_heal_payload",
        "title": "Validate / heal API payload",
        "description": "Validate a JSON payload against a harvested tool's shadow contract; on schema drift remap it "
                       "with Hungarian bipartite matching and return the healed payload + drift report.",
        "inputSchema": {"type": "object", "properties": {"tool_name": {"type": "string"}, "payload": {}},
                        "required": ["tool_name", "payload"]},
    },
]


class MCPError(Exception):
    def __init__(self, code: int, message: str):
        super().__init__(message)
        self.code = code


class MCPServer:
    def __init__(self, mmu: SemanticMMU | None = None, settings: AegisSettings | None = None,
                 http_client_factory: Callable[[], Any] | None = None):
        self.settings = settings or (mmu.settings if mmu else get_settings())
        self.mmu = mmu or SemanticMMU(settings=self.settings)
        self.runtime = ToolRuntime(self.mmu.store)
        self.http_client_factory = http_client_factory
        self.initialized = False
        self._notify: Callable[[dict], None] | None = None

    # ------------------------------------------------------------------ tools
    def list_tools(self) -> list[dict]:
        tools = [dict(t) for t in CORE_TOOLS]
        for c in self.mmu.store.list_contracts():
            spec = ToolSpec(**c["spec"])
            tools.append({"name": spec.name, "title": f"Harvested: {spec.name}", "description": spec.description,
                          "inputSchema": input_json_schema(spec)})
        return tools

    def call_tool(self, name: str, args: dict) -> dict:
        handler = getattr(self, f"_tool_{name}", None)
        if handler is not None:
            structured, text = handler(args or {})
        elif self.mmu.store.get_contract(name) is not None:
            structured, text = self._run_harvested(name, args or {})
        else:
            raise MCPError(JSONRPC_INVALID_PARAMS, f"unknown tool: {name}")
        return {"content": [{"type": "text", "text": text}], "structuredContent": structured, "isError": False}

    def _tool_aegis_fold_context(self, a: dict) -> tuple[dict, str]:
        files = [(f["path"], f["content"], f.get("language")) for f in a.get("files") or []]
        for p in a.get("paths") or []:
            files.append((p, Path(p).read_text(encoding="utf-8"), None))
        if not files:
            raise MCPError(JSONRPC_INVALID_PARAMS, "provide `files` or `paths`")
        if a.get("session_id"):
            pr = self.mmu.context_block(a["session_id"], [(f[0], f[1]) for f in files], a["query"], a.get("budget"))
            orig = sum(self.mmu.ingest(f[0], f[1]).tokens_source for f in files)
            structured = {"session_id": a["session_id"], "epoch": pr.epoch, "turn": pr.turn, "kv_hit_rate": round(pr.kv_hit_rate, 4),
                          "appended": pr.appended, "promoted": pr.promoted, "compacted": pr.compacted,
                          "tokens_original": orig, "tokens_context": pr.resident_tokens}
            return structured, pr.context
        res = self.mmu.fold(files, a["query"], a.get("budget"))
        d = res.to_dict()
        context = d.pop("context")
        return d, context

    def _tool_aegis_expand_node(self, a: dict) -> tuple[dict, str]:
        node = self.mmu.expand(a["vec_id"])
        if node is None:
            raise MCPError(JSONRPC_INVALID_PARAMS, f"unknown vec_id {a['vec_id']}")
        return {k: v for k, v in node.items() if k != "source"}, node["source"]

    def _tool_aegis_record_failure(self, a: dict) -> tuple[dict, str]:
        fid = self.mmu.negative.record(task=a["task"], code=a.get("code", ""), error=a.get("error", ""),
                                       kind=a.get("kind", "exception"), title=a.get("title"), why=a.get("why"),
                                       remedy=a.get("remedy"))
        return {"id": fid}, f"Recorded failure #{fid} in Negative RAG memory."

    def _tool_aegis_harvest_har(self, a: dict) -> tuple[dict, str]:
        src = a.get("har") if a.get("har") is not None else a.get("har_path")
        if src is None:
            raise MCPError(JSONRPC_INVALID_PARAMS, "provide `har` or `har_path`")
        chain = extract_minimal_chain(src, a["target"])
        tool = synthesize(chain, a["tool_name"], self.mmu.engine)
        self.runtime.register(tool)
        if a.get("write_client"):
            Path(a["write_client"]).write_text(tool.code)
        structured = chain.to_dict() | {"tool_name": tool.spec.name, "inputs": [i.model_dump() for i in tool.spec.inputs],
                                        "shadow_fields": len(tool.shadow.fields), "client_path": a.get("write_client")}
        if self._notify:
            self._notify({"jsonrpc": "2.0", "method": "notifications/tools/list_changed"})
        steps = " -> ".join(f"#{s['index']} {s['method']} {s['url']}" for s in structured["chain"])
        return structured, (f"Registered tool `{tool.spec.name}`: {len(chain.chain)}-step chain ({steps}); "
                            f"pruned {len(chain.pruned)}/{len(chain.txns)} requests ({structured['noise_reduction_pct']}%).")

    def _tool_aegis_heal_payload(self, a: dict) -> tuple[dict, str]:
        payload = a["payload"]
        if isinstance(payload, str):
            payload = json.loads(payload)
        healed, rep = self.runtime.heal(a["tool_name"], payload)
        d = rep.to_dict()
        d.pop("matrix", None)
        text = (f"drift={rep.drift_detected} healed={rep.healed} strategy={rep.strategy} "
                f"remapped={sum(1 for m in rep.mappings if m.changed and m.actual)} unresolved={len(rep.unresolved)} "
                f"latency={rep.latency_ms:.2f}ms")
        return {"healed_payload": healed, "report": d}, text

    def _run_harvested(self, name: str, args: dict) -> tuple[dict, str]:
        async def go() -> dict:
            if self.http_client_factory is not None:
                async with self.http_client_factory() as client:
                    return await self.runtime.call(name, args, client=client)
            return await self.runtime.call(name, args)

        out = asyncio.run(go())
        return out, json.dumps(out["data"], ensure_ascii=False)[:20000]

    # ------------------------------------------------------------------ JSON-RPC
    def handle(self, msg: dict) -> dict | None:
        if not isinstance(msg, dict) or msg.get("jsonrpc") != "2.0" or "method" not in msg:
            return {"jsonrpc": "2.0", "id": msg.get("id") if isinstance(msg, dict) else None,
                    "error": {"code": JSONRPC_INVALID_REQUEST, "message": "invalid request"}}
        mid = msg.get("id")
        method = msg["method"]
        params = msg.get("params") or {}
        is_notification = "id" not in msg
        try:
            result = self._dispatch(method, params)
        except MCPError as e:
            return None if is_notification else {"jsonrpc": "2.0", "id": mid, "error": {"code": e.code, "message": str(e)}}
        except Exception as e:  # tool failures are reported as tool results, protocol failures as errors
            if method == "tools/call":
                return {"jsonrpc": "2.0", "id": mid, "result": {
                    "content": [{"type": "text", "text": f"{type(e).__name__}: {e}"}], "isError": True}}
            return None if is_notification else {"jsonrpc": "2.0", "id": mid, "error": {
                "code": JSONRPC_INTERNAL, "message": f"{type(e).__name__}: {e}", "data": traceback.format_exc(limit=3)}}
        if is_notification:
            return None
        return {"jsonrpc": "2.0", "id": mid, "result": result}

    def _dispatch(self, method: str, params: dict) -> Any:
        if method == "initialize":
            req = params.get("protocolVersion", SUPPORTED_PROTOCOLS[0])
            self.initialized = True
            return {"protocolVersion": req if req in SUPPORTED_PROTOCOLS else SUPPORTED_PROTOCOLS[0],
                    "capabilities": {"tools": {"listChanged": True}, "logging": {}},
                    "serverInfo": {"name": "aegis-mmu", "title": "aegis-mmu semantic MMU", "version": __version__},
                    "instructions": "Use aegis_fold_context before reading large codebases into context; expand folded "
                                    "regions with aegis_expand_node; record failed attempts with aegis_record_failure."}
        if method in ("notifications/initialized", "notifications/cancelled", "initialized"):
            return None
        if method == "ping":
            return {}
        if method == "tools/list":
            return {"tools": self.list_tools()}
        if method == "tools/call":
            if "name" not in params:
                raise MCPError(JSONRPC_INVALID_PARAMS, "missing tool name")
            return self.call_tool(params["name"], params.get("arguments") or {})
        if method == "resources/list":
            return {"resources": []}
        if method == "prompts/list":
            return {"prompts": []}
        if method == "logging/setLevel":
            return {}
        raise MCPError(JSONRPC_METHOD_NOT_FOUND, f"method not found: {method}")

    def serve_stdio(self, stdin: TextIO = sys.stdin, stdout: TextIO = sys.stdout) -> None:
        def write(obj: dict) -> None:
            stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            stdout.flush()

        self._notify = write
        for line in stdin:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except ValueError:
                write({"jsonrpc": "2.0", "id": None, "error": {"code": JSONRPC_PARSE_ERROR, "message": "parse error"}})
                continue
            batch = msg if isinstance(msg, list) else [msg]
            responses = [r for r in (self.handle(m) for m in batch) if r is not None]
            if isinstance(msg, list) and responses:
                stdout.write(json.dumps(responses) + "\n")
                stdout.flush()
            else:
                for r in responses:
                    write(r)


def main() -> None:  # pragma: no cover - entry point
    MCPServer().serve_stdio()


if __name__ == "__main__":  # pragma: no cover
    main()
