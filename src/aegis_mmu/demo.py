"""Bundled end-to-end demo + agent-session simulator (used by ``aegis dash --demo`` and the benchmarks)."""

from __future__ import annotations

import json
import os
from pathlib import Path

from .harvester.synthesizer import ToolRuntime, synthesize
from .harvester.taint_graph import extract_minimal_chain


def datasets_dir() -> Path:
    env = os.environ.get("AEGIS_DATASETS")
    if env:
        return Path(env)
    here = Path(__file__).resolve()
    for parent in here.parents:
        cand = parent / "evals" / "datasets"
        if cand.is_dir():
            return cand
    return Path.cwd() / "evals" / "datasets"


def load_codebase(ds: Path | None = None) -> list[tuple[str, str]]:
    root = (ds or datasets_dir()) / "codebase"
    return [(str(p.relative_to(root)), p.read_text(encoding="utf-8")) for p in sorted(root.rglob("*")) if p.is_file()]


def catn(text: str) -> str:
    """Claude Code-style numbered file listing (``cat -n``)."""
    return "\n".join(f"{i:>6}\t{line}" for i, line in enumerate(text.split("\n"), start=1))


def agent_session_requests(turns: list[dict], files: dict[str, str], system: str = "You are a coding agent.",
                           model: str = "claude-opus-5-5") -> list[dict]:
    """Anthropic requests of a simulated agent loop: ask -> Read(tool_use) -> tool_result -> request -> reply."""
    msgs: list[dict] = []
    reqs: list[dict] = []
    for t, turn in enumerate(turns):
        path = turn["target"]["path"]
        tid = f"toolu_{t:02d}"
        msgs.append({"role": "user", "content": [{"type": "text", "text": turn["query"]}]})
        msgs.append({"role": "assistant", "content": [
            {"type": "text", "text": f"Let me read {path}."},
            {"type": "tool_use", "id": tid, "name": "Read", "input": {"file_path": path}}]})
        msgs.append({"role": "user", "content": [{"type": "tool_result", "tool_use_id": tid, "content": catn(files[path])}]})
        reqs.append({"model": model, "max_tokens": 4096, "system": system, "messages": json.loads(json.dumps(msgs))})
        msgs.append({"role": "assistant", "content": [{"type": "text", "text": turn.get("assistant_reply", "Done.")}]})
    return reqs


def run_demo(mmu, ds: Path | None = None, verbose: bool = False) -> dict:
    ds = ds or datasets_dir()
    files = load_codebase(ds)
    fmap = dict(files)
    queries = json.loads((ds / "queries.json").read_text())
    out: dict = {"folds": []}
    # 1. Negative RAG memory
    neg = json.loads((ds / "negative_rag.json").read_text())
    for m in neg["memories"]:
        mmu.negative.record(task=m["task"], code=m["code"], error=m["error"], kind=m["kind"], title=m["title"],
                            why=m["why"], remedy=m["remedy"])
    # 2. KV-stable agent session through the proxy rewriter
    session = json.loads((ds / "session.json").read_text())
    reqs = agent_session_requests(session["turns"], fmap)
    out["session"] = [round(mmu.rewrite_request(r, "anthropic", session_id="demo-session").kv_hit_rate, 4) for r in reqs]
    # 3. Holographic folds of the whole codebase
    for q in queries[:3]:
        r = mmu.fold(files, q["query"])
        out["folds"].append({"query": q["id"], "savings_pct": round(r.savings_pct, 2), "latency_ms": round(r.latency_ms, 3)})
    mmu.fold(files, neg["probes"][0]["task"] + " " + queries[0]["query"], proposed_code=neg["probes"][0]["code"])
    # 4. Web-to-MCP harvest + drift healing
    chain = extract_minimal_chain(ds / "skyfare.har", "price_eur")
    tool = synthesize(chain, "skyfare_flight_search", mmu.engine)
    rt = ToolRuntime(mmu.store)
    rt.register(tool)
    _, rep = rt.heal(tool.spec.name, json.loads((ds / "flight_v2_drift.json").read_text()))
    out["harvest"] = chain.to_dict()
    out["heal"] = {"healed": rep.healed, "latency_ms": round(rep.latency_ms, 3), "strategy": rep.strategy}
    return out
