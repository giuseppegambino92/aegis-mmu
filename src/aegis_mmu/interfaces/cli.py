"""``aegis`` command-line interface."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.syntax import Syntax
from rich.table import Table

from .. import __version__
from ..config import AegisSettings

app = typer.Typer(add_completion=False, no_args_is_help=True, rich_markup_mode="rich",
                  help="aegis-mmu: semantic MMU & self-healing execution proxy for AI coding agents.")
console = Console()
err = Console(stderr=True)
_STATE: dict = {}


def _settings(**overrides) -> AegisSettings:
    s = AegisSettings()
    if _STATE.get("db"):
        overrides.setdefault("db_path", _STATE["db"])
    return s.model_copy(update={k: v for k, v in overrides.items() if v is not None})


def _mmu(**overrides):
    from ..db.store import AegisStore
    from ..mmu.pipeline import SemanticMMU

    s = _settings(**overrides)
    return SemanticMMU(AegisStore(s.resolved_db_path(), settings=s), settings=s)


@app.callback()
def main_callback(db: Optional[str] = typer.Option(None, "--db", envvar="AEGIS_DB_PATH", help="SQLite path (':memory:' for ephemeral)."),
                  version: bool = typer.Option(False, "--version", help="Print version and exit.")) -> None:
    if version:
        console.print(f"aegis-mmu {__version__}")
        raise typer.Exit()
    _STATE["db"] = db


@app.command()
def proxy(host: str = typer.Option("127.0.0.1"), port: int = typer.Option(8787),
          mode: str = typer.Option("upstream-forward", help="fold-local | passthrough | upstream-forward"),
          upstream_anthropic: Optional[str] = typer.Option(None), upstream_openai: Optional[str] = typer.Option(None),
          repo: Optional[Path] = typer.Option(None, help="Index this repository and page in relevant units."),
          allow_upstream_override: bool = typer.Option(False, help="Honour the X-Aegis-Upstream request header.")) -> None:
    """Run the transparent Anthropic/OpenAI-compatible folding proxy."""
    import uvicorn

    from .proxy import create_app

    mmu = _mmu(proxy_mode=mode, upstream_anthropic=upstream_anthropic, upstream_openai=upstream_openai,
               repo_page_in=True if repo else None)
    if repo:
        stats = mmu.index_paths(repo)
        err.print(f"[green]indexed[/] {stats['files']} files / {stats['units']} units from {repo}")
    err.print(f"[bold]aegis-mmu proxy[/] on http://{host}:{port} (mode={mode}) — "
              f"Claude Code: [cyan]ANTHROPIC_BASE_URL=http://{host}:{port}[/]  OpenAI: [cyan]OPENAI_BASE_URL=http://{host}:{port}/v1[/]")
    uvicorn.run(create_app(mmu, allow_upstream_override=allow_upstream_override, tune_gc_on_start=True),
                host=host, port=port, log_level="warning")


@app.command()
def mcp() -> None:
    """Serve the MCP tools over stdio (JSON-RPC 2.0)."""
    from ..runtime import tune_gc
    from .mcp_server import MCPServer

    server = MCPServer(_mmu())
    tune_gc()
    server.serve_stdio()


@app.command()
def fold(paths: list[Path] = typer.Argument(..., help="Files or directories to fold."),
         query: str = typer.Option(..., "--query", "-q", help="Task / bug report."),
         budget: Optional[int] = typer.Option(None, help="Token budget (default adaptive)."),
         as_json: bool = typer.Option(False, "--json", help="Emit JSON telemetry + context.")) -> None:
    """Fold source files into a holographic multi-resolution context."""
    from ..mmu.ast_mipmapper import EXT_LANG

    files = []
    for p in paths:
        if p.is_dir():
            files += [(str(f), f.read_text(encoding="utf-8")) for f in sorted(p.rglob("*")) if f.is_file() and f.suffix in EXT_LANG
                      and not any(part.startswith(".") for part in f.parts)]
        else:
            files.append((str(p), p.read_text(encoding="utf-8")))
    res = _mmu().fold(files, query, budget)
    if as_json:
        console.print_json(json.dumps(res.to_dict()))
        return
    for f in res.files:
        console.rule(f"[bold]{f.path}[/] · skeleton L{f.skeleton_level} · {f.tokens_original}→{f.tokens_folded} tokens")
        console.print(Syntax(f.text, f.language if f.language != "text" else "text", theme="ansi_dark", word_wrap=True))
    for w in res.warnings:
        console.print(f"[bold red]{w.render()}[/]")
    console.print(f"[bold green]{res.tokens_original:,} → {res.tokens_folded:,} tokens "
                  f"({res.savings_pct:.1f}% saved) in {res.latency_ms:.2f} ms[/]")


@app.command()
def expand(vec_id: str) -> None:
    """Print the full L0 source of a folded vec_id."""
    node = _mmu().expand(vec_id)
    if node is None:
        err.print(f"[red]unknown vec_id {vec_id}[/]")
        raise typer.Exit(1)
    console.rule(f"{node['path']} :: {node['qualname']} (lines {node['start_line']}-{node['end_line']})")
    console.print(Syntax(node["source"], node["language"], theme="ansi_dark", line_numbers=True, start_line=node["start_line"]))


@app.command()
def index(root: Path) -> None:
    """Index a repository into the persistent store (for proxy page-in and MCP)."""
    console.print(_mmu().index_paths(root))


@app.command()
def harvest(har: Path, target: str = typer.Option(..., "--target", "-t"), name: str = typer.Option(..., "--name", "-n"),
            out: Optional[Path] = typer.Option(None, "--out", "-o", help="Write the generated Python client here.")) -> None:
    """Compile a HAR capture into a deterministic, self-healing MCP tool."""
    from ..harvester.synthesizer import ToolRuntime, synthesize
    from ..harvester.taint_graph import extract_minimal_chain

    mmu = _mmu()
    chain = extract_minimal_chain(har, target)
    tool = synthesize(chain, name, mmu.engine)
    ToolRuntime(mmu.store).register(tool)
    t = Table(title=f"minimal execution chain for {target!r} ({chain.target_path})")
    for c in ("#", "method", "url", "status"):
        t.add_column(c)
    for s in chain.steps():
        t.add_row(str(s.index), s.method, s.origin + s.path, str(s.status))
    console.print(t)
    console.print(f"pruned [bold]{len(chain.pruned)}/{len(chain.txns)}[/] requests ({100 * chain.noise_reduction:.1f}% noise); "
                  f"categories {chain.categories}")
    console.print("inputs: " + ", ".join(f"{i.name}{' (secret $' + i.env + ')' if i.secret else ''}" for i in tool.spec.inputs))
    if out:
        out.write_text(tool.code)
        console.print(f"[green]client written to {out}[/]")
    console.print(f"[green]registered MCP tool[/] [bold]{tool.spec.name}[/]")


@app.command()
def heal(tool: str, payload: Path, out: Optional[Path] = typer.Option(None, "--out", "-o")) -> None:
    """Validate/heal a drifted JSON payload against a harvested tool's shadow contract."""
    from ..harvester.synthesizer import ToolRuntime

    healed, rep = ToolRuntime(_mmu().store).heal(tool, json.loads(payload.read_text()))
    t = Table(title=f"schema drift report · strategy={rep.strategy} · healed={rep.healed} · {rep.latency_ms:.2f} ms")
    for c in ("expected", "actual", "conf", "name", "value", "struct"):
        t.add_column(c)
    for m in rep.mappings:
        t.add_row(m.expected, m.actual or "[red]UNRESOLVED[/]", f"{m.confidence:.2f}", f"{m.name_sim:.2f}", f"{m.value_sim:.2f}", f"{m.struct_sim:.2f}")
    console.print(t)
    if out:
        out.write_text(json.dumps(healed, indent=2))
    else:
        console.print_json(json.dumps(healed))


@app.command("record-failure")
def record_failure(task: str = typer.Option(...), code_file: Optional[Path] = typer.Option(None), error: str = typer.Option(""),
                   why: Optional[str] = typer.Option(None), remedy: Optional[str] = typer.Option(None),
                   kind: str = typer.Option("exception")) -> None:
    """Record a failed approach into Negative RAG memory."""
    code = code_file.read_text() if code_file else ""
    fid = _mmu().negative.record(task=task, code=code, error=error, why=why, remedy=remedy, kind=kind)
    console.print(f"recorded failure #{fid}")


@app.command()
def stats() -> None:
    """Show aggregate telemetry."""
    console.print_json(json.dumps(_mmu().store.stats()))


@app.command()
def dash(once: bool = typer.Option(False, help="Render a single frame and exit."),
         demo: bool = typer.Option(False, help="Populate an in-memory store with the bundled demo first.")) -> None:
    """Rich terminal dashboard (resolution equalizer, KV stability, Negative RAG, healing matrix)."""
    from .tui import run_dashboard

    if demo:
        from ..db.store import AegisStore
        from ..demo import run_demo
        from ..mmu.pipeline import SemanticMMU

        s = _settings(db_path=":memory:")
        mmu = SemanticMMU(AegisStore(":memory:", settings=s), settings=s)
        run_demo(mmu)
    else:
        mmu = _mmu()
    run_dashboard(mmu.store, once=once)


@app.command("eval")
def eval_cmd(out: Optional[Path] = typer.Option(None, help="Report path (default evals/latest_report.json).")) -> None:
    """Run the benchmark suite and print the Markdown report."""
    from ..demo import datasets_dir

    root = datasets_dir().parent.parent
    sys.path.insert(0, str(root))
    from evals.benchmark_suite import main as bench_main  # type: ignore

    bench_main(out=str(out) if out else None)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
