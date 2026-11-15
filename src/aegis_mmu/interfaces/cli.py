"""``aegis`` command-line interface."""

from __future__ import annotations

import json
import sys
from enum import Enum
from pathlib import Path
from typing import Annotated, Any, Optional

import typer
from rich.console import Console
from rich.syntax import Syntax
from rich.table import Table
from typer.core import TyperCommand, TyperGroup

from .. import __version__, i18n
from ..config import AegisSettings
from ..i18n import t


class Lang(str, Enum):
    en = "en"
    it = "it"


# English --help texts from the catalog; ``_LocalizedGroup`` swaps them for the active language at parse time.
HELP = {k.removeprefix("help."): v for k, v in i18n.MESSAGES[i18n.DEFAULT_LANG].items() if k.startswith("help.")}


def _localize_help(command: TyperCommand | TyperGroup, key: str) -> None:
    """Set ``--help`` texts of ``command`` (and its subcommands) from the catalog, in the active language."""
    lang = i18n.MESSAGES[i18n.get_lang()]
    if f"help.{key}" in lang:
        command.help = lang[f"help.{key}"]
    for param in command.params:
        text = lang.get(f"help.{key}.{param.name}")
        if text is not None and hasattr(param, "help"):
            param.help = text
    for name, sub in getattr(command, "commands", {}).items():
        _localize_help(sub, name)


class _LocalizedGroup(TyperGroup):
    """Resolves ``--lang`` / ``$AEGIS_LANG`` before Click parses anything.

    ``--help`` is an eager option, so it would render before the ``--lang`` callback runs; reading the
    language from the raw arguments first lets ``aegis --lang it fold --help`` show Italian help.
    """

    def parse_args(self, ctx: Any, args: list[str]) -> list[str]:  # ctx: Typer's (vendored) click Context
        i18n.set_lang(i18n.normalize(self._lang_arg(args)))  # unknown values are rejected by the parser below
        _localize_help(self, "app")
        return super().parse_args(ctx, args)

    def _lang_arg(self, args: list[str]) -> str | None:
        for i, arg in enumerate(args):
            if arg == "--" or arg in self.commands:
                break
            if arg == "--lang" and i + 1 < len(args):
                return args[i + 1]
            if arg.startswith("--lang="):
                return arg.split("=", 1)[1]
        return None


app = typer.Typer(cls=_LocalizedGroup, add_completion=False, no_args_is_help=True, rich_markup_mode="rich",
                  help=HELP["app"])
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
def main_callback(db: Optional[str] = typer.Option(None, "--db", envvar="AEGIS_DB_PATH", help=HELP["app.db"]),
                  version: bool = typer.Option(False, "--version", help=HELP["app.version"]),
                  lang: Annotated[Lang | None, typer.Option("--lang", envvar=i18n.ENV_VAR, case_sensitive=False,
                                                            help=HELP["app.lang"])] = None) -> None:
    i18n.set_lang(lang.value if lang else None)
    if version:
        console.print(f"aegis-mmu {__version__}")
        raise typer.Exit()
    _STATE["db"] = db


@app.command(help=HELP["proxy"])
def proxy(host: str = typer.Option("127.0.0.1"), port: int = typer.Option(8787),
          mode: str = typer.Option("upstream-forward", help=HELP["proxy.mode"]),
          upstream_anthropic: Optional[str] = typer.Option(None), upstream_openai: Optional[str] = typer.Option(None),
          repo: Optional[Path] = typer.Option(None, help=HELP["proxy.repo"]),
          allow_upstream_override: bool = typer.Option(False, help=HELP["proxy.allow_upstream_override"])) -> None:
    import uvicorn

    from .proxy import create_app

    mmu = _mmu(proxy_mode=mode, upstream_anthropic=upstream_anthropic, upstream_openai=upstream_openai,
               repo_page_in=True if repo else None)
    if repo:
        stats = mmu.index_paths(repo)
        err.print(t("cli.proxy.indexed", files=stats["files"], units=stats["units"], repo=repo))
    err.print(t("cli.proxy.banner", host=host, port=port, mode=mode))
    uvicorn.run(create_app(mmu, allow_upstream_override=allow_upstream_override, tune_gc_on_start=True),
                host=host, port=port, log_level="warning")


@app.command(help=HELP["mcp"])
def mcp() -> None:
    from ..runtime import tune_gc
    from .mcp_server import MCPServer

    server = MCPServer(_mmu())
    tune_gc()
    server.serve_stdio()


@app.command(help=HELP["fold"])
def fold(paths: list[Path] = typer.Argument(..., help=HELP["fold.paths"]),
         query: str = typer.Option(..., "--query", "-q", help=HELP["fold.query"]),
         budget: Optional[int] = typer.Option(None, help=HELP["fold.budget"]),
         as_json: bool = typer.Option(False, "--json", help=HELP["fold.as_json"])) -> None:
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
        console.rule(t("cli.fold.file_rule", path=f.path, level=f.skeleton_level, orig=f.tokens_original,
                       folded=f.tokens_folded))
        console.print(Syntax(f.text, f.language if f.language != "text" else "text", theme="ansi_dark", word_wrap=True))
    for w in res.warnings:
        console.print(f"[bold red]{w.render()}[/]")  # prompt payload (also injected for the agent): not translated
    console.print(t("cli.fold.summary", orig=res.tokens_original, folded=res.tokens_folded, pct=res.savings_pct,
                    ms=res.latency_ms))


@app.command(help=HELP["expand"])
def expand(vec_id: str) -> None:
    node = _mmu().expand(vec_id)
    if node is None:
        err.print(t("cli.expand.unknown", vec_id=vec_id))
        raise typer.Exit(1)
    console.rule(t("cli.expand.rule", path=node["path"], qualname=node["qualname"], start=node["start_line"],
                   end=node["end_line"]))
    console.print(Syntax(node["source"], node["language"], theme="ansi_dark", line_numbers=True, start_line=node["start_line"]))


@app.command(help=HELP["index"])
def index(root: Path) -> None:
    console.print(_mmu().index_paths(root))


@app.command(help=HELP["harvest"])
def harvest(har: Path, target: str = typer.Option(..., "--target", "-t"), name: str = typer.Option(..., "--name", "-n"),
            out: Optional[Path] = typer.Option(None, "--out", "-o", help=HELP["harvest.out"])) -> None:
    from ..harvester.synthesizer import ToolRuntime, synthesize
    from ..harvester.taint_graph import extract_minimal_chain

    mmu = _mmu()
    chain = extract_minimal_chain(har, target)
    tool = synthesize(chain, name, mmu.engine)
    ToolRuntime(mmu.store).register(tool)
    table = Table(title=t("cli.harvest.title", target=target, path=chain.target_path))
    for c in ("#", t("cli.harvest.col.method"), t("cli.harvest.col.url"), t("cli.harvest.col.status")):
        table.add_column(c)
    for s in chain.steps():
        table.add_row(str(s.index), s.method, s.origin + s.path, str(s.status))
    console.print(table)
    console.print(t("cli.harvest.pruned", pruned=len(chain.pruned), total=len(chain.txns),
                    noise=100 * chain.noise_reduction, categories=chain.categories))
    console.print(t("cli.harvest.inputs", inputs=", ".join(
        t("cli.harvest.secret_input", name=i.name, env=i.env) if i.secret else i.name for i in tool.spec.inputs)))
    if out:
        out.write_text(tool.code)
        console.print(t("cli.harvest.written", out=out))
    console.print(t("cli.harvest.registered", name=tool.spec.name))


@app.command(help=HELP["heal"])
def heal(tool: str, payload: Path, out: Optional[Path] = typer.Option(None, "--out", "-o")) -> None:
    from ..harvester.synthesizer import ToolRuntime

    healed, rep = ToolRuntime(_mmu().store).heal(tool, json.loads(payload.read_text()))
    table = Table(title=t("cli.heal.title", strategy=rep.strategy, healed=rep.healed, ms=rep.latency_ms))
    for c in ("expected", "actual", "conf", "name", "value", "struct"):
        table.add_column(t(f"cli.heal.col.{c}"))
    unresolved = f"[red]{t('common.unresolved')}[/]"
    for m in rep.mappings:
        table.add_row(m.expected, m.actual or unresolved, f"{m.confidence:.2f}", f"{m.name_sim:.2f}", f"{m.value_sim:.2f}", f"{m.struct_sim:.2f}")
    console.print(table)
    if out:
        out.write_text(json.dumps(healed, indent=2))
    else:
        console.print_json(json.dumps(healed))


@app.command("record-failure", help=HELP["record-failure"])
def record_failure(task: str = typer.Option(...), code_file: Optional[Path] = typer.Option(None), error: str = typer.Option(""),
                   why: Optional[str] = typer.Option(None), remedy: Optional[str] = typer.Option(None),
                   kind: str = typer.Option("exception")) -> None:
    code = code_file.read_text() if code_file else ""
    fid = _mmu().negative.record(task=task, code=code, error=error, why=why, remedy=remedy, kind=kind)
    console.print(t("cli.record_failure.done", id=fid))


@app.command(help=HELP["stats"])
def stats() -> None:
    console.print_json(json.dumps(_mmu().store.stats()))


@app.command(help=HELP["dash"])
def dash(once: bool = typer.Option(False, help=HELP["dash.once"]),
         demo: bool = typer.Option(False, help=HELP["dash.demo"])) -> None:
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
    run_dashboard(mmu.store, once=once, console=console)


@app.command("eval", help=HELP["eval"])
def eval_cmd(out: Optional[Path] = typer.Option(None, help=HELP["eval.out"])) -> None:
    from ..demo import datasets_dir

    root = datasets_dir().parent.parent
    sys.path.insert(0, str(root))
    from evals.benchmark_suite import main as bench_main  # type: ignore

    bench_main(out=str(out) if out else None)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
