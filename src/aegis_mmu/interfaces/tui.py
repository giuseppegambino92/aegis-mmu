"""Rich terminal dashboard: resolution equalizer, KV-cache stability, Negative RAG alerts, healing matrix."""

from __future__ import annotations

import time

from rich.align import Align
from rich.console import Console, Group
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .. import __version__

SPARK = "▁▂▃▄▅▆▇█"
LEVEL_STYLE = {"L0": "bold green", "L1": "yellow", "L2": "cyan", "L3": "magenta"}


def sparkline(values: list[float], lo: float = 0.0, hi: float = 1.0) -> str:
    if not values:
        return "—"
    out = []
    for v in values:
        x = 0.0 if hi == lo else (min(max(v, lo), hi) - lo) / (hi - lo)
        out.append(SPARK[min(len(SPARK) - 1, int(round(x * (len(SPARK) - 1))))])
    return "".join(out)


def _bar(frac: float, width: int, style: str, char: str = "█") -> Text:
    n = max(0, min(width, int(round(frac * width))))
    return Text(char * n, style=style) + Text("·" * (width - n), style="grey35")


def header_panel(stats: dict, engine: str) -> Panel:
    t = Table.grid(expand=True, padding=(0, 2))
    for _ in range(6):
        t.add_column(justify="center")

    def cell(label: str, value: str, style: str = "bold white") -> Text:
        return Text.assemble((value + "\n", style), (label, "grey62"))

    t.add_row(
        cell("requests", str(stats.get("requests", 0))),
        cell("tokens saved", f"{stats.get('tokens_saved', 0):,}", "bold green"),
        cell("savings", f"{stats.get('savings_pct', 0.0):.1f}%", "bold green"),
        cell("avg KV prefix hit", f"{(stats.get('kv_hit_rate_avg') or 0) * 100:.1f}%", "bold cyan"),
        cell("p50 / p95 latency", f"{stats.get('latency_ms_p50') or 0:.2f} / {stats.get('latency_ms_p95') or 0:.2f} ms", "bold yellow"),
        cell("units · memories · tools", f"{stats.get('units_indexed', 0)} · {stats.get('negative_memories', 0)} · {stats.get('contracts', 0)}"),
    )
    return Panel(t, title=f"[bold]aegis-mmu[/] v{__version__} · semantic MMU dashboard", subtitle=f"engine: {engine}",
                 border_style="bright_blue")


def equalizer_panel(fold_event: dict | None, width: int = 26) -> Panel:
    if not fold_event or not fold_event.get("payload"):
        return Panel(Align.center(Text("no fold events yet — run `aegis fold` or `aegis dash --demo`", style="grey50")),
                     title="AST mipmap resolution equalizer", border_style="green")
    p = fold_event["payload"]
    t = Table(expand=True, box=None, padding=(0, 1))
    t.add_column("file", overflow="ellipsis", max_width=34, style="white")
    t.add_column("skel", justify="center")
    t.add_column("L0 / L1 / folded units", no_wrap=True)
    t.add_column("tokens", justify="right", style="grey70")
    for f in p.get("files", []):
        units = max(1, f.get("units", 1))
        l0, l1 = f.get("l0", 0), f.get("l1", 0)
        rest = max(0, units - l0 - l1)
        w0, w1 = int(round(width * l0 / units)), int(round(width * l1 / units))
        bar = Text("█" * w0, style=LEVEL_STYLE["L0"]) + Text("▓" * w1, style=LEVEL_STYLE["L1"]) + \
            Text("░" * max(0, width - w0 - w1), style=LEVEL_STYLE[f"L{f.get('skeleton', 3)}"])
        name = f["path"].split("/")[-1]
        t.add_row(name, Text(f"L{f.get('skeleton')}", style=LEVEL_STYLE[f"L{f.get('skeleton', 3)}"]), bar,
                  f"{f.get('orig', 0):,}→{f.get('folded', 0):,}")
    h = p.get("levels", {})
    legend = Text.assemble(("█ L0 full  ", LEVEL_STYLE["L0"]), ("▓ L1 skeleton  ", LEVEL_STYLE["L1"]),
                           ("░ L2/L3 folded", "cyan"), (f"   query: {p.get('query', '')[:60]!r}", "grey50"))
    return Panel(Group(t, legend), title=f"AST mipmap resolution equalizer · L0={h.get('L0', 0)} L1={h.get('L1', 0)}",
                 border_style="green")


def kv_panel(proxy_events: list[dict]) -> Panel:
    ev = list(reversed(proxy_events))[-24:]
    rates = [e["kv_hit_rate"] or 0.0 for e in ev]
    t = Table(expand=True, box=None, padding=(0, 1))
    t.add_column("turn", justify="right", style="grey62")
    t.add_column("prefix hit", no_wrap=True)
    t.add_column("tokens", justify="right")
    t.add_column("Δ", justify="right", style="grey62")
    for i, e in enumerate(ev[-8:], start=max(1, len(ev) - 7)):
        r = e["kv_hit_rate"] or 0.0
        pay = e.get("payload") or {}
        delta = f"+{len(pay.get('promotions', []))}p {len(pay.get('warnings', []))}w"
        t.add_row(str(i), _bar(r, 18, "cyan") + Text(f" {r * 100:5.1f}%"), f"{e.get('tokens_compressed') or 0:,}", delta)
    avg = sum(rates) / len(rates) if rates else 0.0
    head = Text.assemble(("hit-rate trend ", "grey62"), (sparkline(rates), "bold cyan"), (f"   avg {avg * 100:.1f}%", "bold"))
    return Panel(Group(head, t), title="KV-cache prefix stability (append-only paging)", border_style="cyan")


def negative_panel(failures: list[dict]) -> Panel:
    if not failures:
        return Panel(Text("no recorded anti-patterns", style="grey50"), title="Negative RAG alerts", border_style="red")
    t = Table(expand=True, box=None, padding=(0, 1))
    t.add_column("#", style="grey62", justify="right")
    t.add_column("anti-pattern", overflow="ellipsis")
    t.add_column("hits", justify="right", style="bold red")
    for f in sorted(failures, key=lambda x: -x.get("hits", 0))[:8]:
        t.add_row(str(f["id"]), f["title"][:70], str(f.get("hits", 0)))
    return Panel(t, title="Negative RAG alerts (recalled anti-patterns)", border_style="red")


def healing_panel(drift: list[dict]) -> Panel:
    if not drift:
        return Panel(Text("no schema drift observed", style="grey50"), title="Hungarian schema healing", border_style="magenta")
    rep = drift[0]["report"]
    t = Table(expand=True, box=None, padding=(0, 1))
    t.add_column("expected", overflow="ellipsis", style="white")
    t.add_column("→ remapped to", overflow="ellipsis", style="magenta")
    t.add_column("confidence", no_wrap=True)
    for m in rep["mappings"]:
        if not m.get("changed") and m.get("actual") == m.get("expected"):
            continue
        conf = m.get("confidence", 0.0)
        style = "green" if conf >= 0.6 else "yellow" if conf >= 0.4 else "red"
        t.add_row(m["expected"].replace("$.", ""), (m.get("actual") or "UNRESOLVED").replace("$.", ""),
                  _bar(conf, 12, style) + Text(f" {conf:.2f}"))
    sub = (f"{drift[0]['tool_name']} · strategy={rep['strategy']} · healed={rep['healed']} · "
           f"{rep['latency_ms']:.2f} ms · unresolved={len(rep['unresolved'])}")
    return Panel(t, title="Hungarian schema healing matrix", subtitle=sub, border_style="magenta")


def render_dashboard(store, engine_name: str = "") -> Layout:
    stats = store.stats()
    folds = store.recent_events("fold", 1)
    proxy = store.recent_events("proxy", 50)
    layout = Layout()
    layout.split_column(Layout(header_panel(stats, engine_name or store.engine.name), size=5), Layout(name="body"))
    layout["body"].split_row(Layout(name="left", ratio=3), Layout(name="right", ratio=2))
    layout["left"].split_column(Layout(equalizer_panel(folds[0] if folds else None), ratio=3),
                                Layout(healing_panel(store.drift_events(1)), ratio=2))
    layout["right"].split_column(Layout(kv_panel(proxy), ratio=3), Layout(negative_panel(store.list_failures()), ratio=2))
    return layout


def run_dashboard(store, once: bool = False, refresh: float = 1.0, console: Console | None = None,
                  height: int = 42) -> None:
    console = console or Console()
    if once:
        console.print(render_dashboard(store), height=height)
        return
    from rich.live import Live

    with Live(render_dashboard(store), console=console, refresh_per_second=4, screen=True) as live:
        try:
            while True:
                time.sleep(refresh)
                live.update(render_dashboard(store))
        except KeyboardInterrupt:
            pass
