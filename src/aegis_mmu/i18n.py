"""Message catalog for human-facing CLI / dashboard output (English and Italian).

Only text meant for people goes through :func:`t`: status lines, Rich table titles and headers, dashboard
labels and ``--help`` texts. Machine-readable output (``--json``, ``aegis stats``, MCP / proxy payloads,
the Negative RAG blocks injected into prompts, generated reports) is never translated.

Language resolution, highest priority first: :func:`set_lang` (the CLI's ``--lang`` option), the
``AEGIS_LANG`` environment variable, then English.
"""

from __future__ import annotations

import os
import string

DEFAULT_LANG = "en"
LANGS = ("en", "it")
ENV_VAR = "AEGIS_LANG"

MESSAGES: dict[str, dict[str, str]] = {
    "en": {
        # --help texts (keys: help.app[.<param>] for the root command, help.<command>[.<param>] for subcommands)
        "help.app": "aegis-mmu: semantic MMU & self-healing execution proxy for AI coding agents.",
        "help.app.db": "SQLite path (':memory:' for ephemeral).",
        "help.app.version": "Print version and exit.",
        "help.app.lang": "Language of human-readable output (en, it). JSON output is never translated.",
        "help.proxy": "Run the transparent Anthropic/OpenAI-compatible folding proxy.",
        "help.proxy.mode": "fold-local | passthrough | upstream-forward",
        "help.proxy.repo": "Index this repository and page in relevant units.",
        "help.proxy.allow_upstream_override": "Honour the X-Aegis-Upstream request header.",
        "help.mcp": "Serve the MCP tools over stdio (JSON-RPC 2.0).",
        "help.fold": "Fold source files into a holographic multi-resolution context.",
        "help.fold.paths": "Files or directories to fold.",
        "help.fold.query": "Task / bug report.",
        "help.fold.budget": "Token budget (default adaptive).",
        "help.fold.as_json": "Emit JSON telemetry + context.",
        "help.expand": "Print the full L0 source of a folded vec_id.",
        "help.index": "Index a repository into the persistent store (for proxy page-in and MCP).",
        "help.harvest": "Compile a HAR capture into a deterministic, self-healing MCP tool.",
        "help.harvest.out": "Write the generated Python client here.",
        "help.heal": "Validate/heal a drifted JSON payload against a harvested tool's shadow contract.",
        "help.record-failure": "Record a failed approach into Negative RAG memory.",
        "help.stats": "Show aggregate telemetry.",
        "help.dash": "Rich terminal dashboard (resolution equalizer, KV stability, Negative RAG, healing matrix).",
        "help.dash.once": "Render a single frame and exit.",
        "help.dash.demo": "Populate an in-memory store with the bundled demo first.",
        "help.eval": "Run the benchmark suite and print the Markdown report.",
        "help.eval.out": "Report path (default evals/latest_report.json).",
        # CLI status lines and tables
        "cli.proxy.indexed": "[green]indexed[/] {files} files / {units} units from {repo}",
        "cli.proxy.banner": "[bold]aegis-mmu proxy[/] on http://{host}:{port} (mode={mode}) — "
                            "Claude Code: [cyan]ANTHROPIC_BASE_URL=http://{host}:{port}[/]  "
                            "OpenAI: [cyan]OPENAI_BASE_URL=http://{host}:{port}/v1[/]",
        "cli.fold.file_rule": "[bold]{path}[/] · skeleton L{level} · {orig}→{folded} tokens",
        "cli.fold.summary": "[bold green]{orig:,} → {folded:,} tokens ({pct:.1f}% saved) in {ms:.2f} ms[/]",
        "cli.expand.unknown": "[red]unknown vec_id {vec_id}[/]",
        "cli.expand.rule": "{path} :: {qualname} (lines {start}-{end})",
        "cli.harvest.title": "minimal execution chain for {target!r} ({path})",
        "cli.harvest.col.method": "method",
        "cli.harvest.col.url": "url",
        "cli.harvest.col.status": "status",
        "cli.harvest.pruned": "pruned [bold]{pruned}/{total}[/] requests ({noise:.1f}% noise); categories {categories}",
        "cli.harvest.inputs": "inputs: {inputs}",
        "cli.harvest.secret_input": "{name} (secret ${env})",
        "cli.harvest.written": "[green]client written to {out}[/]",
        "cli.harvest.registered": "[green]registered MCP tool[/] [bold]{name}[/]",
        "cli.heal.title": "schema drift report · strategy={strategy} · healed={healed} · {ms:.2f} ms",
        "cli.heal.col.expected": "expected",
        "cli.heal.col.actual": "actual",
        "cli.heal.col.conf": "conf",
        "cli.heal.col.name": "name",
        "cli.heal.col.value": "value",
        "cli.heal.col.struct": "struct",
        "cli.record_failure.done": "recorded failure #{id}",
        "common.unresolved": "UNRESOLVED",
        # dashboard (aegis dash)
        "tui.header.title": "[bold]aegis-mmu[/] v{version} · semantic MMU dashboard",
        "tui.header.engine": "engine: {engine}",
        "tui.header.requests": "requests",
        "tui.header.tokens_saved": "tokens saved",
        "tui.header.savings": "savings",
        "tui.header.kv_hit": "avg KV prefix hit",
        "tui.header.latency": "p50 / p95 latency",
        "tui.header.inventory": "units · memories · tools",
        "tui.eq.empty": "no fold events yet — run `aegis fold` or `aegis dash --demo`",
        "tui.eq.title": "AST mipmap resolution equalizer",
        "tui.eq.title_levels": "AST mipmap resolution equalizer · L0={l0} L1={l1}",
        "tui.eq.col.file": "file",
        "tui.eq.col.skel": "skel",
        "tui.eq.col.units": "L0 / L1 / folded units",
        "tui.eq.col.tokens": "tokens",
        "tui.eq.legend.l0": "█ L0 full  ",
        "tui.eq.legend.l1": "▓ L1 skeleton  ",
        "tui.eq.legend.folded": "░ L2/L3 folded",
        "tui.eq.legend.query": "   query: {query!r}",
        "tui.kv.title": "KV-cache prefix stability (append-only paging)",
        "tui.kv.col.turn": "turn",
        "tui.kv.col.hit": "prefix hit",
        "tui.kv.col.tokens": "tokens",
        "tui.kv.delta": "+{promotions}p {warnings}w",
        "tui.kv.trend": "hit-rate trend ",
        "tui.kv.avg": "   avg {avg:.1f}%",
        "tui.neg.empty": "no recorded anti-patterns",
        "tui.neg.title_empty": "Negative RAG alerts",
        "tui.neg.title": "Negative RAG alerts (recalled anti-patterns)",
        "tui.neg.col.pattern": "anti-pattern",
        "tui.neg.col.hits": "hits",
        "tui.heal.empty": "no schema drift observed",
        "tui.heal.title_empty": "Hungarian schema healing",
        "tui.heal.title": "Hungarian schema healing matrix",
        "tui.heal.col.expected": "expected",
        "tui.heal.col.remapped": "→ remapped to",
        "tui.heal.col.confidence": "confidence",
        "tui.heal.subtitle": "{tool} · strategy={strategy} · healed={healed} · {ms:.2f} ms · unresolved={unresolved}",
    },
    "it": {
        "help.app": "aegis-mmu: MMU semantica e proxy di esecuzione self-healing per agenti di coding AI.",
        "help.app.db": "Percorso del database SQLite (':memory:' per uno store temporaneo).",
        "help.app.version": "Mostra la versione ed esce.",
        "help.app.lang": "Lingua dell'output leggibile (en, it). L'output JSON non viene mai tradotto.",
        "help.proxy": "Avvia il proxy di folding trasparente, compatibile con Anthropic/OpenAI.",
        "help.proxy.mode": "fold-local | passthrough | upstream-forward",
        "help.proxy.repo": "Indicizza questo repository e carica nel contesto le unità rilevanti.",
        "help.proxy.allow_upstream_override": "Rispetta l'header di richiesta X-Aegis-Upstream.",
        "help.mcp": "Espone i tool MCP su stdio (JSON-RPC 2.0).",
        "help.fold": "Ripiega i file sorgente in un contesto olografico multi-risoluzione.",
        "help.fold.paths": "File o directory da ripiegare.",
        "help.fold.query": "Task o bug report.",
        "help.fold.budget": "Budget di token (predefinito: adattivo).",
        "help.fold.as_json": "Emette telemetria e contesto in JSON.",
        "help.expand": "Stampa il sorgente L0 completo di un vec_id ripiegato.",
        "help.index": "Indicizza un repository nello store persistente (per il page-in del proxy e per MCP).",
        "help.harvest": "Compila una cattura HAR in un tool MCP deterministico e self-healing.",
        "help.harvest.out": "Scrive qui il client Python generato.",
        "help.heal": "Valida e ripara un payload JSON con drift rispetto al contratto ombra di un tool generato.",
        "help.record-failure": "Registra un approccio fallito nella memoria Negative RAG.",
        "help.stats": "Mostra la telemetria aggregata.",
        "help.dash": "Dashboard da terminale (equalizzatore di risoluzione, stabilità KV, Negative RAG, "
                     "matrice di healing).",
        "help.dash.once": "Disegna un solo frame ed esce.",
        "help.dash.demo": "Popola prima uno store in memoria con la demo inclusa.",
        "help.eval": "Esegue la suite di benchmark e stampa il report Markdown.",
        "help.eval.out": "Percorso del report (predefinito: evals/latest_report.json).",
        "cli.proxy.indexed": "[green]indicizzati[/] {files} file / {units} unità da {repo}",
        "cli.proxy.banner": "[bold]proxy aegis-mmu[/] su http://{host}:{port} (mode={mode}) — "
                            "Claude Code: [cyan]ANTHROPIC_BASE_URL=http://{host}:{port}[/]  "
                            "OpenAI: [cyan]OPENAI_BASE_URL=http://{host}:{port}/v1[/]",
        "cli.fold.file_rule": "[bold]{path}[/] · scheletro L{level} · {orig}→{folded} token",
        "cli.fold.summary": "[bold green]{orig:,} → {folded:,} token (risparmio {pct:.1f}%) in {ms:.2f} ms[/]",
        "cli.expand.unknown": "[red]vec_id sconosciuto: {vec_id}[/]",
        "cli.expand.rule": "{path} :: {qualname} (righe {start}-{end})",
        "cli.harvest.title": "catena di esecuzione minima per {target!r} ({path})",
        "cli.harvest.col.method": "metodo",
        "cli.harvest.col.url": "url",
        "cli.harvest.col.status": "stato",
        "cli.harvest.pruned": "scartate [bold]{pruned}/{total}[/] richieste ({noise:.1f}% di rumore); "
                              "categorie {categories}",
        "cli.harvest.inputs": "input: {inputs}",
        "cli.harvest.secret_input": "{name} (segreto ${env})",
        "cli.harvest.written": "[green]client scritto in {out}[/]",
        "cli.harvest.registered": "[green]tool MCP registrato:[/] [bold]{name}[/]",
        "cli.heal.title": "report di drift dello schema · strategia={strategy} · riparato={healed} · {ms:.2f} ms",
        "cli.heal.col.expected": "atteso",
        "cli.heal.col.actual": "effettivo",
        "cli.heal.col.conf": "conf",
        "cli.heal.col.name": "nome",
        "cli.heal.col.value": "valore",
        "cli.heal.col.struct": "struttura",
        "cli.record_failure.done": "fallimento #{id} registrato",
        "common.unresolved": "IRRISOLTO",
        "tui.header.title": "[bold]aegis-mmu[/] v{version} · dashboard della MMU semantica",
        "tui.header.engine": "engine: {engine}",
        "tui.header.requests": "richieste",
        "tui.header.tokens_saved": "token risparmiati",
        "tui.header.savings": "risparmio",
        "tui.header.kv_hit": "hit medio prefisso KV",
        "tui.header.latency": "latenza p50 / p95",
        "tui.header.inventory": "unità · memorie · tool",
        "tui.eq.empty": "nessun evento di fold finora — esegui `aegis fold` o `aegis dash --demo`",
        "tui.eq.title": "Equalizzatore di risoluzione mipmap AST",
        "tui.eq.title_levels": "Equalizzatore di risoluzione mipmap AST · L0={l0} L1={l1}",
        "tui.eq.col.file": "file",
        "tui.eq.col.skel": "skel",
        "tui.eq.col.units": "unità L0 / L1 / ripiegate",
        "tui.eq.col.tokens": "token",
        "tui.eq.legend.l0": "█ L0 completo  ",
        "tui.eq.legend.l1": "▓ L1 scheletro  ",
        "tui.eq.legend.folded": "░ L2/L3 ripiegato",
        "tui.eq.legend.query": "   query: {query!r}",
        "tui.kv.title": "Stabilità del prefisso KV-cache (paging append-only)",
        "tui.kv.col.turn": "turno",
        "tui.kv.col.hit": "hit prefisso",
        "tui.kv.col.tokens": "token",
        "tui.kv.delta": "+{promotions}p {warnings}a",
        "tui.kv.trend": "andamento hit rate ",
        "tui.kv.avg": "   media {avg:.1f}%",
        "tui.neg.empty": "nessun anti-pattern registrato",
        "tui.neg.title_empty": "Avvisi Negative RAG",
        "tui.neg.title": "Avvisi Negative RAG (anti-pattern richiamati)",
        "tui.neg.col.pattern": "anti-pattern",
        "tui.neg.col.hits": "hit",
        "tui.heal.empty": "nessun drift di schema rilevato",
        "tui.heal.title_empty": "Self-healing dello schema (algoritmo ungherese)",
        "tui.heal.title": "Matrice di self-healing dello schema (algoritmo ungherese)",
        "tui.heal.col.expected": "atteso",
        "tui.heal.col.remapped": "→ rimappato su",
        "tui.heal.col.confidence": "confidenza",
        "tui.heal.subtitle": "{tool} · strategia={strategy} · riparato={healed} · {ms:.2f} ms · irrisolti={unresolved}",
    },
}

_current: str | None = None


def normalize(lang: str | None) -> str | None:
    """Return the supported language code for ``lang`` (case-insensitive), or ``None``."""
    code = (lang or "").strip().lower()
    return code if code in LANGS else None


def get_lang() -> str:
    """Active language: explicit :func:`set_lang`, else ``$AEGIS_LANG``, else English."""
    return _current or normalize(os.environ.get(ENV_VAR)) or DEFAULT_LANG


def set_lang(lang: str | None) -> None:
    """Force the output language (``None`` falls back to ``$AEGIS_LANG`` / English)."""
    global _current
    if lang is not None and normalize(lang) is None:
        raise ValueError(f"unsupported language {lang!r} (expected one of {', '.join(LANGS)})")
    _current = normalize(lang)


def t(key: str, **kwargs: object) -> str:
    """Translate ``key`` into the active language and fill its ``str.format`` placeholders."""
    template = MESSAGES[get_lang()].get(key) or MESSAGES[DEFAULT_LANG][key]
    return template.format(**kwargs)


def placeholders(template: str) -> set[tuple[str, str, str]]:
    """``(field, conversion, format_spec)`` of every placeholder in ``template``."""
    return {(field, conv or "", spec or "") for _, field, spec, conv in string.Formatter().parse(template)
            if field is not None}
