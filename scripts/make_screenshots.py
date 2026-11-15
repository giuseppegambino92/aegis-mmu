"""Regenerate the README screenshots (``docs/images/{en,it}/<name>.svg``) from real ``aegis`` runs.

Every screenshot is one real invocation of the ``aegis`` Typer app, run in-process on the bundled demo data
(``evals/datasets``), with the CLI's Rich consoles swapped for ``Console(record=True)`` and exported with
``export_svg``. Each command runs once per language (``--lang en`` / ``--lang it``) against a fresh temporary
SQLite store. The offline ``fast-local`` embedder is forced, so nothing is downloaded and no network is used.
The data is deterministic; the timings shown are measured live and differ between runs.

Run with the package installed (``pip install -e .``):  ``python scripts/make_screenshots.py``
(``--only dashboard,heal`` regenerates a subset).
"""

from __future__ import annotations

import argparse
import io
import os
import shlex
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from rich.console import Console

from aegis_mmu import i18n
from aegis_mmu.interfaces import cli

ROOT = Path(__file__).resolve().parents[1]

DS = "evals/datasets"
OUT = ROOT / "docs" / "images"
LANGS = ("en", "it")


@dataclass(frozen=True)
class Shot:
    name: str
    width: int
    argv: list[str]
    setup: list[str] = field(default_factory=list)  # unrecorded command run first on the same store


SHOTS = [
    Shot("dashboard", 150, ["dash", "--demo", "--once"]),
    Shot("fold", 120, ["fold", f"{DS}/codebase/ledgerline/payments/refunds.py",
                       "-q", "refunds for partially shipped JPY orders are one cent short"]),
    Shot("harvest", 100, ["harvest", f"{DS}/skyfare.har", "--target", "price_eur", "--name", "skyfare_flight_search"]),
    Shot("heal", 124, ["heal", "skyfare_flight_search", f"{DS}/flight_v2_drift.json"],
         setup=["harvest", f"{DS}/skyfare.har", "--target", "price_eur", "--name", "skyfare_flight_search"]),
]


def _invoke(argv: list[str], console: Console) -> None:
    cli.console = cli.err = console
    try:
        cli.app(args=argv, prog_name="aegis", standalone_mode=False)
    finally:
        i18n.set_lang(None)


def capture(shot: Shot, lang: str) -> str:
    """Run ``shot`` in ``lang`` on a fresh store and return the recorded terminal as SVG."""
    with tempfile.TemporaryDirectory(prefix="aegis-shot-") as tmp:
        os.environ["AEGIS_DB_PATH"] = str(Path(tmp) / "aegis.db")
        if shot.setup:
            _invoke(["--lang", lang, *shot.setup], Console(file=io.StringIO(), width=shot.width))
        rec = Console(record=True, width=shot.width, file=io.StringIO(), force_terminal=True,
                      color_system="truecolor", legacy_windows=False)
        argv = ["--lang", lang, *shot.argv]
        _invoke(argv, rec)
        return rec.export_svg(title="aegis " + shlex.join(argv))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--only", help="comma-separated screenshot names: " + ", ".join(s.name for s in SHOTS))
    args = ap.parse_args()
    wanted = set(args.only.split(",")) if args.only else {s.name for s in SHOTS}
    os.chdir(ROOT)  # dataset paths in the commands (and in their output) are repository-relative
    for key in [k for k in os.environ if k.startswith("AEGIS_")]:
        del os.environ[key]  # a user's AEGIS_* settings (e.g. a networked embedder) must not leak in
    os.environ["AEGIS_EMBEDDING_PROVIDER"] = "fast-local"
    saved = cli.console, cli.err
    try:
        for shot in SHOTS:
            if shot.name not in wanted:
                continue
            for lang in LANGS:
                path = OUT / lang / f"{shot.name}.svg"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(capture(shot, lang), encoding="utf-8")
                print(f"wrote {path.relative_to(ROOT)}")
    finally:
        cli.console, cli.err = saved


if __name__ == "__main__":
    main()
