"""Italian / English localisation of the human-facing CLI and dashboard output."""

import io
import json
import re

import pytest
import typer
from rich.console import Console
from typer.testing import CliRunner

from aegis_mmu import i18n
from aegis_mmu.interfaces import cli, tui

from .conftest import DATASETS

HAR = str(DATASETS / "skyfare.har")
RICH_TAG = re.compile(r"\[/?[a-z][a-z ]*\]|\[/\]")


@pytest.fixture(autouse=True)
def _clean_lang(monkeypatch):
    monkeypatch.delenv(i18n.ENV_VAR, raising=False)
    i18n.set_lang(None)
    yield
    i18n.set_lang(None)


@pytest.fixture()
def runner() -> CliRunner:
    return CliRunner()


def _harvest(runner: CliRunner, *pre: str, env: dict | None = None):
    res = runner.invoke(cli.app, ["--db", ":memory:", *pre, "harvest", HAR, "--target", "price_eur", "--name", "sky"],
                        env=env)
    assert res.exit_code == 0, res.output
    return res.output


# --- catalog -----------------------------------------------------------------------------------------------------

def test_catalog_languages_have_identical_keys():
    assert set(i18n.LANGS) == set(i18n.MESSAGES)
    en = set(i18n.MESSAGES["en"])
    for lang in i18n.LANGS:
        assert set(i18n.MESSAGES[lang]) == en, f"{lang}: {set(i18n.MESSAGES[lang]) ^ en}"


def test_catalog_entries_share_placeholders_and_markup():
    problems = []
    for key, en in i18n.MESSAGES["en"].items():
        for lang in i18n.LANGS:
            text = i18n.MESSAGES[lang][key]
            if not text.strip():
                problems.append(f"{lang}:{key} is empty")
            if i18n.placeholders(text) != i18n.placeholders(en):
                problems.append(f"{lang}:{key} placeholders {i18n.placeholders(text)} != {i18n.placeholders(en)}")
            if sorted(RICH_TAG.findall(text)) != sorted(RICH_TAG.findall(en)):
                problems.append(f"{lang}:{key} Rich markup differs")
    assert not problems, "\n".join(problems)


def test_lang_resolution_order(monkeypatch):
    assert i18n.get_lang() == "en"
    monkeypatch.setenv(i18n.ENV_VAR, "IT")
    assert i18n.get_lang() == "it"
    i18n.set_lang("en")
    assert i18n.get_lang() == "en" and i18n.t("tui.header.requests") == "requests"
    monkeypatch.setenv(i18n.ENV_VAR, "fr")  # unsupported -> English
    i18n.set_lang(None)
    assert i18n.get_lang() == "en"
    with pytest.raises(ValueError):
        i18n.set_lang("fr")


def test_every_cli_help_text_is_in_the_catalog():
    group = typer.main.get_command(cli.app)
    missing = [] if "help.app" in i18n.MESSAGES["en"] else ["help.app"]
    missing += [f"help.app.{p.name}" for p in group.params if getattr(p, "help", None) and p.name != "help"
                and f"help.app.{p.name}" not in i18n.MESSAGES["en"]]
    for name, cmd in group.commands.items():
        if f"help.{name}" not in i18n.MESSAGES["en"]:
            missing.append(f"help.{name}")
        missing += [f"help.{name}.{p.name}" for p in cmd.params if getattr(p, "help", None) and p.name != "help"
                    and f"help.{name}.{p.name}" not in i18n.MESSAGES["en"]]
    assert not missing


# --- real commands -----------------------------------------------------------------------------------------------

def test_harvest_is_english_by_default(runner):
    out = _harvest(runner)
    assert "minimal execution chain for 'price_eur'" in out
    assert "method" in out and "status" in out
    assert "registered MCP tool sky" in out
    assert "catena" not in out and "registrato" not in out


def test_harvest_in_italian_with_lang_option(runner):
    out = _harvest(runner, "--lang", "it")
    assert "catena di esecuzione minima per 'price_eur'" in out
    assert "metodo" in out and "stato" in out
    assert "scartate 37/40 richieste (92.5% di rumore)" in out
    assert "segreto $" in out
    assert "tool MCP registrato: sky" in out
    assert "registered MCP tool" not in out


def test_aegis_lang_env_and_option_precedence(runner):
    assert "tool MCP registrato: sky" in _harvest(runner, env={i18n.ENV_VAR: "it"})
    assert "registered MCP tool sky" in _harvest(runner, "--lang", "en", env={i18n.ENV_VAR: "it"})


def test_record_failure_message_is_localised(runner):
    args = ["--db", ":memory:", "record-failure", "--task", "cache keyed by request id", "--error", "MemoryError"]
    assert runner.invoke(cli.app, args).output.strip() == "recorded failure #1"
    assert runner.invoke(cli.app, ["--lang", "it", *args]).output.strip() == "fallimento #1 registrato"


def test_json_output_is_never_translated(runner):
    en = runner.invoke(cli.app, ["--db", ":memory:", "stats"])
    it = runner.invoke(cli.app, ["--db", ":memory:", "--lang", "it", "stats"])
    assert en.exit_code == it.exit_code == 0
    assert json.loads(en.output).keys() == json.loads(it.output).keys()


def test_help_is_localised_even_though_help_is_eager(runner):
    en = runner.invoke(cli.app, ["--help"])
    it = runner.invoke(cli.app, ["--lang", "it", "--help"])
    sub = runner.invoke(cli.app, ["--lang", "it", "fold", "--help"])
    assert "Show aggregate telemetry." in en.output
    assert "Mostra la telemetria aggregata." in it.output
    assert "File o directory da ripiegare." in sub.output
    # the language does not leak into the next English invocation
    assert "Show aggregate telemetry." in runner.invoke(cli.app, ["--help"]).output


def test_invalid_lang_is_rejected(runner):
    res = runner.invoke(cli.app, ["--lang", "fr", "stats"])
    assert res.exit_code != 0


def test_dashboard_renders_in_italian(store):
    def frame() -> str:
        con = Console(record=True, width=150, file=io.StringIO())
        tui.run_dashboard(store, once=True, console=con)
        return con.export_text()

    en = frame()
    assert "tokens saved" in en and "no fold events yet" in en and "no schema drift observed" in en
    i18n.set_lang("it")
    it = frame()
    assert "token risparmiati" in it and "nessun evento di fold finora" in it
    assert "nessun drift di schema rilevato" in it and "nessun anti-pattern registrato" in it
    assert "tokens saved" not in it
