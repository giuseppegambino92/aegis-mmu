"""Splice the generated benchmark report into README.md and README.it.md (no hand-transcribed numbers).

Replaces the regions between ``<!-- BEGIN:COMPLIANCE -->``/``<!-- END:COMPLIANCE -->`` and
``<!-- BEGIN:REPORT -->``/``<!-- END:REPORT -->`` with content from evals/latest_report.{json,md}.
When ``evals/latest_runs.json`` (from ``python -m evals.repeat_runs``) ends with the current report, the
``<!-- BEGIN:VARIANCE -->`` region gets the per-run spread of every row that varied between runs.
Run: ``python -m evals.update_readme`` (the benchmark suite calls it automatically).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _splice(text: str, tag: str, body: str) -> str:
    pat = re.compile(rf"(<!-- BEGIN:{tag} -->).*?(<!-- END:{tag} -->)", re.S)
    if not pat.search(text):
        return text
    return pat.sub(lambda m: m.group(1) + "\n" + body.strip("\n") + "\n" + m.group(2), text)


HEADER = {
    "en": ("_Computed by the suite on {at} ({machine}, {cpus} CPUs, load avg {l0}→{l1}, tokenizer {tok}). "
           "**{n}/{tot} targets pass.**_", "| Target (acceptance rule) | Measured | Pass |"),
    "it": ("_Calcolato dalla suite il {at} ({machine}, {cpus} CPU, load avg {l0}→{l1}, tokenizer {tok}). "
           "**{n}/{tot} obiettivi superati.** Le righe sono generate automaticamente e restano in inglese._",
           "| Obiettivo (regola di accettazione) | Misurato | Esito |"),
}


VARIANCE = {
    "en": {"head": "_Run-to-run spread: {n} consecutive runs of the suite ({loads}). The last run is the one reported "
                   "above. In every run {all_pass} rows passed and {all_fail} failed; {flip} row(s) passed in some runs "
                   "and failed in others. Rows not listed had the same value in every run._",
           "cols": ("Target", "Run {i}", "Passed"), "load": "load {a}→{b}",
           "stale": "_No multi-run data for the current report: run `python -m evals.repeat_runs 3`._"},
    "it": {"head": "_Variabilità tra esecuzioni: {n} esecuzioni consecutive della suite ({loads}). L'ultima è quella "
                   "riportata sopra. In tutte le esecuzioni {all_pass} righe sono state superate e {all_fail} no; "
                   "{flip} riga/e superata/e solo in alcune. Le righe non elencate hanno avuto lo stesso valore in ogni "
                   "esecuzione._",
           "cols": ("Obiettivo", "Esecuzione {i}", "Superato"), "load": "load {a}→{b}",
           "stale": "_Nessun dato multi-esecuzione per il report corrente: eseguire `python -m evals.repeat_runs 3`._"},
}


def variance_block(report: dict, lang: str) -> str:
    t = VARIANCE[lang]
    path = ROOT / "evals" / "latest_runs.json"
    runs = json.loads(path.read_text())["runs"] if path.exists() else []
    if not runs or runs[-1]["generated_at"] != report["generated_at"] or runs[-1]["compliance"] != [
            {"target": c["target"], "measured": c["measured"], "pass": bool(c["pass"])} for c in report["compliance"]]:
        return t["stale"]
    n = len(runs)
    rows = list(zip(*[r["compliance"] for r in runs]))
    passes = [sum(c["pass"] for c in row) for row in rows]
    loads = ", ".join(t["load"].format(a=r["load_avg_start"][0], b=r["load_avg_end"][0]) for r in runs)
    out = [t["head"].format(n=n, loads=loads, all_pass=sum(k == n for k in passes),
                            all_fail=sum(k == 0 for k in passes), flip=sum(0 < k < n for k in passes)), ""]
    c0, ci, cp = t["cols"]
    out.append("| " + " | ".join([c0] + [ci.format(i=i + 1) for i in range(n)] + [cp]) + " |")
    out.append("|" + "---|" * (n + 2))
    for row, k in zip(rows, passes):
        if len({c["measured"] for c in row}) > 1 or 0 < k < n:
            cells = [f"{c['measured']} {'✅' if c['pass'] else '❌'}" for c in row]
            out.append(f"| {row[0]['target']} | " + " | ".join(cells) + f" | {k}/{n} |")
    return "\n".join(out)


def main() -> None:
    report = json.loads((ROOT / "evals" / "latest_report.json").read_text())
    md = (ROOT / "evals" / "latest_report.md").read_text()
    comp = report["compliance"]
    n = sum(c["pass"] for c in comp)
    body = re.sub(r"^# .*\n", "", md, count=1)
    body = re.sub(r"^## Target compliance.*?(?=^## )", "", body, flags=re.S | re.M)  # compliance shown above
    body = re.sub(r"^(#{2,5}) ", lambda m: "#" + m.group(1) + " ", body, flags=re.M)  # demote headings
    for name, lang in (("README.md", "en"), ("README.it.md", "it")):
        readme = ROOT / name
        if not readme.exists():
            continue
        head, cols = HEADER[lang]
        table = [head.format(at=report["generated_at"], machine=report["machine"], cpus=report["cpu_count"],
                             l0=report["load_avg_start"][0], l1=report["load_avg_end"][0], tok=report["tokenizer"],
                             n=n, tot=len(comp)), "", cols, "|---|---|---|"]
        table += [f"| {c['target']} | {c['measured']} | {'✅' if c['pass'] else '❌'} |" for c in comp]
        text = readme.read_text()
        text = _splice(text, "COMPLIANCE", "\n".join(table))
        text = _splice(text, "VARIANCE", variance_block(report, lang))
        text = _splice(text, "REPORT", body)
        readme.write_text(text)
        print(f"{name} updated ({n}/{len(comp)} targets pass)")


if __name__ == "__main__":
    main()
