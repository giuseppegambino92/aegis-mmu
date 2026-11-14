"""Splice the generated benchmark report into README.md and README.it.md (no hand-transcribed numbers).

Replaces the regions between ``<!-- BEGIN:COMPLIANCE -->``/``<!-- END:COMPLIANCE -->`` and
``<!-- BEGIN:REPORT -->``/``<!-- END:REPORT -->`` with content from evals/latest_report.{json,md}.
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
        text = _splice(text, "REPORT", body)
        readme.write_text(text)
        print(f"{name} updated ({n}/{len(comp)} targets pass)")


if __name__ == "__main__":
    main()
