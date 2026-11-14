"""Run the benchmark suite N times back to back and record the run-to-run spread.

A single latency sample on a shared machine can land on either side of a threshold. This script
re-runs ``evals.benchmark_suite`` ``N`` times (each run still waits for ``AEGIS_BENCH_MAX_LOAD``),
collects every compliance row from each run into ``evals/latest_runs.json`` and lets
``evals/update_readme.py`` splice a per-run table into both READMEs. The last run's full report
stays in ``evals/latest_report.{json,md}``; no run is selected or discarded.

Run: ``AEGIS_BENCH_MAX_LOAD=4 python -m evals.repeat_runs 3``
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "evals" / "latest_runs.json"


def main(n: int = 3) -> None:
    runs = []
    for i in range(n):
        subprocess.run([sys.executable, "-m", "evals.benchmark_suite"], cwd=ROOT, check=True)
        rep = json.loads((ROOT / "evals" / "latest_report.json").read_text())
        runs.append({"generated_at": rep["generated_at"], "load_avg_start": rep["load_avg_start"],
                     "load_avg_end": rep["load_avg_end"],
                     "compliance": [{"target": c["target"], "measured": c["measured"], "pass": bool(c["pass"])}
                                    for c in rep["compliance"]]})
        print(f"run {i + 1}/{n}: {sum(c['pass'] for c in rep['compliance'])}/{len(rep['compliance'])} targets pass")
    RUNS.write_text(json.dumps({"runs": runs}, indent=2))
    from evals.update_readme import main as update_readme

    update_readme()


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 3)
