"""Negative RAG: causal anti-pattern memory.

Failed attempts (exceptions, rolled-back diffs, rejected reviews) are embedded and stored. Every
intercepted task/proposed code is compared against that memory; a cosine similarity above the
threshold injects an ``<aegis_negative_memory_warning>`` block explaining why the approach failed
before and how to avoid it.

The similarity is ``0.65 * cos(task, mem_task_view) + 0.35 * cos(task+code, mem_all_view)``: the task
view carries most of the signal, the combined view (task, code, error, rationale) adds structural
evidence without letting short, generic code snippets trigger on their own.

The threshold self-calibrates to the active embedding provider: a built-in reference set of
mutually unrelated engineering tasks (independent of any benchmark data) is embedded once, and the
threshold is ``mean + z * std`` of their pairwise cosine (z = 5 by default, floor 0.25), i.e. a
recall must sit five standard deviations above the engine's own noise floor. If calibration cannot
run (e.g. a remote embedder is unreachable) the classic dense-embedding value 0.75 is used. An
explicit ``AEGIS_NEGATIVE_THRESHOLD`` always wins.
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass

import numpy as np

_TRACEBACK = re.compile(r"(Traceback \(most recent call last\)|Exception|Error:|error\[E\d+\]|panic:|FAILED|AssertionError)")


@dataclass
class NegativeWarning:
    id: int
    similarity: float
    kind: str
    title: str
    error: str
    why: str
    remedy: str

    def render(self) -> str:
        e = html.escape
        return (
            f'<aegis_negative_memory_warning id="{self.id}" kind="{e(self.kind)}" similarity="{self.similarity:.2f}">\n'
            f"Previously failed approach: {e(self.title)}\n"
            f"Observed failure: {e(self.error.strip().splitlines()[-1] if self.error.strip() else 'n/a')}\n"
            f"Why it failed: {e(self.why)}\n"
            f"How to avoid it: {e(self.remedy)}\n"
            f"</aegis_negative_memory_warning>"
        )


# Mutually unrelated engineering tasks used only to measure an engine's cosine noise floor.
NULL_REFERENCE = (
    "Add a dark mode toggle to the settings page",
    "Translate the onboarding emails into Portuguese",
    "Upgrade the CI runners to the latest Ubuntu image",
    "Write a migration that adds a nullable nickname column",
    "Render a bar chart of monthly signups on the admin dashboard",
    "Rename the Kotlin package from legacy to core",
    "Fix the typo in the README installation section",
    "Compress PNG assets in the marketing site build",
    "Add keyboard shortcuts to the document editor",
    "Expose a health check endpoint for the load balancer",
    "Sort the product catalogue alphabetically by default",
    "Document the release process in the wiki",
    "Add a CSV export button to the reports screen",
    "Increase test coverage of the date picker component",
    "Replace the logo in the mobile splash screen",
    "Support right-to-left layouts in the checkout form",
    "Bump the Rust toolchain and fix clippy warnings",
    "Make the search box autofocus on page load",
    "Add pagination to the audit log API",
    "Configure Dependabot for the frontend repository",
    "Show a tooltip explaining the loyalty points balance",
    "Split the monolithic settings module into smaller files",
    "Record a demo video of the new onboarding flow",
    "Add an index to speed up the customer search query",
    "Let users upload a profile picture from the camera",
    "Archive projects that have been inactive for a year",
    "Change the default font to Inter across the web app",
    "Add Swift unit tests for the address formatter",
    "Generate API docs from the OpenAPI specification",
    "Remove the deprecated feature flag for the old navbar",
    "Send a weekly digest summarising unread comments",
    "Add accessibility labels to the icon-only buttons",
)

_CALIBRATION_CACHE: dict[tuple, dict] = {}


def calibrate_threshold(engine, z: float = 5.0, floor: float = 0.25) -> dict:
    """Threshold = mean + z * std of the engine's cosine between unrelated tasks (cached per engine)."""
    key = (getattr(engine, "name", type(engine).__name__), float(z), float(floor))
    if key in _CALIBRATION_CACHE:
        return _CALIBRATION_CACHE[key]
    try:
        E = engine.embed(list(NULL_REFERENCE))
        S = E @ E.T
        iu = np.triu_indices(len(NULL_REFERENCE), 1)
        mu, sd = float(S[iu].mean()), float(S[iu].std())
        out = {"threshold": round(max(floor, min(0.95, mu + z * sd)), 4), "null_mean": round(mu, 4),
               "null_std": round(sd, 4), "z": z, "source": "calibrated"}
    except Exception as exc:  # noqa: BLE001 - e.g. remote embedder unreachable
        out = {"threshold": 0.75, "null_mean": None, "null_std": None, "z": z, "source": f"fallback ({type(exc).__name__})"}
    _CALIBRATION_CACHE[key] = out
    return out


def looks_like_failure(text: str) -> bool:
    return bool(_TRACEBACK.search(text or ""))


class NegativeRAG:
    def __init__(self, store, threshold: float | None = None):
        self.store = store
        self.engine = store.engine
        if threshold is None:
            threshold = store.settings.negative_threshold
        self.calibration: dict | None = None
        if threshold is None:
            self.calibration = calibrate_threshold(self.engine, getattr(store.settings, "negative_threshold_z", 5.0))
            threshold = self.calibration["threshold"]
        self.threshold = float(threshold)
        self._views: dict[int, np.ndarray] = {}

    @staticmethod
    def _views_text(title: str, task: str, code: str, error: str, why: str) -> tuple[str, str, str]:
        task_view = f"{title}\n{task}\n{why}"
        code_view = code or task_view
        all_view = f"{title}\n{task}\n{error}\n{why}\n{code}"
        return task_view, code_view, all_view

    def record(self, *, task: str, code: str = "", error: str = "", kind: str = "exception", title: str | None = None,
               why: str | None = None, remedy: str | None = None) -> int:
        last_err = error.strip().splitlines()[-1] if error.strip() else ""
        title = title or (task.strip().splitlines()[0][:120] if task.strip() else last_err[:120] or "failed attempt")
        why = why or (f"The attempt raised `{last_err}`." if last_err else "The attempt was rolled back.")
        remedy = remedy or "Do not repeat this approach verbatim; address the root cause named above first."
        tv, cv, av = self._views_text(title, task, code, error, why)
        embs = self.engine.embed([tv, cv, av])
        fid = self.store.add_failure(kind=kind, title=title, task=task, code=code, error=error, why=why,
                                     remedy=remedy, emb=embs.reshape(-1))
        self._views[fid] = embs
        return fid

    def _matrix(self) -> tuple[list[int], np.ndarray]:
        ids, flat = self.store.failure_matrix()
        if not ids:
            return [], np.zeros((0, 3, self.engine.dim), dtype=np.float32)
        return ids, flat.reshape(len(ids), 3, self.engine.dim)

    def scores(self, task: str, code: str = "") -> list[tuple[int, float]]:
        ids, mats = self._matrix()
        if not ids:
            return []
        probe = self.engine.embed([task or code, f"{task}\n{code}"])
        s_task = mats[:, 0, :] @ probe[0]
        s_all = mats[:, 2, :] @ probe[1]
        best = 0.65 * s_task + 0.35 * s_all
        return sorted(zip(ids, best.tolist()), key=lambda kv: -kv[1])

    def check(self, task: str, code: str = "", top_k: int = 3) -> list[NegativeWarning]:
        out = []
        for fid, sim in self.scores(task, code)[:top_k]:
            if sim < self.threshold:
                break
            row = self.store.get_failure(fid)
            if row is None:
                continue
            self.store.bump_failure(fid)
            out.append(NegativeWarning(fid, float(sim), row["kind"], row["title"], row["error"], row["why"], row["remedy"]))
        return out

    @staticmethod
    def render(warnings: list[NegativeWarning]) -> str:
        return "\n".join(w.render() for w in warnings)
