"""Knapsack token-budget optimiser.

Multiple-choice knapsack over two item families:

* every **file** must take exactly one skeleton option: L3 (mandatory baseline) or L2,
* every **leaf** takes one of: none (covered by the skeleton), L1 (folded), L0 (full source).

    maximise   sum_i v_i(o_i)     s.t.   sum_i c_i(o_i) <= B

with ``v_leaf(L0) = rel``, ``v_leaf(L1) = 0.55 rel``, ``v_file(L2) = 0.12 max_rel + 0.01``.
Solved exactly by dynamic programming over a discretised budget axis (<= 2048 cells, vectorised
in NumPy) with back-pointers for reconstruction. All costs are measured token counts of the
pre-computed renderings, so any choice is syntactically valid by construction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

L1_FIDELITY = 0.55
L2_FILE_VALUE = 0.12
L2_FILE_BASE = 0.01


@dataclass
class Allocation:
    leaf_levels: dict[str, int] = field(default_factory=dict)
    file_levels: dict[str, int] = field(default_factory=dict)
    budget: int = 0
    estimated_tokens: int = 0
    utility: float = 0.0
    passthrough: bool = False


@dataclass
class _Item:
    key: str
    kind: str  # "leaf" | "file"
    options: list[tuple[int, float, int]]  # (cost_tokens, value, level)


def compute_budget(original_tokens: int, settings) -> int:
    return int(min(settings.token_budget, max(settings.min_budget, settings.budget_ratio * original_tokens)))


def allocate(files: Sequence, relevance: dict[str, float], budget: int, max_cells: int = 2048,
             callees=None) -> Allocation:
    """Choose per-file skeleton levels and per-leaf levels maximising utility under ``budget`` tokens."""
    original = sum(f.tokens_source for f in files)
    if original <= budget:
        return Allocation(
            leaf_levels={u.vec_id: 0 for f in files for u in f.units},
            file_levels={f.file_vec_id: 2 for f in files},
            budget=budget, estimated_tokens=original, utility=float(sum(relevance.values())), passthrough=True,
        )
    mandatory = sum(f.skeleton_tokens[3] for f in files)
    capacity = max(0, budget - mandatory)

    items: list[_Item] = []
    for f in files:
        file_rel = max((relevance.get(u.vec_id, 0.0) for u in f.units), default=0.0)
        up = max(0, f.skeleton_tokens[2] - f.skeleton_tokens[3])
        items.append(_Item(f.file_vec_id, "file", [(up, L2_FILE_VALUE * file_rel + L2_FILE_BASE, 2)]))
        for u in f.units:
            r = relevance.get(u.vec_id, 0.0)
            if r <= 1e-4:
                continue
            items.append(_Item(u.vec_id, "leaf", [(u.tokens[1], L1_FIDELITY * r, 1), (u.tokens[0], r, 0)]))

    g = max(1, math.ceil(capacity / max_cells))
    C = capacity // g
    best = np.zeros(C + 1, dtype=np.float64)
    choices = np.zeros((len(items), C + 1), dtype=np.int8)  # 0 = default option, k = options[k-1]
    for i, it in enumerate(items):
        new = best.copy()
        ch = np.zeros(C + 1, dtype=np.int8)
        for k, (cost, val, _lvl) in enumerate(it.options, start=1):
            w = math.ceil(cost / g)
            if w > C:
                continue
            cand = np.full(C + 1, -np.inf)
            cand[w:] = best[: C + 1 - w] + val
            better = cand > new + 1e-12
            new = np.where(better, cand, new)
            ch = np.where(better, k, ch)
        best = new
        choices[i] = ch

    c = int(np.argmax(best))
    alloc = Allocation(budget=budget, utility=float(best[c]))
    alloc.file_levels = {f.file_vec_id: 3 for f in files}
    used = mandatory
    for i in range(len(items) - 1, -1, -1):
        k = int(choices[i, c])
        if k == 0:
            continue
        cost, _val, lvl = items[i].options[k - 1]
        c -= math.ceil(cost / g)
        used += cost
        if items[i].kind == "file":
            alloc.file_levels[items[i].key] = lvl
        else:
            alloc.leaf_levels[items[i].key] = lvl
    alloc.estimated_tokens = used
    if callees is not None:
        _co_promote(files, alloc, budget, callees)
    return alloc


def _co_promote(files: Sequence, alloc: Allocation, budget: int, callees) -> None:
    """Call-cluster L0 expansion: when a function is at L0, upgrade its direct callees in the same
    file to L0 (most relevant parents first) while the token budget allows."""
    units = {u.vec_id: (u, f.file_vec_id) for f in files for u in f.units}
    parents = [v for v, lv in alloc.leaf_levels.items() if lv == 0 and v in units]
    for p in parents:
        pu, pf = units[p]
        for h in sorted(callees(p)):
            if h not in units or units[h][1] != pf or alloc.leaf_levels.get(h) == 0:
                continue
            hu = units[h][0]
            cur = hu.tokens[alloc.leaf_levels[h]] if h in alloc.leaf_levels else 0
            delta = hu.tokens[0] - cur
            if alloc.estimated_tokens + delta <= budget:
                alloc.leaf_levels[h] = 0
                alloc.estimated_tokens += delta
