"""3-stage hybrid retrieval.

Stage 1 (candidate generation, sub-millisecond on in-memory mirrors):
    BM25 (SQLite FTS5), concept-expanded BM25 (SE thesaurus), 1-bit Hamming (XOR/popcount) and
    Matryoshka-64 cosine, fused with weighted RRF; the top fused seeds then drive a 2-hop
    reference-graph walk whose ranking joins a second RRF:

        RRF(d) = sum_{m in {BM25, BM25x, BQ, MRL64, Graph}} w_m / (k + rank_m(d))

Stage 2 (late interaction):
    ColBERT MaxSim  ``sum_i max_j q_i . d_j / |Q|``  over the fused candidates, blended with the
    min-max normalised RRF score.

Stage 2b (1-hop call/reference-graph propagation):
    neighbours of the top hits (same file or imported module) join the candidate set and every
    score is propagated one hop:  s'(B) = s(B) + alpha * max_{A in N(B)} s(A)   (alpha = 0.35).
    A handler that matches the symptom therefore lifts the helper it calls, and vice versa.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from ..embeddings.colbert import batch_maxsim
from ..embeddings.matryoshka import rrf_fuse


@dataclass
class Hit:
    vec_id: str
    score: float
    rrf: float
    maxsim: float
    ranks: dict[str, int] = field(default_factory=dict)


@dataclass
class RetrievalResult:
    hits: list[Hit]
    timings_ms: dict[str, float]

    def relevance(self, temperature: float = 0.08) -> dict[str, float]:
        """Soft relevance in (0, 1]: ``exp((s - s_max) / T)``."""
        if not self.hits:
            return {}
        top = self.hits[0].score
        return {h.vec_id: float(np.exp((h.score - top) / max(temperature, 1e-6))) for h in self.hits}

    def ranked_ids(self) -> list[str]:
        return [h.vec_id for h in self.hits]


def _minmax(x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return x
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-12:
        return np.ones_like(x)
    return (x - lo) / (hi - lo)


class HybridRetriever:
    def __init__(self, store, settings=None):
        from ..config import get_settings

        self.store = store
        self.engine = store.engine
        self.settings = settings or get_settings()

    def _graph_rank(self, seeds: list[str], allowed: set[str] | None, limit: int) -> list[str]:
        scores: dict[str, float] = {}
        for r, s in enumerate(seeds, start=1):
            w = 1.0 / r
            for n1 in self.store.neighbors(s):
                if allowed is not None and n1 not in allowed:
                    continue
                scores[n1] = scores.get(n1, 0.0) + w
                for n2 in self.store.neighbors(n1):
                    if n2 != s and (allowed is None or n2 in allowed):
                        scores[n2] = scores.get(n2, 0.0) + 0.35 * w
        return [v for v, _ in sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:limit]]

    def _final(self, qvec: np.ndarray, cand: list[str], ms: np.ndarray, rrf_scores: np.ndarray) -> np.ndarray:
        """final = w * dense + (1 - w) * (blend * MaxSim + (1 - blend) * RRF), all min-max normalised.

        ``dense_weight`` (w) lets a strong dense model (e.g. the neural tier) contribute its full
        cosine instead of only its BQ / MRL-64 ranks inside RRF."""
        s = self.settings
        base = s.maxsim_blend * _minmax(ms) + (1.0 - s.maxsim_blend) * _minmax(rrf_scores)
        w = self.dense_weight
        if w <= 0 or not cand:
            return base
        dense = self.store.dense(cand) @ qvec
        return w * _minmax(dense.astype(np.float32)) + (1.0 - w) * base

    @property
    def dense_weight(self) -> float:
        s = self.settings
        w = s.dense_weight
        if w is None:  # engine default, frozen on the dev query set (see README)
            w = s.dense_weight_local if str(getattr(self.engine, "name", "")).startswith("fast-local") else s.dense_weight_neural
        return float(w)

    def _prior_weights(self) -> tuple[float, float, float]:
        s = self.settings
        local = str(getattr(self.engine, "name", "")).startswith("fast-local")
        dflt = (0.25, 0.15, 0.0) if local else (0.0, 0.0, 0.0)  # frozen on dev MRR (see README)
        vals = (s.file_prior_weight, s.sibling_caller_weight, s.salience_weight)
        return tuple(float(v if v is not None else d) for v, d in zip(vals, dflt))  # type: ignore[return-value]

    def _hierarchical(self, qtok, qw, qvec, cand, final, ms, rrf_scores, allowed):
        """FinalScore(sym) = s(sym) + wf * FileScore(file(sym)) + wc * BestSameFileCaller(sym) + ws * salience.

        FileScore is the best symbol score in the file; the units of the top-ranked file join the
        candidate set; the caller transfer only applies inside that top file (no cross-file
        pollution); salience is min-max normalised over the candidates."""
        store = self.store
        wf, wc, ws = self._prior_weights()
        top_file = store.file_of(cand[int(np.argmax(final))])
        if top_file is not None:
            present = set(cand)
            extra = [v for v in store.units_of_file(top_file) if v not in present and (allowed is None or v in allowed)]
            if extra:
                ms_x = batch_maxsim(qtok, store.token_matrices(extra), weights=qw)
                cand = cand + extra
                ms = np.concatenate([ms, ms_x])
                rrf_scores = np.concatenate([rrf_scores, np.zeros(len(extra), dtype=np.float32)])
                final = self._final(qvec, cand, ms, rrf_scores)
        base = final.copy()
        file_best: dict[str, float] = {}
        for v, sc in zip(cand, base):
            f = store.file_of(v)
            if f is not None and sc > file_best.get(f, -1.0):
                file_best[f] = float(sc)
        sal = np.array([store.salience(v) for v in cand], dtype=np.float32)
        sal = _minmax(sal) if sal.size and sal.max() > 0 else np.zeros_like(sal)
        pos = {v: i for i, v in enumerate(cand)}
        # per-file diminishing returns: only the top `prior_full_per_file` units of each file get the
        # full prior; the rest get `prior_decay` x it, so one file cannot monopolise the L0 budget
        rank_in_file: dict[int, int] = {}
        seen: dict[str, int] = {}
        for i in np.argsort(-base, kind="stable"):
            f = store.file_of(cand[int(i)])
            rank_in_file[int(i)] = seen.get(f, 0)
            seen[f] = seen.get(f, 0) + 1
        full_n, decay = self.settings.prior_full_per_file, self.settings.prior_decay
        out = base.copy()
        for i, v in enumerate(cand):
            f = store.file_of(v)
            k = 1.0 if (full_n <= 0 or rank_in_file[i] < full_n) else decay
            out[i] += k * wf * file_best.get(f, 0.0) + ws * float(sal[i])
            if wc > 0 and f == top_file:
                best = 0.0
                for p in cand:
                    j = pos[p]
                    if store.file_of(p) == f and v in store.callees(p) and base[j] > best:
                        best = float(base[j])
                out[i] += k * wc * best
        return cand, out, ms, rrf_scores

    def _activate_callees(self, cand, final, ms, rrf_scores, allowed):
        """s(H) = max(s(H), g1 * s(P)) for same-file direct callees H of a top hit P, g2 for 2-hop
        callees, plus a small boost when the activated callee is a leaf (calls no further project
        function); an activated helper is capped just below the parent that activated it."""
        s = self.settings
        g1, g2, leaf = s.callee_gamma1, s.callee_gamma2, s.callee_leaf_boost
        order = np.argsort(-final, kind="stable")[: s.graph_seed_top]
        boost: dict[str, float] = {}
        cap: dict[str, float] = {}  # an activated helper never outranks the parent that activated it
        same_file = self.store.same_file
        max_in = s.callee_max_in_degree
        if max_in > 0:  # exclusive helpers only: generic high-fan-in utilities are never boosted
            base_same = same_file
            same_file = lambda a, b: base_same(a, b) and self.store.same_file_in_degree(b) <= max_in  # noqa: E731
        for i in order:
            p, sp = cand[int(i)], float(final[int(i)])
            for h in self.store.callees(p):
                if (allowed is not None and h not in allowed) or not same_file(p, h):
                    continue
                boost[h] = max(boost.get(h, 0.0), g1 * sp)
                cap[h] = max(cap.get(h, 0.0), sp - 1e-4)
                for h2 in self.store.callees(h):
                    if h2 != p and same_file(h, h2) and (allowed is None or h2 in allowed):
                        boost[h2] = max(boost.get(h2, 0.0), g2 * sp)
                        cap[h2] = max(cap.get(h2, 0.0), sp - 2e-4)
        if not boost:
            return cand, final, ms, rrf_scores
        pos = {v: i for i, v in enumerate(cand)}
        extra = [h for h in boost if h not in pos]
        if extra:
            cand = cand + extra
            final = np.concatenate([final, np.zeros(len(extra), dtype=np.float32)])
            ms = np.concatenate([ms, np.zeros(len(extra), dtype=np.float32)])
            rrf_scores = np.concatenate([rrf_scores, np.zeros(len(extra), dtype=np.float32)])
            pos = {v: i for i, v in enumerate(cand)}
        final = final.copy()
        for h, b in boost.items():
            i = pos[h]
            if b > final[i]:
                final[i] = min(b + (leaf if not self.store.callees(h) else 0.0), cap[h])
        return cand, final, ms, rrf_scores

    def _propagate(self, cand: list[str], scores: np.ndarray, alpha: float) -> np.ndarray:
        """s'(B) = s(B) + alpha * max_{A in N(B)} s(A), computed from the pre-propagation scores."""
        idx = {v: i for i, v in enumerate(cand)}
        out = scores.copy()
        for i, v in enumerate(cand):
            best = 0.0
            for nb in self.store.neighbors(v):
                j = idx.get(nb)
                if j is not None and scores[j] > best:
                    best = float(scores[j])
            out[i] += alpha * best
        return out

    def search(self, query: str, candidates: Sequence[str] | None = None, k: int | None = None) -> RetrievalResult:
        s = self.settings
        K = s.stage1_candidates
        t = {}
        t0 = time.perf_counter()
        qw = None
        tw = None
        if s.query_idf and hasattr(self.engine, "query_token_list"):
            qtoks = self.engine.query_token_list(query, 64)
            qw = np.asarray(self.store.idf(qtoks), dtype=np.float32) if qtoks else None
            tw = dict(zip(qtoks, qw.tolist())) if qtoks else None
        qvec = self.engine.embed_query(query, term_weights=tw) if tw else self.engine.embed_query(query)
        qtok = self.engine.embed_tokens(query, is_query=True, max_tokens=64)
        t["embed_query"] = (time.perf_counter() - t0) * 1e3

        allowed_list = list(dict.fromkeys(candidates)) if candidates is not None else None
        allowed_idx = self.store.indices(allowed_list) if allowed_list is not None else None
        allowed_set = set(allowed_list) if allowed_list is not None else None
        if allowed_list is not None and allowed_idx.size == 0:
            return RetrievalResult([], t)

        t1 = time.perf_counter()
        bm = [v for v, _ in self.store.bm25(query, K, allowed_list)]
        bmx = [v for v, _ in self.store.bm25(query, K, allowed_list, expand=True)] if s.rrf_weights.bm25x > 0 else []
        t["bm25"] = (time.perf_counter() - t1) * 1e3
        t1 = time.perf_counter()
        bq = [v for v, _ in self.store.bq_search(qvec, K, allowed_idx)]
        t["bq_hamming"] = (time.perf_counter() - t1) * 1e3
        t1 = time.perf_counter()
        mrl = [v for v, _ in self.store.mrl_search(qvec, K, allowed_idx)]
        t["mrl64"] = (time.perf_counter() - t1) * 1e3

        w = s.rrf_weights
        weights = {"bm25": w.bm25, "bm25x": w.bm25x, "bq": w.bq, "mrl64": w.mrl64, "graph": w.graph}
        rankings: dict[str, list[str]] = {"bm25": bm, "bm25x": bmx, "bq": bq, "mrl64": mrl}
        t1 = time.perf_counter()
        pre = rrf_fuse(rankings, weights, s.rrf_k)
        seeds = [v for v, _ in pre[:5]]
        rankings["graph"] = self._graph_rank(seeds, allowed_set, K)
        fused = rrf_fuse(rankings, weights, s.rrf_k)[:K]
        t["rrf_graph"] = (time.perf_counter() - t1) * 1e3

        cand = [v for v, _ in fused]
        rrf_scores = np.array([sc for _, sc in fused], dtype=np.float32)
        t1 = time.perf_counter()
        ms = batch_maxsim(qtok, self.store.token_matrices(cand), weights=qw) if cand else np.zeros(0, np.float32)
        t["colbert_maxsim"] = (time.perf_counter() - t1) * 1e3

        final = self._final(qvec, cand, ms, rrf_scores)

        # stage 2b: 1-hop graph propagation (neighbours of the top hits join the candidate set)
        t1 = time.perf_counter()
        if s.graph_alpha > 0 and cand:
            order = np.argsort(-final, kind="stable")[: s.graph_seed_top]
            present = set(cand)
            extra = []
            for i in order:
                for nb in self.store.neighbors(cand[int(i)]):
                    if nb not in present and (allowed_set is None or nb in allowed_set):
                        present.add(nb)
                        extra.append(nb)
            if extra:
                ms_x = batch_maxsim(qtok, self.store.token_matrices(extra), weights=qw)
                ms_all = np.concatenate([ms, ms_x])
                rrf_all = np.concatenate([rrf_scores, np.zeros(len(extra), dtype=np.float32)])
                cand = cand + extra
                ms, rrf_scores = ms_all, rrf_all
                final = self._final(qvec, cand, ms, rrf_scores)
            final = self._propagate(cand, final, s.graph_alpha)
        t["graph_propagation"] = (time.perf_counter() - t1) * 1e3

        # stage 2d: hierarchical file prior, sibling-caller transfer and structural salience
        t1 = time.perf_counter()
        wf, wc, ws = self._prior_weights()
        if cand and (wf > 0 or wc > 0 or ws > 0):
            cand, final, ms, rrf_scores = self._hierarchical(qtok, qw, qvec, cand, final, ms, rrf_scores, allowed_set)
        t["hierarchical_prior"] = (time.perf_counter() - t1) * 1e3

        # stage 2c: downward callee activation (symptom matches the entry point, bug lives in a helper)
        t1 = time.perf_counter()
        if s.callee_activation and cand:
            cand, final, ms, rrf_scores = self._activate_callees(cand, final, ms, rrf_scores, allowed_set)
        t["callee_activation"] = (time.perf_counter() - t1) * 1e3

        pos = {m: {v: i + 1 for i, v in enumerate(r)} for m, r in rankings.items()}
        hits = [Hit(v, float(final[i]), float(rrf_scores[i]), float(ms[i]),
                    {m: p[v] for m, p in pos.items() if v in p}) for i, v in enumerate(cand)]
        hits.sort(key=lambda h: (-h.score, h.vec_id))
        if k is not None:
            hits = hits[:k]
        t["total"] = (time.perf_counter() - t0) * 1e3
        return RetrievalResult(hits, t)
