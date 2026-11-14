"""SemanticMMU: the orchestrator wiring mipmapping, retrieval, allocation, paging and Negative RAG."""

from __future__ import annotations

import copy
import hashlib
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

from ..config import AegisSettings, get_settings
from ..db.store import AegisStore
from ..tokens import common_prefix_len, count_tokens
from .allocator import Allocation, allocate, compute_budget
from .ast_mipmapper import EXT_LANG, FileMipmap, make_vec_id, mipmap_file, syntax_ok
from .kv_paginator import PageRequest, SemanticPager, order_by_stability
from .negative_rag import NegativeRAG, NegativeWarning, looks_like_failure
from .prompt_codec import (append_text, canonical_hash, find_code, iter_slots, natural_text, prompt_text, replace_found,
                           set_in, strip_cache_control, tool_paths)
from .retriever import HybridRetriever, RetrievalResult


@dataclass
class FoldedFile:
    path: str
    language: str
    file_vec_id: str
    text: str
    skeleton_level: int
    leaf_levels: dict[str, int]
    tokens_original: int
    tokens_folded: int
    syntax_ok: bool | None


@dataclass
class FoldResult:
    files: list[FoldedFile]
    warnings: list[NegativeWarning]
    hits: list[dict]
    tokens_original: int
    tokens_folded: int
    budget: int
    latency_ms: float
    timings_ms: dict[str, float] = field(default_factory=dict)

    @property
    def savings_pct(self) -> float:
        if not self.tokens_original:
            return 0.0
        return 100.0 * (self.tokens_original - self.tokens_folded) / self.tokens_original

    @property
    def context(self) -> str:
        parts = []
        for f in self.files:
            exp = sum(1 for v in f.leaf_levels.values() if v == 0)
            parts.append(f'<aegis_file path="{f.path}" skeleton="L{f.skeleton_level}" l0_units="{exp}">\n{f.text.rstrip()}\n</aegis_file>')
        if self.warnings:
            parts.append(NegativeRAG.render(self.warnings))
        return "\n".join(parts) + "\n"

    def level_histogram(self) -> dict[str, int]:
        h = {"L0": 0, "L1": 0, "L2": 0, "L3": 0}
        for f in self.files:
            for lv in f.leaf_levels.values():
                h[f"L{lv}"] += 1
            h[f"L{f.skeleton_level}"] += 0
        return h

    def to_dict(self) -> dict:
        return {
            "tokens_original": self.tokens_original,
            "tokens_folded": self.tokens_folded,
            "savings_pct": round(self.savings_pct, 2),
            "budget": self.budget,
            "latency_ms": round(self.latency_ms, 3),
            "timings_ms": {k: round(v, 3) for k, v in self.timings_ms.items()},
            "files": [{k: v for k, v in asdict(f).items() if k != "text"} for f in self.files],
            "warnings": [asdict(w) for w in self.warnings],
            "hits": self.hits,
            "context": self.context,
        }


@dataclass
class RewriteResult:
    body: dict
    session_id: str
    tokens_original: int
    tokens_rewritten: int
    kv_hit_rate: float
    latency_ms: float
    folded_blocks: int
    promotions: list[str]
    warnings: list[int]
    reset: bool

    @property
    def savings_pct(self) -> float:
        if not self.tokens_original:
            return 0.0
        return 100.0 * (self.tokens_original - self.tokens_rewritten) / self.tokens_original

    def headers(self) -> dict[str, str]:
        return {
            "X-Aegis-Tokens-Original": str(self.tokens_original),
            "X-Aegis-Tokens-Compressed": str(self.tokens_rewritten),
            "X-Aegis-Savings-Pct": f"{self.savings_pct:.2f}",
            "X-Aegis-KV-Hit-Rate": f"{self.kv_hit_rate:.4f}",
            "X-Aegis-Latency-Ms": f"{self.latency_ms:.3f}",
            "X-Aegis-Session": self.session_id,
        }


class SemanticMMU:
    def __init__(self, store: AegisStore | None = None, settings: AegisSettings | None = None):
        self.settings = settings or (store.settings if store is not None else get_settings())
        self.store = store or AegisStore(self.settings.resolved_db_path(), settings=self.settings)
        self.engine = self.store.engine
        self.retriever = HybridRetriever(self.store, self.settings)
        self.negative = NegativeRAG(self.store)
        self.pager = SemanticPager(self.store)
        self._mm_cache: dict[str, FileMipmap] = {}
        self._last_prompts: dict[str, tuple[str, int]] = {}  # session -> (last rewritten prompt, tokens)
        self._tok_cache: dict[tuple, tuple[str, int]] = {}

    # ------------------------------------------------------------------ ingest
    def ingest_many(self, files: Sequence, persist: bool = True) -> list[FileMipmap]:
        """Mipmap every file, then index all new units in one batch (shared 1-hop call context)."""
        fms = [self._mipmap(f[0], f[1], f[2] if len(f) > 2 else None) for f in files]
        self.store.upsert_files(fms, persist_mipmap=persist)
        return fms

    def _mipmap(self, path: str, source: str, language: str | None = None) -> FileMipmap:
        key = hashlib.blake2b(f"{path}\0{language}\0{source}".encode(), digest_size=8).hexdigest()
        fm = self._mm_cache.get(key)
        if fm is None:
            fid = make_vec_id(path, "<file>", source.replace("\r\n", "\n"))
            fm = self.store.get_file(fid) if self.store.has_file(fid) else None
            if fm is None or (language and fm.language != language):
                fm = mipmap_file(path, source, language, max_depth=self.settings.l1_max_depth, l1_mode=self.settings.l1_mode,
                                 exact_tokens=not self.settings.cold_token_estimate)
            if len(self._mm_cache) > 4096:
                self._mm_cache.clear()
            self._mm_cache[key] = fm
        return fm

    def ingest(self, path: str, source: str, language: str | None = None, persist: bool = True) -> FileMipmap:
        """Content-addressed ingest: in-process cache -> persisted mipmap (64-bit blake2b) -> parse."""
        fm = self._mipmap(path, source, language)
        self.store.upsert_file(fm, persist_mipmap=persist)
        return fm

    def index_paths(self, root: str | Path, exts: Iterable[str] | None = None, max_bytes: int = 400_000) -> dict:
        root = Path(root)
        exts = set(exts or EXT_LANG)
        files, units, skipped = 0, 0, 0
        batch: list[tuple[str, str]] = []
        t0 = time.perf_counter()

        def flush() -> None:
            nonlocal units
            if batch:
                units += sum(len(fm.units) for fm in self.ingest_many(batch))
                batch.clear()

        for p in sorted(root.rglob("*")):
            if not p.is_file() or p.suffix not in exts or any(part.startswith(".") or part in ("node_modules", "target", "build", "dist", "__pycache__", ".venv") for part in p.relative_to(root).parts):
                continue
            if p.stat().st_size > max_bytes:
                skipped += 1
                continue
            try:
                src = p.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                skipped += 1
                continue
            batch.append((str(p.relative_to(root)), src))
            files += 1
            if len(batch) >= 64:
                flush()
        flush()
        return {"files": files, "units": units, "skipped": skipped, "seconds": round(time.perf_counter() - t0, 3)}

    # ------------------------------------------------------------------ fold
    def fold(self, files: Sequence[tuple[str, str] | tuple[str, str, str | None]], query: str, budget: int | None = None,
             proposed_code: str = "", record: bool = True, verify: bool = False) -> FoldResult:
        """Fold ``files`` for ``query``. ``verify=True`` additionally re-parses every rendered file
        (``FoldedFile.syntax_ok``); it is off in the serving path."""
        t0 = time.perf_counter()
        timings: dict[str, float] = {}
        fms = self.ingest_many(files)
        timings["ingest"] = (time.perf_counter() - t0) * 1e3
        res, alloc = self._allocate(fms, query, budget)
        timings.update({f"retrieve.{k}": v for k, v in res.timings_ms.items()})
        t1 = time.perf_counter()
        out_files = self._render(fms, alloc, verify)
        timings["render"] = (time.perf_counter() - t1) * 1e3
        t1 = time.perf_counter()
        warnings = self.negative.check(query, proposed_code) if query or proposed_code else []
        timings["negative_rag"] = (time.perf_counter() - t1) * 1e3
        orig = sum(f.tokens_original for f in out_files)
        folded = sum(f.tokens_folded for f in out_files)
        lat = (time.perf_counter() - t0) * 1e3
        timings["total"] = lat
        hits = [{"vec_id": h.vec_id, "score": round(h.score, 4), "maxsim": round(h.maxsim, 4), "rrf": round(h.rrf, 5),
                 "ranks": h.ranks} for h in res.hits[:10]]
        for h in hits:
            u = self.store.get_unit(h["vec_id"])
            if u:
                h.update(path=u.path, qualname=u.qualname, level=alloc.leaf_levels.get(u.vec_id))
        result = FoldResult(out_files, warnings, hits, orig, folded, alloc.budget, lat, timings)
        if record:
            self.store.log_event("fold", tokens_original=orig, tokens_compressed=folded, latency_ms=lat,
                                 payload={"query": query[:200], "levels": result.level_histogram(),
                                          "files": [{"path": f.path, "skeleton": f.skeleton_level,
                                                     "l0": sum(1 for v in f.leaf_levels.values() if v == 0),
                                                     "l1": sum(1 for v in f.leaf_levels.values() if v == 1),
                                                     "units": len(next(x for x in fms if x.file_vec_id == f.file_vec_id).units),
                                                     "orig": f.tokens_original, "folded": f.tokens_folded} for f in out_files],
                                          "warnings": [w.id for w in warnings]})
        return result

    def _allocate(self, fms: Sequence[FileMipmap], query: str, budget: int | None,
                  extra_candidates: Sequence[str] = ()) -> tuple[RetrievalResult, Allocation]:
        cand = [u.vec_id for f in fms for u in f.units] + list(extra_candidates)
        res = self.retriever.search(query, cand) if query.strip() else RetrievalResult([], {})
        rel = res.relevance(self.settings.relevance_temperature)
        orig = sum(f.tokens_source for f in fms)
        b = budget if budget is not None else compute_budget(orig, self.settings)
        return res, allocate(fms, rel, b, callees=self.store.callees if self.settings.co_promote_callees else None)

    def _render(self, fms: Sequence[FileMipmap], alloc: Allocation, verify: bool = False) -> list[FoldedFile]:
        out = []
        for fm in fms:
            levels = {u.vec_id: alloc.leaf_levels[u.vec_id] for u in fm.units if u.vec_id in alloc.leaf_levels}
            skel = alloc.file_levels.get(fm.file_vec_id, 3)
            text = fm.render(levels, skel)
            out.append(FoldedFile(fm.path, fm.language, fm.file_vec_id, text, skel, levels, fm.exact_tokens(),
                                  count_tokens(text), syntax_ok(text, fm.language) if verify else None))
        return out

    # ------------------------------------------------------------------ expand
    def expand(self, vec_id: str) -> dict | None:
        u = self.store.get_unit(vec_id)
        if u is not None:
            return {"vec_id": u.vec_id, "path": u.path, "qualname": u.qualname, "kind": u.kind, "language": u.language,
                    "start_line": u.start_line, "end_line": u.end_line, "tokens": u.tokens[0], "source": u.l0}
        fm = self.store.get_file(vec_id)
        if fm is not None:
            return {"vec_id": vec_id, "path": fm.path, "qualname": "<file>", "kind": "file", "language": fm.language,
                    "start_line": 1, "end_line": len(fm.lines), "tokens": fm.tokens_source, "source": fm.source}
        return None

    # ------------------------------------------------------------------ paged context block (MCP / library)
    def context_block(self, session_id: str, files: Sequence[tuple[str, str]], query: str, budget: int | None = None):
        """Spec-literal append-only context block.

        File skeletons are the stable architectural prefix: their level is fixed the first time a
        file is paged in. Later turns only append new/promoted leaf pages (highest relevance first)
        and Negative-RAG warnings; compaction is tail-only (see :class:`SemanticPager`).

        ``kv_policy="stable"`` (default): turn 1 pins an L2 map of every file; on later turns at most
        ``kv_max_new_l0`` new functions are appended at L0 (further ones at L1) and the whole delta
        is capped at ``kv_delta_frac`` of the resident block, which bounds the per-turn prefix hit
        from below by ~1 / (1 + kv_delta_frac). ``"adaptive"``: per-turn delta capped only by
        ``delta_budget``.
        """
        st = self.settings
        stable = st.kv_policy == "stable"
        fms = self.ingest_many(files)
        res, alloc = self._allocate(fms, query, budget)
        resident = self.pager.resident(session_id)
        first = not resident
        rel = res.relevance(st.relevance_temperature)
        reqs: list[PageRequest] = []
        for fm in fms:
            key = "file:" + fm.file_vec_id
            default_skel = 2 if (stable and first) else alloc.file_levels.get(fm.file_vec_id, 3)
            skel = resident.get(key, default_skel)
            reqs.append(PageRequest(key, skel, fm.skeleton[skel], "skeleton", fm.path, 0))
        leaves: list[tuple[float, object, object, int]] = []
        for fm in fms:
            for u in fm.units:
                lv = alloc.leaf_levels.get(u.vec_id)
                if lv is None or lv > 1 or resident.get(u.vec_id, 9) <= lv:
                    continue
                leaves.append((rel.get(u.vec_id, 0.0), fm, u, lv))
        leaves.sort(key=lambda x: -x[0])
        limit = st.delta_budget
        if stable and not first:
            limit = min(limit, int(st.kv_delta_frac * self.pager.resident_tokens(session_id)))
        spent, new_l0 = 0, 0
        for _, fm, u, lv in leaves:
            if stable and not first and lv == 0 and new_l0 >= st.kv_max_new_l0:
                if resident.get(u.vec_id, 9) <= 1:
                    continue
                lv = 1  # beyond the per-turn L0 cap: supporting functions arrive as L1 skeletons
            r = PageRequest(u.vec_id, lv, fm.render_page(u, lv), "leaf", fm.path, u.start_line)
            cost = count_tokens(r.text)
            if not first and spent + cost > limit:
                continue
            spent += cost
            new_l0 += lv == 0
            reqs.append(r)
        for w in self.negative.check(query):
            reqs.append(PageRequest(f"warn:{w.id}", 0, w.render(), "warning", "", 0))
        return self.pager.page(session_id, order_by_stability(reqs), max(alloc.budget, self.settings.token_budget))

    # ------------------------------------------------------------------ conversation rewriting (proxy)
    def rewrite_request(self, body: dict, fmt: str = "anthropic", session_id: str | None = None) -> RewriteResult:
        """Fold code in an Anthropic/OpenAI chat request with append-only, turn-anchored paging.

        Every user message is rewritten exactly once (the turn it first appears) and its rewritten
        form is stored. Later turns substitute the stored bytes, so the whole previous prompt is a
        prefix of the new one; promotions and Negative-RAG warnings are appended to the newest user
        message only.
        """
        t0 = time.perf_counter()
        s = self.settings
        msgs: list[dict] = body.get("messages") or []
        sid = session_id or self._session_id(body, fmt)
        state = self.store.load_session("conv:" + sid) or {}
        anchors: list[dict] = state.get("anchors", [])
        reset = False
        if any(a["i"] >= len(msgs) or canonical_hash(msgs[a["i"]]) != a["h"] for a in anchors):
            reset = True
            state = {"epoch": state.get("epoch", 0) + 1}
            anchors = []
        anchored = {a["i"]: a for a in anchors}
        resident: dict[str, int] = state.get("resident", {})
        warned: list[int] = state.get("warned", [])
        tpaths = tool_paths(msgs, fmt)
        user_roles = ("user", "tool") if fmt == "openai" else ("user",)
        new_idx = [i for i, m in enumerate(msgs) if i not in anchored and m.get("role") in user_roles]

        # 1. extract code from new user messages
        located: list[tuple[int, object, list]] = []  # (msg index, slot, found list)
        fms: dict[str, FileMipmap] = {}
        block_tokens = 0
        for i in new_idx:
            for slot in iter_slots(msgs[i], fmt, tpaths):
                found = find_code(slot)
                if found:
                    located.append((i, slot, found))
        found_all = [f for _, _, found in located for f in found]
        for fm in self.ingest_many([(f.path, f.code, f.language) for f in found_all], persist=False):
            if fm.file_vec_id not in fms:
                fms[fm.file_vec_id] = fm
                block_tokens += fm.tokens_source
        query = self._query(msgs, fmt)

        # 2. retrieval + allocation over new code (resident units are candidates for promotion)
        extra = [v for v in resident if self.store.has_unit(v)]
        if s.repo_page_in:
            extra = self.store.all_vec_ids()
        res, alloc = self._allocate(list(fms.values()), query, None, extra_candidates=extra)
        if not fms:
            alloc = Allocation(budget=0)

        # 3. rewrite new messages in place
        rewritten: dict[int, dict] = {}
        n_blocks = 0
        by_msg: dict[int, list] = {}
        for i, slot, found in located:
            by_msg.setdefault(i, []).append((slot, found))
        for i, items in by_msg.items():
            m = copy.deepcopy(msgs[i])
            for slot, found in items:
                text = slot.text
                for f in sorted(found, key=lambda x: -x.start):
                    fm = self.ingest(f.path, f.code, f.language, persist=False)
                    levels = {u.vec_id: alloc.leaf_levels[u.vec_id] for u in fm.units if u.vec_id in alloc.leaf_levels}
                    skel = alloc.file_levels.get(fm.file_vec_id, 2)
                    rendered = fm.render(levels, skel)
                    if rendered == fm.source:
                        for u in fm.units:
                            resident[u.vec_id] = 0
                        continue
                    approx = "~" if getattr(fm, "tokens_estimated", False) else ""
                    note = (f"[aegis-mmu: holographic view of {fm.path} ({approx}{fm.tokens_source}→{count_tokens(rendered)} tokens); "
                            f"folded regions can be expanded by vec_id]")
                    text = replace_found(text, f, rendered, note)
                    n_blocks += 1
                    for u in fm.units:
                        lv = levels.get(u.vec_id, skel)
                        resident[u.vec_id] = min(resident.get(u.vec_id, 9), lv)
                set_in(m, slot.path, text)
            rewritten[i] = m

        # 4. promotions / page-ins + negative memory, appended to the newest user message
        promotions: list[str] = []
        delta_pages: list[str] = []
        spent = 0
        new_vecs = {u.vec_id for fm in fms.values() for u in fm.units}
        for h in res.hits[: s.page_in_top_k]:
            if h.vec_id in new_vecs or resident.get(h.vec_id, 9) == 0:
                continue
            if res.hits and h.score < 0.5 * res.hits[0].score:
                continue
            u = self.store.get_unit(h.vec_id)
            fm = self.store.get_file(u.file_vec_id) if u else None
            unit = fm.unit(h.vec_id) if fm else None
            if unit is None:
                continue
            page = fm.render_page(unit, 0)
            cost = count_tokens(page)
            if spent + cost > s.delta_budget:
                continue
            spent += cost
            delta_pages.append(page)
            promotions.append(h.vec_id)
            resident[h.vec_id] = 0
        code_probe = "\n".join(f.code for _, _, found in located for f in found)[-4000:]
        warnings = [w for w in self.negative.check(query, code_probe) if w.id not in warned] if query else []
        self._capture_failures(msgs, new_idx, fmt)
        last_user = max((i for i, m in enumerate(msgs) if m.get("role") in user_roles), default=None)
        if (delta_pages or warnings) and last_user is not None and last_user in new_idx:
            parts = [f'<aegis_context_delta turn="{state.get("turn", 0) + 1}">'] + [p.rstrip() for p in delta_pages]
            parts += [w.render() for w in warnings]
            parts.append("</aegis_context_delta>")
            rewritten[last_user] = append_text(rewritten.get(last_user, msgs[last_user]), fmt, "\n".join(parts))
            warned += [w.id for w in warnings]

        # 5. assemble the outgoing body (anchored messages replay their stored bytes)
        out_msgs = []
        for i, m in enumerate(msgs):
            if i in anchored and anchored[i].get("c") is not None:
                out_msgs.append(_reapply_cache_control(m, anchored[i]["c"]))
            elif i in rewritten:
                out_msgs.append(_reapply_cache_control(m, strip_cache_control(rewritten[i])))
            else:
                out_msgs.append(m)
        for i in new_idx:
            anchors.append({"i": i, "h": canonical_hash(msgs[i]),
                            "c": strip_cache_control(rewritten[i]) if i in rewritten else None})
        new_body = dict(body)
        new_body["messages"] = out_msgs

        orig_prompt = prompt_text(body, fmt)
        new_prompt = prompt_text(new_body, fmt)
        prev = None if reset else self._last_prompts.get(sid)
        if s.cold_token_estimate:  # header telemetry only; the rewritten prompt (KV metric) stays exact
            from ..tokens import estimate_tokens

            tok_orig = estimate_tokens(orig_prompt)
        else:
            tok_orig = self._incremental_tokens(("orig", sid), orig_prompt)
        tok_new = self._incremental_tokens(("new", sid), new_prompt)
        if prev is None:
            kv = 0.0
        else:
            prev_text, prev_tokens = prev
            lcp = common_prefix_len(prev_text, new_prompt)
            lcp_tokens = prev_tokens if lcp == len(prev_text) else count_tokens(new_prompt[:lcp])
            kv = min(1.0, lcp_tokens / tok_new) if tok_new else 0.0
        self._last_prompts[sid] = (new_prompt, tok_new)
        if len(self._last_prompts) > 256:
            self._last_prompts.pop(next(iter(self._last_prompts)))
        state.pop("last_prompt", None)
        state.update(anchors=anchors, resident=resident, warned=warned,
                     turn=state.get("turn", 0) + 1)
        self.store.save_session("conv:" + sid, state)
        lat = (time.perf_counter() - t0) * 1e3
        self.store.log_event("proxy", session_id=sid, tokens_original=tok_orig, tokens_compressed=tok_new,
                             kv_hit_rate=kv, latency_ms=lat,
                             payload={"fmt": fmt, "blocks": n_blocks, "promotions": promotions,
                                      "warnings": [w.id for w in warnings], "reset": reset})
        return RewriteResult(new_body, sid, tok_orig, tok_new, kv, lat, n_blocks, promotions,
                             [w.id for w in warnings], reset)

    def _incremental_tokens(self, key: tuple, text: str) -> int:
        """Token count of an append-only prompt: only the appended suffix is tokenised (BPE boundary
        effects at the seam are at most a token)."""
        prev = self._tok_cache.get(key)
        if prev is not None and text.startswith(prev[0]):
            n = prev[1] + count_tokens(text[len(prev[0]):])
        else:
            n = count_tokens(text)
        self._tok_cache[key] = (text, n)
        if len(self._tok_cache) > 512:
            self._tok_cache.pop(next(iter(self._tok_cache)))
        return n

    @staticmethod
    def _session_id(body: dict, fmt: str) -> str:
        msgs = body.get("messages") or []
        first_user = next((m for m in msgs if m.get("role") == "user"), {})
        sysd = body.get("system") if fmt == "anthropic" else [m for m in msgs if m.get("role") in ("system", "developer")]
        seed = f"{fmt}\0{canonical_hash({'s': sysd or ''})}\0{canonical_hash(first_user)}"
        return hashlib.blake2b(seed.encode(), digest_size=8).hexdigest()

    @staticmethod
    def _query(msgs: list[dict], fmt: str) -> str:
        parts: list[str] = []
        for m in reversed(msgs):
            if m.get("role") == "user":
                t = natural_text(m, fmt)
                if t:
                    parts.append(t)
                    break
        for m in reversed(msgs):
            if m.get("role") == "assistant":
                t = natural_text(m, fmt)
                if t:
                    parts.append(t[-600:])
                break
        return "\n".join(parts)[:4000]

    def _capture_failures(self, msgs: list[dict], new_idx: list[int], fmt: str) -> None:
        """Auto-record failed tool executions (is_error tool_results with tracebacks) into Negative RAG."""
        if fmt != "anthropic":
            return
        for i in new_idx:
            c = msgs[i].get("content")
            if not isinstance(c, list):
                continue
            for b in c:
                if not (isinstance(b, dict) and b.get("type") == "tool_result" and b.get("is_error")):
                    continue
                err = b.get("content")
                err = err if isinstance(err, str) else "\n".join(x.get("text", "") for x in (err or []) if isinstance(x, dict))
                if not looks_like_failure(err):
                    continue
                attempt = ""
                if i > 0 and isinstance(msgs[i - 1].get("content"), list):
                    for tb in msgs[i - 1]["content"]:
                        if isinstance(tb, dict) and tb.get("type") == "tool_use" and tb.get("id") == b.get("tool_use_id"):
                            attempt = str(tb.get("input", ""))[:2000]
                task = self._query(msgs[: i], fmt)[:500]
                self.negative.record(task=task, code=attempt, error=err[-1500:], kind="exception")


def _reapply_cache_control(original: dict, stored: dict) -> dict:
    """Copy current cache_control markers of ``original`` onto the stored rewritten message."""
    out = copy.deepcopy(stored)
    oc, sc = original.get("content"), out.get("content")
    if not (isinstance(oc, list) and isinstance(sc, list)) or not sc:
        return out
    for j, b in enumerate(oc):
        if isinstance(b, dict) and "cache_control" in b:
            target = j if j < len(oc) - 1 else len(sc) - 1
            if 0 <= target < len(sc) and isinstance(sc[target], dict):
                sc[target]["cache_control"] = b["cache_control"]
    return out
