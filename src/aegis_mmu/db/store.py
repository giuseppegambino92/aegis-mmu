"""Hybrid storage engine.

SQLite is the durable source of truth (WAL, FTS5 BM25, vector blobs). Dense, Matryoshka-64 and
1-bit vectors are mirrored into contiguous NumPy arrays so the hot search paths (Hamming XOR +
popcount, MRL-64 dot products) run in microseconds without touching SQLite.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import threading
import time
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np

from ..embeddings.colbert import decode_token_matrix, encode_token_matrix
from ..embeddings.lexicon import STOPWORDS, expand_term, stem, subword_tokens
from ..embeddings.matryoshka import binary_quantize, cosine_topk, hamming_topk, mrl_slice
from .schema import connect, migrate

_COMMON_NAMES = frozenset({"get", "set", "run", "main", "init", "new", "update", "load", "save", "apply",
                           "build", "create", "close", "open", "start", "stop", "reset", "clear", "handle", "process"})


@dataclass
class UnitRecord:
    vec_id: str
    file_vec_id: str
    path: str
    language: str
    qualname: str
    kind: str
    start_line: int
    end_line: int
    l0: str
    l1: str
    l2: str
    l3: str
    tokens: dict[int, int]
    role: str

    def level_text(self, level: int) -> str:
        return (self.l0, self.l1, self.l2, self.l3)[level]


class AegisStore:
    def __init__(self, path: str = ":memory:", engine=None, settings=None):
        from ..config import get_settings
        from ..embeddings.providers import get_engine

        self.settings = settings or get_settings()
        self.engine = engine or get_engine(self.settings)
        self.path = path
        self.conn = connect(path, autocheckpoint=False)
        migrate(self.conn)
        self.lock = threading.RLock()
        self.mrl_dim = min(self.settings.mrl_dims) if self.settings.mrl_dims else 64
        self._file_cache: dict[str, Any] = {}
        self._load_mirror()
        self._sessions: dict[str, dict] = {}
        self._writer: _WriteBehind | None = None
        if path != ":memory:" and not path.startswith("file::memory:"):
            self._writer = _WriteBehind(path)

    # ------------------------------------------------------------------ mirror
    def _load_mirror(self) -> None:
        rows = self.conn.execute("SELECT id, vec_id, qualname, file_vec_id, path, idents, calls, l0, emb, mrl64, bq, colbert FROM units ORDER BY id").fetchall()
        dim, td = self.engine.dim, self.engine.token_dim
        n = len(rows)
        self._n = 0
        cap = max(256, 1 << (n - 1).bit_length()) if n else 256
        self._emb_buf = np.zeros((cap, dim), dtype=np.float32)
        self._mrl_buf = np.zeros((cap, self.mrl_dim), dtype=np.float32)
        self._bq_buf = np.zeros((cap, dim // 8), dtype=np.uint8)
        self._rowids: list[int] = []
        self._vec_ids: list[str] = []
        self._idx: dict[str, int] = {}
        self._tok: list[np.ndarray] = []
        self._names: dict[str, set[str]] = {}
        self._ident_index: dict[str, set[str]] = {}  # identifier -> units mentioning it
        self._unit_file: dict[str, str] = {}
        self._unit_name: dict[str, str] = {}
        self._unit_qual: dict[str, str] = {}
        self._calls_raw: dict[str, list[str]] = {}  # vec_id -> called names (resolved lazily)
        self._salience: dict[str, float] = {}  # vec_id -> structural bug-proneness prior
        self._callees: dict[str, set[str]] = {}  # vec_id -> callee vec_ids (same file / imported module, live)
        self._file_meta: dict[str, tuple[str, str]] = {}  # file_vec_id -> (module stem, import text)
        self._df: dict[str, int] = {}  # subword -> number of units containing it (for query IDF)
        self._latest: dict[str, str] = {}  # path -> file_vec_id of its most recent version
        self._file_paths: dict[str, str] = {}  # file_vec_id -> path
        for r in self.conn.execute("SELECT file_vec_id, path, language, source FROM files ORDER BY created_at, rowid"):
            self._file_meta[r["file_vec_id"]] = (_module_stem(r["path"]), _import_text(r["source"]))
            self._latest[r["path"]] = r["file_vec_id"]
            self._file_paths[r["file_vec_id"]] = r["path"]
        if n:
            self._append_vectors(np.array([np.frombuffer(r["emb"], dtype=np.float32) for r in rows]),
                                 np.array([np.frombuffer(r["mrl64"], dtype=np.float32) for r in rows]),
                                 np.array([np.frombuffer(r["bq"], dtype=np.uint8) for r in rows]))
        for r in rows:
            self._count_df(f"{r['path']} {r['qualname']}\n{r['l0']}")
        from ..mmu.ast_mipmapper import structural_salience

        for r in rows:
            self._unit_qual[r["vec_id"]] = r["qualname"]
            self._salience[r["vec_id"]] = structural_salience(r["l0"])
            self._calls_raw[r["vec_id"]] = (r["calls"] or "").split()
        for i, r in enumerate(rows):
            self._register(r["vec_id"], r["id"], r["qualname"].rsplit(".", 1)[-1], r["file_vec_id"],
                           (r["idents"] or "").split(), decode_token_matrix(r["colbert"], td), i)
        self._adj: dict[str, set[str]] = {}
        for e in self.conn.execute("SELECT src, dst FROM edges"):
            self._adj.setdefault(e["src"], set()).add(e["dst"])
            self._adj.setdefault(e["dst"], set()).add(e["src"])
        neg = self.conn.execute("SELECT id, emb FROM negative_memory ORDER BY id").fetchall()
        self._neg_ids = [r["id"] for r in neg]
        self._neg_emb = (np.array([np.frombuffer(r["emb"], dtype=np.float32) for r in neg], dtype=np.float32)
                         if neg else np.zeros((0, 3 * dim), dtype=np.float32))

    # contiguous views over amortised-growth buffers (no O(N) vstack per insert)
    @property
    def _emb(self) -> np.ndarray:
        return self._emb_buf[: self._n]

    @property
    def _mrl(self) -> np.ndarray:
        return self._mrl_buf[: self._n]

    @property
    def _bq(self) -> np.ndarray:
        return self._bq_buf[: self._n]

    def _append_vectors(self, emb: np.ndarray, mrl: np.ndarray, bq: np.ndarray) -> None:
        need = self._n + emb.shape[0]
        if need > self._emb_buf.shape[0]:
            cap = 1 << (need - 1).bit_length()
            for name in ("_emb_buf", "_mrl_buf", "_bq_buf"):
                old = getattr(self, name)
                new = np.zeros((cap, old.shape[1]), dtype=old.dtype)
                new[: self._n] = old[: self._n]
                setattr(self, name, new)
        self._emb_buf[self._n:need] = emb
        self._mrl_buf[self._n:need] = mrl
        self._bq_buf[self._n:need] = bq
        self._n = need

    def _register(self, vec_id: str, rowid: int, name: str, file_vec_id: str, idents: list[str], tok: np.ndarray, i: int) -> None:
        self._rowids.append(rowid)
        self._vec_ids.append(vec_id)
        self._idx[vec_id] = i
        self._tok.append(tok)
        self._names.setdefault(name, set()).add(vec_id)
        self._unit_file[vec_id] = file_vec_id
        self._unit_name[vec_id] = name
        for ident in idents:
            self._ident_index.setdefault(ident, set()).add(vec_id)

    def _count_df(self, text: str) -> None:
        for t in set(subword_tokens(text)):
            self._df[t] = self._df.get(t, 0) + 1

    def idf(self, tokens) -> list[float]:
        """Smoothed IDF ``log((N + 1) / (df + 0.5))`` over indexed units, floored at 0.05."""
        n = len(self._vec_ids)
        return [max(0.05, math.log((n + 1) / (self._df.get(t, 0) + 0.5))) for t in tokens]

    @property
    def size(self) -> int:
        return len(self._vec_ids)

    def has_unit(self, vec_id: str) -> bool:
        return vec_id in self._idx

    def has_file(self, file_vec_id: str) -> bool:
        return file_vec_id in self._file_meta

    # ------------------------------------------------------------------ ingest
    def upsert_file(self, fm, persist_mipmap: bool = True) -> list[str]:
        """Store one :class:`FileMipmap` (see :meth:`upsert_files`)."""
        return self.upsert_files([fm], persist_mipmap)

    def upsert_files(self, fms: Sequence, persist_mipmap: bool = True) -> list[str]:
        """Store mipmaps and index their new units in ONE transaction: batched embedding, executemany.

        With ``settings.composite_index`` every unit is indexed (dense vector, FTS5 body, ColBERT
        tokens) from its composite document, including its 1-hop ``calls`` / ``called_by`` names
        computed across the store *and* the batch, so callers in sibling files count.
        ``persist_mipmap=False`` (proxy hot path) skips serialising the parsed mipmap."""
        with self.lock:
            todo: list[tuple] = []
            unknown = []
            for fm in fms:
                self._file_cache[fm.file_vec_id] = fm
                known = fm.file_vec_id in self._file_meta
                if not known:
                    unknown.append(fm)
                    self._file_meta[fm.file_vec_id] = (_module_stem(fm.path), _import_text(fm.source))
                self._latest[fm.path] = fm.file_vec_id
                self._file_paths[fm.file_vec_id] = fm.path
                new_units = [u for u in fm.units if u.vec_id not in self._idx]
                if new_units:
                    todo.append((fm, new_units))
            if not unknown and not todo:
                return []
            self.conn.execute("BEGIN")
            try:
                if unknown:
                    self.conn.executemany(
                        "INSERT OR IGNORE INTO files(file_vec_id, path, language, source, tokens, mipmap) VALUES (?,?,?,?,?,?)",
                        [(fm.file_vec_id, fm.path, fm.language, fm.source, fm.tokens_source,
                          _pack_mipmap(fm) if persist_mipmap else None) for fm in unknown])
                texts = self._index_texts(todo)
                for (fm, units), tx in zip(todo, texts):
                    self._insert_units(fm.file_vec_id, units, tx)
                self.conn.execute("COMMIT")
            except Exception:
                self.conn.execute("ROLLBACK")
                raise
            return [u.vec_id for _, units in todo for u in units]

    def _index_texts(self, todo: Sequence[tuple]) -> list[list[str]]:
        if not getattr(self.settings, "composite_index", False):
            return [[u.embed_text() for u in units] for _, units in todo]
        batch_names: dict[str, list[tuple[str, str]]] = {}  # name -> [(qualname, file_vec_id)]
        batch_mentions: dict[str, list[tuple[str, str]]] = {}  # identifier -> [(qualname, file_vec_id)]
        for fm, _ in todo:
            for u in fm.units:
                if u.kind in ("function", "method", "type"):
                    batch_names.setdefault(u.name, []).append((u.qualname, fm.file_vec_id))
                for ident in u.identifiers:
                    batch_mentions.setdefault(ident, []).append((u.qualname, fm.file_vec_id))
        out = []
        for fm, units in todo:
            fid = fm.file_vec_id
            texts = []
            for u in units:
                calls: dict[str, None] = {}
                for ident in u.identifiers:
                    if len(ident) < 4 or ident in _COMMON_NAMES or ident == u.name:
                        continue
                    targets = [(q, f) for q, f in batch_names.get(ident, ())]
                    targets += [(self._unit_qual.get(v, ident), self._unit_file.get(v, "")) for v in self._names.get(ident, ())
                                if self._is_live(v)]
                    if any(self._may_reference(fid, f) for _, f in targets):
                        calls.setdefault(ident, None)
                called_by: dict[str, None] = {}
                if len(u.name) >= 4 and u.name not in _COMMON_NAMES and not u.name.startswith("<"):
                    for q, f in batch_mentions.get(u.name, ()):
                        if q != u.qualname and self._may_reference(f, fid):
                            called_by.setdefault(q, None)
                    for v in self._ident_index.get(u.name, ()):
                        if self._is_live(v) and self._may_reference(self._unit_file.get(v, ""), fid):
                            called_by.setdefault(self._unit_qual.get(v, ""), None)
                texts.append(u.composite_text(list(calls), [c for c in called_by if c]))
            out.append(texts)
        return out

    def _insert_units(self, file_vec_id: str, units: Sequence, texts: list[str] | None = None) -> None:
        texts = texts or [u.embed_text() for u in units]
        embs = self.engine.embed(texts)  # one batched (N_units x D) projection
        mrl = mrl_slice(embs, self.mrl_dim)
        bq = binary_quantize(embs)
        toks = self.engine.embed_tokens_batch(texts, max_tokens=self.settings.colbert_max_doc_tokens) \
            if hasattr(self.engine, "embed_tokens_batch") else \
            [self.engine.embed_tokens(t, max_tokens=self.settings.colbert_max_doc_tokens) for t in texts]
        toks = [t.astype(np.float16) for t in toks]
        first = (self.conn.execute("SELECT COALESCE(MAX(id), 0) FROM units").fetchone()[0] or 0) + 1
        rowids = list(range(first, first + len(units)))
        self.conn.executemany(
            """INSERT INTO units(id, vec_id, file_vec_id, path, language, qualname, kind, start_line, end_line,
               l0, l1, l2, l3, tok_l0, tok_l1, tok_l2, tok_l3, role, idents, calls, emb, mrl64, bq, colbert, colbert_n, engine)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            [(rid, u.vec_id, file_vec_id, u.path, u.language, u.qualname, u.kind, u.start_line, u.end_line,
              u.l0, u.l1, u.l2, u.l3, u.tokens[0], u.tokens[1], u.tokens[2], u.tokens[3], u.role, " ".join(u.identifiers),
              " ".join(getattr(u, "calls", []) or []),
              e.tobytes(), m.astype(np.float32).tobytes(), b.tobytes(), encode_token_matrix(t), t.shape[0], self.engine.name)
             for rid, u, e, m, b, t in zip(rowids, units, embs, mrl, bq, toks)],
        )
        self.conn.executemany(
            "INSERT INTO units_fts(rowid, path, qualname, body) VALUES (?,?,?,?)",
            [(rid, " ".join(subword_tokens(u.path)), " ".join(subword_tokens(u.qualname)), " ".join(_fts_terms(tx)))
             for rid, u, tx in zip(rowids, units, texts)],
        )
        for u in units:
            self._count_df(f"{u.path} {u.qualname}\n{u.l0}")
        base = self._n
        self._append_vectors(embs, mrl, bq)
        for k, (rid, u, t) in enumerate(zip(rowids, units, toks)):
            self._register(u.vec_id, rid, u.name, file_vec_id, u.identifiers, t, base + k)
            self._unit_qual[u.vec_id] = u.qualname
            self._calls_raw[u.vec_id] = list(getattr(u, "calls", []) or [])
            self._salience[u.vec_id] = _salience_of(u.l0) if u.kind in ("function", "method") else 0.0
        self._callees_dirty = True
        self._indeg_dirty = True
        self._link(file_vec_id, units)

    def _is_live(self, vec_id: str) -> bool:
        f = self._unit_file.get(vec_id, "")
        path = self._file_paths.get(f)
        return path is None or self._latest.get(path) == f

    def _may_reference(self, src_file: str, dst_file: str) -> bool:
        """Edges are kept within a file or towards a module the source file imports."""
        if src_file == dst_file:
            return True
        stem, _ = self._file_meta.get(dst_file, ("", ""))
        _, imports = self._file_meta.get(src_file, ("", ""))
        return bool(stem) and stem in imports

    def _link(self, file_vec_id: str, units: Sequence) -> None:
        """1-hop reference graph: A -> B when A mentions B's distinctive name (same file or imported module)."""
        edges: list[tuple[str, str]] = []
        new_ids = {u.vec_id for u in units}
        # only the latest version of every path participates (superseded versions stay stored, unlinked)
        for u in units:
            for ident in u.identifiers:
                if len(ident) < 4 or ident in _COMMON_NAMES or ident == u.name:
                    continue
                for dst in self._names.get(ident, ()):
                    if dst != u.vec_id and self._is_live(dst) and self._may_reference(file_vec_id, self._unit_file.get(dst, "")):
                        edges.append((u.vec_id, dst))
        for u in units:  # reverse direction through the in-memory identifier index (O(mentions), no table scan)
            if len(u.name) < 4 or u.name in _COMMON_NAMES or u.name.startswith("<"):
                continue
            for src in self._ident_index.get(u.name, ()):
                if src not in new_ids and self._is_live(src) and self._may_reference(self._unit_file.get(src, ""), file_vec_id):
                    edges.append((src, u.vec_id))
        if edges:
            self.conn.executemany("INSERT OR IGNORE INTO edges(src, dst) VALUES (?,?)", edges)
            for a, b in edges:
                self._adj.setdefault(a, set()).add(b)
                self._adj.setdefault(b, set()).add(a)

    # ------------------------------------------------------------------ lookup
    def get_file(self, file_vec_id: str):
        fm = self._file_cache.get(file_vec_id)
        if fm is not None:
            return fm
        row = self.conn.execute("SELECT path, language, source, mipmap FROM files WHERE file_vec_id=?", (file_vec_id,)).fetchone()
        if row is None:
            return None
        fm = _unpack_mipmap(row["mipmap"])
        if fm is None:
            from ..mmu.ast_mipmapper import mipmap_file

            fm = mipmap_file(row["path"], row["source"], row["language"], max_depth=self.settings.l1_max_depth,
                             l1_mode=self.settings.l1_mode)
        self._file_cache[file_vec_id] = fm
        return fm

    def get_unit(self, vec_id: str) -> UnitRecord | None:
        r = self.conn.execute("SELECT * FROM units WHERE vec_id=?", (vec_id,)).fetchone()
        if r is None:
            return None
        return UnitRecord(r["vec_id"], r["file_vec_id"], r["path"], r["language"], r["qualname"], r["kind"],
                          r["start_line"], r["end_line"], r["l0"], r["l1"], r["l2"], r["l3"],
                          {0: r["tok_l0"], 1: r["tok_l1"], 2: r["tok_l2"], 3: r["tok_l3"]}, r["role"])

    def find_units(self, path_suffix: str | None = None, qualname: str | None = None) -> list[UnitRecord]:
        q, args = "SELECT vec_id FROM units WHERE 1=1", []
        if path_suffix:
            q += " AND path LIKE ?"
            args.append("%" + path_suffix)
        if qualname:
            q += " AND qualname = ?"
            args.append(qualname)
        return [u for r in self.conn.execute(q, args) if (u := self.get_unit(r["vec_id"])) is not None]

    def all_vec_ids(self) -> list[str]:
        return list(self._vec_ids)

    def indices(self, vec_ids: Iterable[str]) -> np.ndarray:
        return np.array([self._idx[v] for v in vec_ids if v in self._idx], dtype=np.int64)

    def token_matrices(self, vec_ids: Sequence[str]) -> list[np.ndarray]:
        return [self._tok[self._idx[v]] for v in vec_ids]

    def dense(self, vec_ids: Sequence[str]) -> np.ndarray:
        return self._emb[self.indices(vec_ids)]

    def callees(self, vec_id: str) -> set[str]:
        """Direct callees: units named by an extracted call, defined in the same file or in a module
        the caller's file imports, latest version of their path only. Resolved lazily, cached."""
        if getattr(self, "_callees_dirty", True):
            self._callees = {}
            self._callees_dirty = False
        got = self._callees.get(vec_id)
        if got is not None:
            return got
        src_file = self._unit_file.get(vec_id, "")
        out: set[str] = set()
        for name in self._calls_raw.get(vec_id, ()):
            for dst in self._names.get(name, ()):
                if dst != vec_id and self._is_live(dst) and self._may_reference(src_file, self._unit_file.get(dst, "")):
                    out.add(dst)
        self._callees[vec_id] = out
        return out

    def same_file_in_degree(self, vec_id: str) -> int:
        """Number of distinct units in the same file that call ``vec_id`` (cached per file)."""
        f = self._unit_file.get(vec_id)
        if f is None:
            return 0
        cache = getattr(self, "_indeg_cache", None)
        if cache is None or getattr(self, "_indeg_dirty", True):
            cache = self._indeg_cache = {}
            self._indeg_dirty = False
        per_file = cache.get(f)
        if per_file is None:
            per_file = {}
            for v, fv in self._unit_file.items():
                if fv != f:
                    continue
                for h in self.callees(v):
                    if self._unit_file.get(h) == f:
                        per_file[h] = per_file.get(h, 0) + 1
            cache[f] = per_file
        return per_file.get(vec_id, 0)

    def salience(self, vec_id: str) -> float:
        return self._salience.get(vec_id, 0.0)

    def file_of(self, vec_id: str) -> str | None:
        return self._unit_file.get(vec_id)

    def units_of_file(self, file_vec_id: str) -> list[str]:
        return [v for v, f in self._unit_file.items() if f == file_vec_id]

    def same_file(self, a: str, b: str) -> bool:
        return self._unit_file.get(a) is not None and self._unit_file.get(a) == self._unit_file.get(b)

    def neighbors(self, vec_id: str) -> set[str]:
        return self._adj.get(vec_id, set())

    # ------------------------------------------------------------------ search
    def bm25(self, query: str, k: int, allowed: Sequence[str] | None = None, expand: bool = False) -> list[tuple[str, float]]:
        """FTS5 BM25 (column weights path=2, qualname=4, body=1); optional restriction to ``allowed`` units.

        ``expand=True`` ORs in concept siblings of every query term (sparse thesaurus expansion)."""
        terms = _fts_query_terms(query)
        if expand:  # digits such as HTTP status codes are not indexed but do carry concepts
            seeds = list(dict.fromkeys(terms + [t for t in subword_tokens(query) if t.isdigit()]))
            extra: list[str] = []
            for t in seeds:  # keep siblings that occur in the index, most specific (rarest) first
                sib = [x for x in expand_term(t, 64) if not x.isdigit() and self._df.get(x, 0) > 0]
                extra += sorted(sib, key=lambda x: self._df[x])[:8]
            terms = list(dict.fromkeys(terms + extra))[:64]
        if allowed is not None and len(allowed) >= len(self._vec_ids) and set(allowed) >= set(self._vec_ids):
            allowed = None  # the whole index is allowed: let FTS5 apply LIMIT k directly
        if not terms:
            return []
        match = " OR ".join(f'"{t}"' for t in terms)
        sql = "SELECT rowid, bm25(units_fts, 2.0, 4.0, 1.0) AS s FROM units_fts WHERE units_fts MATCH ? ORDER BY s LIMIT ?"
        pos = self._rowpos()
        if allowed is None:
            with self.lock:
                rows = self.conn.execute(sql, (match, k)).fetchall()
            return [(self._vec_ids[pos[r[0]]], -float(r[1])) for r in rows if r[0] in pos]
        # Restriction is applied in Python: any rowid constraint (IN-list, json_each) makes FTS5
        # re-evaluate the MATCH per rowid (~100x slower). Over-fetch, filter, widen if needed.
        allow = {self._rowids[self._idx[v]] for v in allowed if v in self._idx}
        if not allow:
            return []
        limit = max(8 * k, 512)
        while True:
            with self.lock:
                rows = self.conn.execute(sql, (match, limit)).fetchall()
            out = [(self._vec_ids[pos[r[0]]], -float(r[1])) for r in rows if r[0] in allow and r[0] in pos]
            if len(out) >= k or len(rows) < limit:
                return out[:k]
            limit *= 8

    def _rowpos(self) -> dict[int, int]:
        if getattr(self, "_rowpos_cache", None) is None or len(self._rowpos_cache) != len(self._rowids):
            self._rowpos_cache = {rid: i for i, rid in enumerate(self._rowids)}
        return self._rowpos_cache

    def bq_search(self, qvec: np.ndarray, k: int, allowed_idx: np.ndarray | None = None) -> list[tuple[str, int]]:
        qb = binary_quantize(qvec)
        db = self._bq if allowed_idx is None else self._bq[allowed_idx]
        idx, dist = hamming_topk(qb, db, k)
        base = np.arange(len(self._vec_ids)) if allowed_idx is None else allowed_idx
        return [(self._vec_ids[base[i]], int(d)) for i, d in zip(idx, dist)]

    def mrl_search(self, qvec: np.ndarray, k: int, allowed_idx: np.ndarray | None = None) -> list[tuple[str, float]]:
        q = mrl_slice(qvec, self.mrl_dim)
        db = self._mrl if allowed_idx is None else self._mrl[allowed_idx]
        idx, s = cosine_topk(q, db, k)
        base = np.arange(len(self._vec_ids)) if allowed_idx is None else allowed_idx
        return [(self._vec_ids[base[i]], float(v)) for i, v in zip(idx, s)]

    # ------------------------------------------------------------------ negative memory
    def add_failure(self, *, kind: str, title: str, task: str, code: str, error: str, why: str, remedy: str,
                    emb: np.ndarray) -> int:
        fp = hashlib.blake2b(f"{kind}\0{task}\0{code}\0{error}".encode(), digest_size=8).hexdigest()
        with self.lock:
            row = self.conn.execute("SELECT id FROM negative_memory WHERE fingerprint=?", (fp,)).fetchone()
            if row is not None:
                return int(row["id"])
            cur = self.conn.execute(
                "INSERT INTO negative_memory(fingerprint, kind, title, task, code, error, why, remedy, emb) VALUES (?,?,?,?,?,?,?,?,?)",
                (fp, kind, title, task, code, error, why, remedy, emb.astype(np.float32).tobytes()),
            )
            self._neg_ids.append(int(cur.lastrowid))
            e = emb.astype(np.float32).reshape(1, -1)
            self._neg_emb = np.vstack([self._neg_emb, e]) if self._neg_emb.size else e
            return int(cur.lastrowid)

    def failure_matrix(self) -> tuple[list[int], np.ndarray]:
        return self._neg_ids, self._neg_emb

    def get_failure(self, fid: int) -> dict | None:
        r = self.conn.execute("SELECT * FROM negative_memory WHERE id=?", (fid,)).fetchone()
        if r is None:
            return None
        d = dict(r)
        d.pop("emb", None)
        return d

    def bump_failure(self, fid: int) -> None:
        with self.lock:
            self.conn.execute("UPDATE negative_memory SET hits = hits + 1 WHERE id=?", (fid,))

    def list_failures(self) -> list[dict]:
        return [{k: r[k] for k in r.keys() if k != "emb"} for r in self.conn.execute("SELECT * FROM negative_memory ORDER BY id")]

    # ------------------------------------------------------------------ sessions
    def load_session(self, session_id: str) -> dict | None:
        st = self._sessions.get(session_id)
        if st is not None:
            return json.loads(json.dumps(st))  # callers mutate their copy
        r = self.conn.execute("SELECT state FROM sessions WHERE session_id=?", (session_id,)).fetchone()
        return json.loads(r["state"]) if r else None

    def save_session(self, session_id: str, state: dict) -> None:
        data = json.dumps(state, separators=(",", ":"))
        self._sessions[session_id] = json.loads(data)
        if len(self._sessions) > 1024:
            self._sessions.pop(next(iter(self._sessions)))
        self._write(
            "INSERT INTO sessions(session_id, state, updated_at) VALUES (?,?,?) "
            "ON CONFLICT(session_id) DO UPDATE SET state=excluded.state, updated_at=excluded.updated_at",
            (session_id, data, time.time()),
        )

    def _write(self, sql: str, params: tuple) -> None:
        """Durable write: queued to the write-behind connection for file DBs, synchronous otherwise."""
        if self._writer is not None:
            self._writer.put(sql, params)
        else:
            with self.lock:
                self.conn.execute(sql, params)

    def flush(self) -> None:
        if self._writer is not None:
            self._writer.flush()

    # ------------------------------------------------------------------ contracts
    def save_contract(self, tool_name: str, spec: dict, shadow: dict, code: str) -> None:
        with self.lock:
            self.conn.execute(
                "INSERT INTO contracts(tool_name, spec, shadow, code) VALUES (?,?,?,?) ON CONFLICT(tool_name) DO UPDATE "
                "SET spec=excluded.spec, shadow=excluded.shadow, code=excluded.code, version=version+1, updated_at=unixepoch('subsec')",
                (tool_name, json.dumps(spec), json.dumps(shadow), code),
            )

    def update_shadow(self, tool_name: str, shadow: dict) -> None:
        with self.lock:
            self.conn.execute("UPDATE contracts SET shadow=?, version=version+1, updated_at=unixepoch('subsec') WHERE tool_name=?",
                              (json.dumps(shadow), tool_name))

    def get_contract(self, tool_name: str) -> dict | None:
        r = self.conn.execute("SELECT * FROM contracts WHERE tool_name=?", (tool_name,)).fetchone()
        if r is None:
            return None
        return {"tool_name": r["tool_name"], "spec": json.loads(r["spec"]), "shadow": json.loads(r["shadow"]),
                "code": r["code"], "version": r["version"]}

    def list_contracts(self) -> list[dict]:
        return [self.get_contract(r["tool_name"]) for r in self.conn.execute("SELECT tool_name FROM contracts ORDER BY tool_name")]  # type: ignore[misc]

    def log_drift(self, tool_name: str, report: dict) -> None:
        with self.lock:
            self.conn.execute("INSERT INTO drift_events(tool_name, report) VALUES (?,?)", (tool_name, json.dumps(report)))

    def drift_events(self, limit: int = 20) -> list[dict]:
        return [{"tool_name": r["tool_name"], "report": json.loads(r["report"]), "ts": r["created_at"]}
                for r in self.conn.execute("SELECT * FROM drift_events ORDER BY id DESC LIMIT ?", (limit,))]

    # ------------------------------------------------------------------ telemetry
    def log_event(self, kind: str, *, session_id: str | None = None, tokens_original: int | None = None,
                  tokens_compressed: int | None = None, kv_hit_rate: float | None = None,
                  latency_ms: float | None = None, payload: dict | None = None) -> None:
        self._write(
            "INSERT INTO telemetry(kind, session_id, tokens_original, tokens_compressed, kv_hit_rate, latency_ms, payload) "
            "VALUES (?,?,?,?,?,?,?)",
            (kind, session_id, tokens_original, tokens_compressed, kv_hit_rate, latency_ms,
             json.dumps(payload) if payload is not None else None),
        )

    def recent_events(self, kind: str | None = None, limit: int = 50) -> list[dict]:
        self.flush()
        q = "SELECT * FROM telemetry" + (" WHERE kind=?" if kind else "") + " ORDER BY id DESC LIMIT ?"
        args = ([kind] if kind else []) + [limit]
        out = []
        for r in self.conn.execute(q, args):
            d = dict(r)
            d["payload"] = json.loads(d["payload"]) if d["payload"] else None
            out.append(d)
        return out

    def stats(self) -> dict:
        self.flush()
        r = self.conn.execute(
            """SELECT COUNT(*) AS requests, COALESCE(SUM(tokens_original),0) AS orig, COALESCE(SUM(tokens_compressed),0) AS comp,
               AVG(kv_hit_rate) AS kv, AVG(latency_ms) AS lat FROM telemetry WHERE kind IN ('proxy','fold')"""
        ).fetchone()
        lat = [x["latency_ms"] for x in self.conn.execute(
            "SELECT latency_ms FROM telemetry WHERE kind IN ('proxy','fold') AND latency_ms IS NOT NULL ORDER BY id DESC LIMIT 1000")]
        p50 = float(np.percentile(lat, 50)) if lat else None
        p95 = float(np.percentile(lat, 95)) if lat else None
        orig, comp = int(r["orig"]), int(r["comp"])
        return {
            "requests": int(r["requests"]),
            "tokens_original": orig,
            "tokens_compressed": comp,
            "tokens_saved": orig - comp,
            "savings_pct": round(100.0 * (orig - comp) / orig, 2) if orig else 0.0,
            "kv_hit_rate_avg": round(float(r["kv"]), 4) if r["kv"] is not None else None,
            "latency_ms_avg": round(float(r["lat"]), 3) if r["lat"] is not None else None,
            "latency_ms_p50": round(p50, 3) if p50 is not None else None,
            "latency_ms_p95": round(p95, 3) if p95 is not None else None,
            "units_indexed": self.size,
            "negative_memories": len(self._neg_ids),
            "contracts": int(self.conn.execute("SELECT COUNT(*) FROM contracts").fetchone()[0]),
            "drift_events": int(self.conn.execute("SELECT COUNT(*) FROM drift_events").fetchone()[0]),
        }

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
        self.conn.close()


def _fts_terms(text: str) -> list[str]:
    return [t for t in subword_tokens(text) if not t.isdigit()]


def _fts_query_terms(query: str) -> list[str]:
    seen: dict[str, None] = {}
    for t in subword_tokens(query):
        if t in STOPWORDS or t.isdigit() or len(t) < 2:
            continue
        seen.setdefault(t, None)
        st = stem(t)
        if st != t and len(st) >= 3:
            seen.setdefault(st, None)
    return list(seen)[:48]


_IMPORT_LINE = re.compile(r"^[ \t]*(?:import\b|from[ \t]+\S+[ \t]+import|use[ \t]|#include|#import|using[ \t])[^\n]*", re.M)


def _module_stem(path: str) -> str:
    name = path.rsplit("/", 1)[-1]
    return name.rsplit(".", 1)[0] if "." in name else name


def _import_text(source: str) -> str:
    parts = [m.group(0) for m in _IMPORT_LINE.finditer(source)]
    if "require(" in source:
        parts += [ln for ln in source.split("\n") if "require(" in ln]
    return " ".join(parts)


def _pack_mipmap(fm) -> bytes:
    """Persist the pre-computed mipmap so restarts / re-sightings skip re-parsing (trusted local DB)."""
    import pickle
    import zlib

    try:
        return zlib.compress(pickle.dumps(fm, protocol=pickle.HIGHEST_PROTOCOL), 1)
    except Exception:  # noqa: BLE001
        return b""


def _unpack_mipmap(blob: bytes | None):
    import pickle
    import zlib

    if not blob:
        return None
    try:
        return pickle.loads(zlib.decompress(blob))
    except Exception:  # noqa: BLE001
        return None


class _WriteBehind:
    """Single background writer (own WAL connection) for session state and telemetry.

    Keeps durable bookkeeping off the request path; ``flush()`` is a barrier used before reads."""

    def __init__(self, path: str):
        import queue

        self.path = path
        self.q: "queue.Queue[tuple | None]" = queue.Queue()
        self.t = threading.Thread(target=self._run, name="aegis-write-behind", daemon=True)
        self.t.start()

    def _run(self) -> None:
        import queue

        conn = connect(self.path)
        last_ckpt = time.monotonic()
        while True:
            try:
                item = self.q.get(timeout=1.0)
            except queue.Empty:
                item = ("__checkpoint__", ())
            if item == ("__checkpoint__", ()) or time.monotonic() - last_ckpt > 2.0:
                try:  # the serving connection never auto-checkpoints; do it here, off the request path
                    conn.execute("PRAGMA wal_checkpoint(PASSIVE)")
                except Exception:  # noqa: BLE001
                    pass
                last_ckpt = time.monotonic()
                if item == ("__checkpoint__", ()):
                    continue
            batch = [item]
            while True:  # coalesce bursts into one transaction
                try:
                    batch.append(self.q.get_nowait())
                except queue.Empty:
                    break
            stop = False
            try:
                conn.execute("BEGIN")
                for it in batch:
                    if it is None:
                        stop = True
                    elif isinstance(it, threading.Event):
                        continue
                    else:
                        conn.execute(it[0], it[1])
                conn.execute("COMMIT")
            except Exception:  # noqa: BLE001 - never let bookkeeping take the proxy down
                try:
                    conn.execute("ROLLBACK")
                except Exception:  # noqa: BLE001
                    pass
            for it in batch:
                if isinstance(it, threading.Event):
                    it.set()
                self.q.task_done()
            if stop:
                conn.close()
                return

    def put(self, sql: str, params: tuple) -> None:
        self.q.put((sql, params))

    def flush(self, timeout: float = 5.0) -> None:
        ev = threading.Event()
        self.q.put(ev)
        ev.wait(timeout)

    def close(self) -> None:
        self.flush()
        self.q.put(None)
        self.t.join(timeout=5.0)
        try:
            c = connect(self.path)
            c.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            c.close()
        except Exception:  # noqa: BLE001
            pass


def _salience_of(l0: str) -> float:
    from ..mmu.ast_mipmapper import structural_salience

    return structural_salience(l0)
