"""Embedding backends.

``FastLocalEngine`` (default, offline, deterministic)
    Signed feature hashing + Gaussian random projection (Johnson-Lindenstrauss).
    Every feature f (subword, light stem, char-trigram, concept, bigram, syntax operator) is hashed
    with CRC32 to a bucket ``h(f)`` and a sign ``s(f)``; the document vector is

        v = sum_f  w_f * s(f) * R[h(f)],      R ~ N(0, 1)^{B x D}

    which is exactly ``x_hashed @ R`` (sparse hashed TF vector times a Gaussian projection). Any
    prefix of the columns of a Gaussian projection is itself a JL projection, so Matryoshka prefix
    slicing (64/128/256/512) preserves cosine geometry with graceful precision loss. Per-subword
    projections are cached, which keeps throughput well above 2,000 docs/sec on a laptop.

    Token vectors for ColBERT MaxSim use the first ``token_dim`` columns of the same projection
    with a +-1 neighbour context mix.

Remote / accelerated adapters (``OllamaEngine``, ``OpenAIEngine``, ``VoyageEngine``, ``MLXEngine``)
    produce dense vectors from the respective services; token-level vectors for late interaction
    fall back to the local featuriser unless the backend exposes multi-vector outputs.
"""

from __future__ import annotations

import math
import re
import zlib
from collections import Counter
from typing import Sequence

import numpy as np

from .base import l2_normalize
from .lexicon import STOPWORDS, concepts_of, stem, subword_tokens

_OPS = re.compile(r"==|!=|<=|>=|\+=|-=|\*=|->|=>|::|\?\.|\?\?|&&|\|\||<<|>>|\*\*|\.\.\.|@\w+")


def _h(feature: str) -> int:
    return zlib.crc32(feature.encode("utf-8"))


class FastLocalEngine:
    """Deterministic offline engine: signed feature hashing + Signed Random Projection."""

    def __init__(self, dim: int = 512, token_dim: int = 128, buckets: int = 1 << 14, seed: int = 0xAE615):
        if buckets & (buckets - 1):
            raise ValueError("buckets must be a power of two")
        self.name = f"fast-local-srp-{dim}"
        self.dim = dim
        self.token_dim = min(token_dim, dim)
        self.buckets = buckets
        self._mask = buckets - 1
        rng = np.random.default_rng(seed)
        self._R = rng.standard_normal((buckets, dim), dtype=np.float32)
        self._word_cache: dict[str, np.ndarray] = {}
        self._feat_cache: dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------ features
    def _feature_row(self, feature: str) -> tuple[int, float]:
        """(bucket, sign) of a hashed feature (cached)."""
        r = self._feat_cache.get(feature)
        if r is None:
            h = _h(feature)
            r = (h & self._mask, 1.0 if (h >> 20) & 1 else -1.0)
            if len(self._feat_cache) > 400_000:
                self._feat_cache.clear()
            self._feat_cache[feature] = r
        return r

    def _feature_vec(self, feature: str) -> np.ndarray:
        row, sign = self._feature_row(feature)
        return self._R[row] * sign

    def _word_vec(self, w: str) -> np.ndarray:
        """Projection of all features derived from one subword (cached)."""
        v = self._word_cache.get(w)
        if v is not None:
            return v
        acc = self._feature_vec("w:" + w).copy()
        st = stem(w)
        if st != w:
            acc += 0.8 * self._feature_vec("w:" + st)
        if not w.isdigit():
            padded = f"#{st}#"
            grams = [padded[i : i + 3] for i in range(len(padded) - 2)]
            if grams:
                gw = 0.6 / math.sqrt(len(grams))
                for g in grams:
                    acc += gw * self._feature_vec("g:" + g)
        cs = concepts_of(w)
        if cs:
            cw = 0.7 / math.sqrt(len(cs))
            for c in cs:
                acc += cw * self._feature_vec("c:" + c)
        if len(self._word_cache) > 200_000:
            self._word_cache.clear()
        self._word_cache[w] = acc
        return acc

    # ------------------------------------------------------------------ dense
    def _embed_one(self, text: str, use_stopwords: bool = True, term_weights: dict[str, float] | None = None) -> np.ndarray:
        toks = subword_tokens(text)
        if not toks:
            return np.zeros(self.dim, dtype=np.float32)
        counts = Counter(toks)
        stop = STOPWORDS if use_stopwords else frozenset()
        words = list(counts)
        tw = term_weights or {}
        cw = np.fromiter(((1.0 + math.log(counts[w])) * (0.25 if w in stop else 1.0) * tw.get(w, 1.0) for w in words),
                         dtype=np.float32, count=len(words))
        acc = cw @ np.stack([self._word_vec(w) for w in words])
        # sparse features (bigrams, operators) -> one gather + one GEMV over projection rows
        rows: list[int] = []
        coef: list[float] = []
        content = [t for t in toks if t not in stop]
        if len(content) > 1:
            for (a, b), c in Counter(zip(content, content[1:])).items():
                r, sg = self._feature_row(f"b:{a}_{b}")
                rows.append(r)
                coef.append(0.35 * (1.0 + math.log(c)) * sg)
        for op, c in Counter(_OPS.findall(text)).items():
            r, sg = self._feature_row("o:" + op)
            rows.append(r)
            coef.append(0.15 * (1.0 + math.log(c)) * sg)
        if rows:
            acc = acc + np.asarray(coef, dtype=np.float32) @ self._R[rows]
        return acc

    def embed(self, texts: Sequence[str], use_stopwords: bool = True) -> np.ndarray:
        """Dense embeddings for a batch: one sparse (N x V) @ (V x D) product over cached word
        projections plus one sparse (N x B) @ (B x D) product for bigram/operator features.
        ``use_stopwords=False`` for short names (JSON keys) where words like ``from``/``to`` matter."""
        n = len(texts)
        if not n:
            return np.zeros((0, self.dim), dtype=np.float32)
        if n == 1:
            return l2_normalize(self._embed_one(texts[0], use_stopwords)[None, :])
        from scipy.sparse import csr_matrix

        stop = STOPWORDS if use_stopwords else frozenset()
        vocab: dict[str, int] = {}
        rows: list[int] = []
        cols: list[int] = []
        vals: list[float] = []
        frows: list[int] = []
        fcols: list[int] = []
        fvals: list[float] = []
        for i, text in enumerate(texts):
            toks = subword_tokens(text)
            for w, c in Counter(toks).items():
                rows.append(i)
                cols.append(vocab.setdefault(w, len(vocab)))
                vals.append((1.0 + math.log(c)) * (0.25 if w in stop else 1.0))
            content = [t for t in toks if t not in stop]
            if len(content) > 1:
                for (a, b), c in Counter(zip(content, content[1:])).items():
                    r, sg = self._feature_row(f"b:{a}_{b}")
                    frows.append(i)
                    fcols.append(r)
                    fvals.append(0.35 * (1.0 + math.log(c)) * sg)
            for op, c in Counter(_OPS.findall(text)).items():
                r, sg = self._feature_row("o:" + op)
                frows.append(i)
                fcols.append(r)
                fvals.append(0.15 * (1.0 + math.log(c)) * sg)
        out = np.zeros((n, self.dim), dtype=np.float32)
        if vocab:
            W = np.stack([self._word_vec(w) for w in vocab])
            out += csr_matrix((np.asarray(vals, np.float32), (rows, cols)), shape=(n, len(vocab))) @ W
        if frows:
            out += csr_matrix((np.asarray(fvals, np.float32), (frows, fcols)), shape=(n, self.buckets)) @ self._R
        return l2_normalize(out)

    def embed_query(self, text: str, term_weights: dict[str, float] | None = None) -> np.ndarray:
        """Query embedding; optional per-subword weights (e.g. corpus IDF) re-weight query terms only."""
        if term_weights:
            return l2_normalize(self._embed_one(text, True, term_weights)[None, :])[0]
        return self.embed([text])[0]

    @staticmethod
    def query_token_list(text: str, max_tokens: int | None = None) -> list[str]:
        """The subwords that ``embed_tokens(text, is_query=True)`` emits one row for, in order."""
        toks = [t for t in subword_tokens(text) if t not in STOPWORDS]
        return toks[:max_tokens] if max_tokens else toks

    # ------------------------------------------------------------------ multi-vector
    def embed_tokens_batch(self, texts: Sequence[str], max_tokens: int | None = None) -> list[np.ndarray]:
        """Document token matrices for many texts at once (contextual mix never crosses documents)."""
        td = self.token_dim
        seqs = [[t for t in subword_tokens(x) if t not in STOPWORDS] for x in texts]
        lens = np.array([len(q) for q in seqs], dtype=np.int64)
        if lens.sum() == 0:
            return [np.zeros((0, td), dtype=np.float32) for _ in texts]
        vocab: dict[str, int] = {}
        ids = np.fromiter((vocab.setdefault(t, len(vocab)) for q in seqs for t in q), dtype=np.int64, count=int(lens.sum()))
        Bu = l2_normalize(np.stack([self._word_vec(w)[:td] for w in vocab]))
        base = Bu[ids]
        doc = np.repeat(np.arange(len(texts)), lens)
        same_prev = np.zeros(len(ids), dtype=bool)
        same_prev[1:] = doc[1:] == doc[:-1]
        ctx = base.copy()
        ctx[1:][same_prev[1:]] += 0.2 * base[:-1][same_prev[1:]]
        ctx[:-1][same_prev[1:]] += 0.2 * base[1:][same_prev[1:]]
        key = doc * len(vocab) + ids
        ukeys, inv, freq = np.unique(key, return_inverse=True, return_counts=True)
        pooled = np.zeros((len(ukeys), td), dtype=np.float32)
        np.add.at(pooled, inv, ctx)
        pooled = l2_normalize(pooled)
        udoc = ukeys // len(vocab)
        bounds = np.searchsorted(udoc, np.arange(len(texts) + 1))
        out = []
        for i in range(len(texts)):
            m = pooled[bounds[i]:bounds[i + 1]]
            if max_tokens and m.shape[0] > max_tokens:
                f = freq[bounds[i]:bounds[i + 1]]
                m = m[np.sort(np.argsort(-f, kind="stable")[:max_tokens])]
            out.append(m)
        return out

    def embed_tokens(self, text: str, *, is_query: bool = False, max_tokens: int | None = None) -> np.ndarray:
        toks = [t for t in subword_tokens(text) if t not in STOPWORDS]
        td = self.token_dim
        if not toks:
            return np.zeros((0, td), dtype=np.float32)
        uniq: dict[str, int] = {}
        inv = np.fromiter((uniq.setdefault(t, len(uniq)) for t in toks), dtype=np.int64, count=len(toks))
        base_u = l2_normalize(np.stack([self._word_vec(t)[:td] for t in uniq]))
        base = base_u[inv]
        ctx = base.copy()
        if len(toks) > 1:
            ctx[1:] += 0.2 * base[:-1]
            ctx[:-1] += 0.2 * base[1:]
        if is_query:
            out = l2_normalize(ctx)
            return out[:max_tokens] if max_tokens else out
        # Documents: one contextual vector per unique subword (mean over its occurrences).
        pooled = np.zeros((len(uniq), td), dtype=np.float32)
        np.add.at(pooled, inv, ctx)
        out = l2_normalize(pooled)
        if max_tokens and out.shape[0] > max_tokens:
            # keep the most frequent subwords
            freq = np.bincount(inv, minlength=len(uniq))
            keep = np.sort(np.argsort(-freq, kind="stable")[:max_tokens])
            out = out[keep]
        return out


class EmbeddingCache:
    """Content-addressed embedding cache: blake2b(model | kind | dim | text) -> float32 vector.

    In-memory dict in front of an optional SQLite file, so warm re-indexing and repeated proxy
    intercepts never call the model twice for the same text."""

    def __init__(self, path: str | None = None):
        import sqlite3
        import threading

        self.mem: dict[str, np.ndarray] = {}
        self.lock = threading.Lock()
        self.conn = None
        if path:
            from pathlib import Path

            Path(path).parent.mkdir(parents=True, exist_ok=True)
            self.conn = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
            self.conn.execute("PRAGMA journal_mode=WAL")
            self.conn.execute("CREATE TABLE IF NOT EXISTS emb (k TEXT PRIMARY KEY, v BLOB NOT NULL)")

    @staticmethod
    def key(model: str, kind: str, dim: int, text: str) -> str:
        import hashlib

        return hashlib.blake2b(f"{model}\x00{kind}\x00{dim}\x00{text}".encode("utf-8"), digest_size=16).hexdigest()

    def get_many(self, keys: list[str]) -> dict[str, np.ndarray]:
        out = {k: self.mem[k] for k in keys if k in self.mem}
        missing = [k for k in keys if k not in out]
        if missing and self.conn is not None:
            with self.lock:
                for i in range(0, len(missing), 500):
                    chunk = missing[i : i + 500]
                    for k, v in self.conn.execute(f"SELECT k, v FROM emb WHERE k IN ({','.join('?' * len(chunk))})", chunk):
                        out[k] = self.mem[k] = np.frombuffer(v, dtype=np.float32)
        return out

    def put_many(self, items: dict[str, np.ndarray]) -> None:
        self.mem.update(items)
        if self.conn is not None and items:
            with self.lock:
                self.conn.executemany("INSERT OR REPLACE INTO emb(k, v) VALUES (?, ?)",
                                      [(k, np.asarray(v, dtype=np.float32).tobytes()) for k, v in items.items()])


class _RemoteDenseEngine:
    """Shared plumbing for model-backed embedders: content-hash cache, Matryoshka truncation,
    query/document task modes. Token vectors for ColBERT late interaction come from the local
    featuriser (these services expose no multi-vector output)."""

    def __init__(self, model: str, dim: int, token_dim: int = 128, timeout: float = 60.0, cache_path: str | None = None):
        import httpx

        self.model = model
        self.dim = dim
        self.token_dim = token_dim
        self._http = httpx.Client(timeout=timeout)
        self._local = FastLocalEngine(dim=max(token_dim, 128), token_dim=token_dim)
        self.cache = EmbeddingCache(cache_path)
        self.calls = 0  # number of model round-trips (cache misses), for reporting

    def _post(self, url: str, payload: dict, headers: dict | None = None) -> dict:
        r = self._http.post(url, json=payload, headers=headers or {})
        r.raise_for_status()
        return r.json()

    def _fit(self, vecs) -> np.ndarray:
        m = np.asarray(vecs, dtype=np.float32)
        if m.shape[1] >= self.dim:
            m = m[:, : self.dim]  # Matryoshka-trained models tolerate prefix truncation
        else:
            m = np.pad(m, ((0, 0), (0, self.dim - m.shape[1])))
        return l2_normalize(m)

    def _raw_embed(self, texts: list[str], kind: str) -> list[list[float]]:  # pragma: no cover - overridden
        raise NotImplementedError

    def _embed_kind(self, texts: Sequence[str], kind: str) -> np.ndarray:
        texts = list(texts)
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        keys = [EmbeddingCache.key(self.model, kind, self.dim, t) for t in texts]
        hit = self.cache.get_many(keys)
        todo = [i for i, k in enumerate(keys) if k not in hit]
        if todo:
            new: dict[str, np.ndarray] = {}
            for b in range(0, len(todo), 64):
                idx = todo[b : b + 64]
                self.calls += 1
                vecs = self._fit(self._raw_embed([texts[i] for i in idx], kind))
                for i, v in zip(idx, vecs):
                    new[keys[i]] = v
            self.cache.put_many(new)
            hit.update(new)
        return np.stack([hit[k] for k in keys]).astype(np.float32)

    def embed(self, texts: Sequence[str], use_stopwords: bool = True, kind: str = "document") -> np.ndarray:
        return self._embed_kind(texts, kind)

    def embed_query(self, text: str, term_weights: dict[str, float] | None = None) -> np.ndarray:
        return self._embed_kind([text], "query")[0]

    def embed_tokens(self, text: str, *, is_query: bool = False, max_tokens: int | None = None) -> np.ndarray:
        return self._local.embed_tokens(text, is_query=is_query, max_tokens=max_tokens)

    def embed_tokens_batch(self, texts: Sequence[str], max_tokens: int | None = None) -> list[np.ndarray]:
        return self._local.embed_tokens_batch(texts, max_tokens=max_tokens)

    def query_token_list(self, text: str, max_tokens: int | None = None) -> list[str]:
        return self._local.query_token_list(text, max_tokens)


class OllamaEngine(_RemoteDenseEngine):
    """Local Ollama server (``/api/embed``), e.g. ``nomic-embed-text`` (Matryoshka-trained, 768-d,
    truncated to ``dim``). nomic models expect ``search_query:`` / ``search_document:`` task prefixes."""

    def __init__(self, model: str = "nomic-embed-text", base_url: str = "http://localhost:11434", dim: int = 512, **kw):
        super().__init__(model, dim, **kw)
        self.name = f"ollama:{model}"
        self.base_url = base_url.rstrip("/")
        self._prefix = "nomic" in model

    def _raw_embed(self, texts: list[str], kind: str) -> list[list[float]]:
        if self._prefix:
            pre = "search_query: " if kind == "query" else "search_document: "
            texts = [pre + t for t in texts]
        data = self._post(f"{self.base_url}/api/embed", {"model": self.model, "input": texts, "truncate": True})
        return data["embeddings"]

    def available(self) -> bool:
        try:
            r = self._http.get(f"{self.base_url}/api/tags", timeout=2.0)
            return r.status_code == 200 and any(m.get("name", "").startswith(self.model) for m in r.json().get("models", []))
        except Exception:  # noqa: BLE001
            return False


class OpenAIEngine(_RemoteDenseEngine):
    """OpenAI ``/v1/embeddings`` (``text-embedding-3-*`` support native Matryoshka ``dimensions``)."""

    def __init__(self, model: str = "text-embedding-3-small", api_key: str | None = None,
                 base_url: str = "https://api.openai.com", dim: int = 512, **kw):
        import os

        super().__init__(model, dim, **kw)
        self.name = f"openai:{model}"
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.base_url = base_url.rstrip("/")

    def _raw_embed(self, texts: list[str], kind: str) -> list[list[float]]:
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        data = self._post(f"{self.base_url}/v1/embeddings", {"model": self.model, "input": texts, "dimensions": self.dim},
                          {"Authorization": f"Bearer {self.api_key}"})
        return [d["embedding"] for d in sorted(data["data"], key=lambda d: d["index"])]


class VoyageEngine(_RemoteDenseEngine):
    """Voyage AI embeddings (Anthropic's recommended embedding provider; Anthropic has no embeddings API)."""

    def __init__(self, model: str = "voyage-code-3", api_key: str | None = None, dim: int = 512, **kw):
        import os

        super().__init__(model, dim, **kw)
        self.name = f"voyage:{model}"
        self.api_key = api_key or os.environ.get("VOYAGE_API_KEY", "")

    def _raw_embed(self, texts: list[str], kind: str) -> list[list[float]]:
        if not self.api_key:
            raise RuntimeError("VOYAGE_API_KEY is not set")
        data = self._post("https://api.voyageai.com/v1/embeddings",
                          {"model": self.model, "input": texts, "output_dimension": self.dim, "input_type": kind},
                          {"Authorization": f"Bearer {self.api_key}"})
        return [d["embedding"] for d in sorted(data["data"], key=lambda d: d["index"])]


class MLXEngine(_RemoteDenseEngine):
    """Apple-silicon MLX embeddings via the optional ``mlx-embeddings`` package."""

    def __init__(self, model: str = "mlx-community/nomicai-modernbert-embed-base-4bit", dim: int = 512, **kw):
        try:
            from mlx_embeddings import load  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional
            raise RuntimeError("MLX backend requires `pip install mlx-embeddings` on Apple silicon") from exc
        super().__init__(model, dim, **kw)
        self.name = f"mlx:{model}"
        self._model, self._tok = load(model)

    def _raw_embed(self, texts: list[str], kind: str) -> list[list[float]]:  # pragma: no cover - optional hardware
        import mlx.core as mx  # type: ignore

        enc = self._tok.batch_encode_plus(list(texts), return_tensors="mlx", padding=True, truncation=True, max_length=512)
        out = self._model(enc["input_ids"], attention_mask=enc["attention_mask"])
        return np.array(mx.array(out.text_embeds)).tolist()


_ENGINE_CACHE: dict[tuple, object] = {}


def get_engine(settings=None):
    """Build (and memoise) the engine configured in :class:`~aegis_mmu.config.AegisSettings`."""
    from ..config import get_settings

    s = settings or get_settings()
    key = (s.embedding_provider, s.embedding_model, s.embed_dim, s.colbert_dim, s.hash_buckets, s.embed_seed,
           s.embedding_cache_path)
    eng = _ENGINE_CACHE.get(key)
    if eng is not None:
        return eng
    p = s.embedding_provider
    if p == "fast-local":
        eng = FastLocalEngine(dim=s.embed_dim, token_dim=s.colbert_dim, buckets=s.hash_buckets, seed=s.embed_seed)
    elif p == "ollama":
        eng = OllamaEngine(model=s.embedding_model or "nomic-embed-text", dim=s.embed_dim, token_dim=s.colbert_dim,
                           cache_path=s.embedding_cache_path or str(s.home / "embeddings.db"))
    elif p == "openai":
        eng = OpenAIEngine(model=s.embedding_model or "text-embedding-3-small", dim=s.embed_dim, token_dim=s.colbert_dim)
    elif p == "voyage":
        eng = VoyageEngine(model=s.embedding_model or "voyage-code-3", dim=s.embed_dim, token_dim=s.colbert_dim)
    elif p == "mlx":
        eng = MLXEngine(dim=s.embed_dim, token_dim=s.colbert_dim)
    else:  # pragma: no cover
        raise ValueError(f"unknown embedding provider {p}")
    _ENGINE_CACHE[key] = eng
    return eng
