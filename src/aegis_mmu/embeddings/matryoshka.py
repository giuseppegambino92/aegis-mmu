"""Matryoshka prefix slicing, 1-bit binary quantization, Hamming search and weighted RRF.

* Matryoshka: ``v~_{1:k} = v_{1:k} / (||v_{1:k}||_2 + 1e-9)`` for k in {64, 128, 256, 512}.
* Binary quantization: ``np.packbits(v > 0)`` -> ``dim/8`` bytes per vector.
* Hamming distance: XOR + popcount over uint64 words (``np.bitwise_count``) with a 256-entry
  lookup-table fallback for NumPy < 2.0.
* RRF: ``RRF(d) = sum_m w_m / (k + rank_m(d))`` with 1-based ranks.
"""

from __future__ import annotations

from typing import Hashable, Iterable, Mapping, Sequence

import numpy as np

MRL_DIMS: tuple[int, ...] = (64, 128, 256, 512)

_POPCOUNT_LUT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
_HAS_BITWISE_COUNT = hasattr(np, "bitwise_count")


def mrl_slice(v: np.ndarray, k: int) -> np.ndarray:
    """Return the first ``k`` dims of ``v`` (1-D or 2-D) re-normalised to unit L2 norm."""
    v = np.asarray(v, dtype=np.float32)
    if k <= 0 or k > v.shape[-1]:
        raise ValueError(f"Matryoshka prefix {k} out of range for dim {v.shape[-1]}")
    p = v[..., :k]
    return p / (np.linalg.norm(p, axis=-1, keepdims=True) + 1e-9)


def matryoshka_pyramid(v: np.ndarray, dims: Iterable[int] = MRL_DIMS) -> dict[int, np.ndarray]:
    return {d: mrl_slice(v, d) for d in dims if d <= np.asarray(v).shape[-1]}


def binary_quantize(v: np.ndarray) -> np.ndarray:
    """Pack sign bits of float vectors into uint8 arrays: shape (..., dim) -> (..., dim/8)."""
    v = np.asarray(v)
    return np.packbits(v > 0, axis=-1)


def _as_words(packed: np.ndarray) -> np.ndarray:
    packed = np.ascontiguousarray(packed, dtype=np.uint8)
    if packed.shape[-1] % 8 == 0:
        return packed.view(np.uint64)
    return packed


def popcount(x: np.ndarray) -> np.ndarray:
    if _HAS_BITWISE_COUNT:
        return np.bitwise_count(x)
    return _POPCOUNT_LUT[x.view(np.uint8)].reshape(*x.shape, -1).sum(axis=-1)


def hamming_distances(query_packed: np.ndarray, db_packed: np.ndarray) -> np.ndarray:
    """Hamming distance between one packed query (nbytes,) and a packed matrix (n, nbytes)."""
    if db_packed.size == 0:
        return np.zeros(0, dtype=np.int32)
    q = _as_words(query_packed.reshape(1, -1))
    db = _as_words(db_packed)
    x = np.bitwise_xor(db, q)
    return popcount(x).sum(axis=-1, dtype=np.int32)


def hamming_topk(query_packed: np.ndarray, db_packed: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Indices and distances of the ``k`` nearest packed vectors (ascending distance, stable)."""
    d = hamming_distances(query_packed, db_packed)
    n = d.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.int64), d
    k = min(k, n)
    if k < n:
        kth = np.partition(d, k - 1)[k - 1]
        idx = np.nonzero(d <= kth)[0]  # include every tie at the boundary, then break ties by index
        idx = idx[np.lexsort((idx, d[idx]))][:k]
    else:
        idx = np.lexsort((np.arange(n), d))
    return idx, d[idx]


def cosine_topk(query: np.ndarray, matrix: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Top-k by dot product (inputs assumed L2-normalised)."""
    if matrix.size == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float32)
    s = matrix @ query
    n = s.shape[0]
    k = min(k, n)
    if k < n:
        kth = np.partition(-s, k - 1)[k - 1]
        idx = np.nonzero(-s <= kth)[0]
    else:
        idx = np.arange(n)
    idx = idx[np.lexsort((idx, -s[idx]))][:k]
    return idx, s[idx]


def rrf_fuse(
    rankings: Mapping[str, Sequence[Hashable]],
    weights: Mapping[str, float] | None = None,
    k: int = 60,
) -> list[tuple[Hashable, float]]:
    """Weighted Reciprocal Rank Fusion. Returns ``[(doc, score)]`` sorted by score desc."""
    scores: dict[Hashable, float] = {}
    first_seen: dict[Hashable, int] = {}
    order = 0
    for method, ranked in rankings.items():
        w = 1.0 if weights is None else float(weights.get(method, 1.0))
        if w == 0.0:
            continue
        for rank, doc in enumerate(ranked, start=1):
            scores[doc] = scores.get(doc, 0.0) + w / (k + rank)
            if doc not in first_seen:
                first_seen[doc] = order
                order += 1
    return sorted(scores.items(), key=lambda kv: (-kv[1], first_seen[kv[0]]))
