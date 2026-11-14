"""ColBERT-style late interaction: ``MaxSim(Q, D) = sum_i max_j q_i . d_j``.

Documents are stored as float16 token matrices. ``batch_maxsim`` scores many documents with a
single GEMM against the concatenated token matrix followed by a segmented max
(``np.maximum.reduceat``), so reranking 50 candidates costs one matrix multiply.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def maxsim(q: np.ndarray, d: np.ndarray) -> float:
    if q.size == 0 or d.size == 0:
        return 0.0
    return float((q @ d.T).max(axis=1).sum())


def batch_maxsim(q: np.ndarray, docs: Sequence[np.ndarray], normalize: bool = True,
                 weights: np.ndarray | None = None) -> np.ndarray:
    """MaxSim of query tokens ``q`` (|Q|, d) against each doc matrix. Normalised by |Q| (or by the
    weight mass when per-query-token ``weights`` such as IDF are given)."""
    n = len(docs)
    out = np.zeros(n, dtype=np.float32)
    if n == 0 or q.size == 0:
        return out
    lengths = np.array([0 if dm is None else dm.shape[0] for dm in docs], dtype=np.int64)
    nonempty = np.nonzero(lengths > 0)[0]
    if nonempty.size == 0:
        return out
    big = np.concatenate([docs[i] for i in nonempty], axis=0).astype(np.float32, copy=False)
    sims = q.astype(np.float32, copy=False) @ big.T  # (|Q|, total_tokens)
    offsets = np.concatenate(([0], np.cumsum(lengths[nonempty])[:-1]))
    per_doc_max = np.maximum.reduceat(sims, offsets, axis=1)  # (|Q|, n_nonempty)
    if weights is not None:
        w = np.asarray(weights, dtype=np.float32)[: q.shape[0]]
        scores = (w[:, None] * per_doc_max).sum(axis=0)
        if normalize:
            scores = scores / max(float(w.sum()), 1e-9)
    else:
        scores = per_doc_max.sum(axis=0)
        if normalize:
            scores = scores / q.shape[0]
    out[nonempty] = scores
    return out


def encode_token_matrix(m: np.ndarray) -> bytes:
    return np.ascontiguousarray(m, dtype=np.float16).tobytes()


def decode_token_matrix(blob: bytes, dim: int) -> np.ndarray:
    if not blob:
        return np.zeros((0, dim), dtype=np.float16)
    return np.frombuffer(blob, dtype=np.float16).reshape(-1, dim)
