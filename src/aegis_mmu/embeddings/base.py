"""EmbeddingEngine protocol shared by every embedding backend."""

from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

import numpy as np


@runtime_checkable
class EmbeddingEngine(Protocol):
    """A dense + multi-vector embedding backend.

    * ``embed`` returns an ``(n, dim)`` float32 matrix of L2-normalised vectors whose leading
      coordinates are usable as Matryoshka prefixes (re-normalised by :mod:`.matryoshka`).
    * ``embed_tokens`` returns a ``(t, token_dim)`` float32 matrix of L2-normalised per-token
      vectors for ColBERT-style late interaction.
    """

    name: str
    dim: int
    token_dim: int

    def embed(self, texts: Sequence[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...

    def embed_tokens(self, text: str, *, is_query: bool = False, max_tokens: int | None = None) -> np.ndarray: ...


def l2_normalize(m: np.ndarray, axis: int = -1) -> np.ndarray:
    m = np.asarray(m, dtype=np.float32)
    norms = np.linalg.norm(m, axis=axis, keepdims=True)
    return m / (norms + 1e-9)
