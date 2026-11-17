"""Token accounting.

Uses exact ``tiktoken`` cl100k_base BPE counts when ``tiktoken`` is installed and its encoding file
is available (``pip install aegis-mmu[tiktoken]``), otherwise a deterministic offline approximation
of BPE behaviour on source code (``AEGIS_TOKENIZER=heuristic`` forces it). cl100k is OpenAI's
tokenizer; Claude's tokenizer is not public, so counts for Claude models remain an approximation.
The heuristic:
letter runs cost ~1 token per 6 chars, digit runs 1 per 3, every punctuation char is a token,
and each newline (with its indentation) is a token.
"""

from __future__ import annotations

import os
import re
from functools import lru_cache

_PIECE = re.compile(r"[A-Za-z]+|[0-9]+|\n[ \t]*|[^\sA-Za-z0-9]")


def _heuristic(text: str) -> int:
    n = 0
    for m in _PIECE.finditer(text):
        s = m.group()
        c = s[0]
        if c.isalpha():
            n += 1 + (len(s) - 1) // 6
        elif c.isdigit():
            n += 1 + (len(s) - 1) // 3
        else:
            n += 1
    return n


@lru_cache(maxsize=1)
def _tiktoken_encoder():
    if os.environ.get("AEGIS_TOKENIZER", "auto") == "heuristic":
        return None
    try:
        import tiktoken  # type: ignore

        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        return None


def tokenizer_name() -> str:
    return "tiktoken/cl100k_base" if _tiktoken_encoder() is not None else "aegis-heuristic-bpe"


def count_tokens(text: str) -> int:
    if not text:
        return 0
    if len(text) <= 2048:
        return _count_cached(text)
    return _count(text)


def _count(text: str) -> int:
    enc = _tiktoken_encoder()
    if enc is not None:
        return len(enc.encode(text, disallowed_special=()))
    return _heuristic(text)


@lru_cache(maxsize=65536)
def _count_cached(text: str) -> int:
    """Short strings (signatures, fold markers, pages) recur constantly in the hot path."""
    return _count(text)


@lru_cache(maxsize=1)
def _token_byte_lengths():
    """Byte length of every cl100k token id (built once) so per-line counts need a single encode."""
    import numpy as np

    enc = _tiktoken_encoder()
    lens = np.zeros(enc.n_vocab, dtype=np.int64)
    for i in range(enc.n_vocab):
        try:
            lens[i] = len(enc.decode_single_token_bytes(i))
        except KeyError:
            lens[i] = 0
    return lens


def line_prefix(text: str):
    """Prefix sums of per-line token counts from ONE encode of the whole text.

    ``P[i]`` = tokens in the first ``i`` lines (``text.split("\\n")``); tokens of lines ``a..b``
    (1-based, inclusive) = ``P[b] - P[a-1]``. Each BPE token is attributed to the line its first
    byte falls on (cl100k never merges across a newline into the next line's code)."""
    import numpy as np

    lines = text.split("\n")
    enc = _tiktoken_encoder()
    if enc is None or not text:
        # +1 per line for its newline token; the last line has no newline after it.
        counts = np.fromiter((_heuristic(ln) + 1 for ln in lines), dtype=np.int64, count=len(lines))
        counts[-1] -= 1
    else:
        data = text.encode("utf-8")
        toks = np.asarray(enc.encode_ordinary(text), dtype=np.int64)
        starts = np.concatenate(([0], np.cumsum(_token_byte_lengths()[toks])[:-1])) if toks.size else np.zeros(0, np.int64)
        nl = np.frombuffer(data, dtype=np.uint8)
        nl_pos = np.flatnonzero(nl == 10)
        line_of = np.searchsorted(nl_pos, starts, side="right")
        counts = np.bincount(line_of, minlength=len(lines))[: len(lines)]
    out = np.zeros(len(lines) + 1, dtype=np.int64)
    np.cumsum(counts, out=out[1:])
    return out


EST_A, EST_B = 0.216, 0.742  # least-squares fit on codebase_v1 + test samples vs cl100k (total bias +0.1%)


def estimate_tokens(text: str) -> int:
    """O(1) BPE estimate for cold-path sizing: ``0.216 * chars + 0.742 * newlines`` (~11% per-chunk error)."""
    return int(EST_A * len(text) + EST_B * text.count("\n")) + 1 if text else 0


def line_prefix_estimate(text: str):
    """Vectorised per-line estimate with the same contract as :func:`line_prefix`."""
    import numpy as np

    lens = np.fromiter((len(ln) for ln in text.split("\n")), dtype=np.float64)
    counts = EST_A * lens + EST_B
    out = np.zeros(len(lens) + 1, dtype=np.float64)
    np.cumsum(counts, out=out[1:])
    return np.rint(out).astype(np.int64)


def common_prefix_len(a: str, b: str) -> int:
    """Length of the longest common prefix, via binary search on C-level slice compares."""
    lo, hi = 0, min(len(a), len(b))
    if a[:hi] == b[:hi]:
        return hi
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if a[:mid] == b[:mid]:
            lo = mid
        else:
            hi = mid - 1
    return lo


def prefix_token_hit_rate(prev: str, cur: str) -> float:
    """KV_Hit_Rate = LCP_tokens(prev, cur) / tokens(cur)."""
    total = count_tokens(cur)
    if total == 0 or not prev:
        return 0.0
    lcp = common_prefix_len(prev, cur)
    return min(1.0, count_tokens(cur[:lcp]) / total)
