"""Append-only semantic paging for prefix KV-cache stability.

Provider prompt caches (Anthropic, OpenAI, vLLM/llama.cpp prefix caches) only reuse the longest
byte-identical *prefix*. Classic RAG re-ranks chunks every turn and therefore rewrites the prefix.
The pager keeps a per-session, ordered page table:

* pages already resident keep their exact sequence index as long as the requested resolution has
  not *improved* (a degraded request keeps the richer resident page: sticky residency);
* new pages and promotions (e.g. L1 -> L0) are appended strictly at the tail, ordered by temporal
  stability (file skeletons L3/L2 first, then L1, then L0, then volatile warnings);
* when resident tokens exceed the cap (``overflow x budget``) only the tail is compacted: the
  leading pages up to ``pin_frac x cap`` form an immutable pinned prefix that is never reordered,
  re-rendered or evicted (the header's epoch does not change either), and the tail is rebuilt
  from the current requests within the remaining room.

    KV_Hit_Rate_t = LCP_tokens(Prompt_{t-1}, Prompt_t) / Tokens(Prompt_t)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Sequence

from ..tokens import count_tokens, prefix_token_hit_rate

# lower = more stable = earlier in the prefix
STABILITY = {"skeleton3": 0, "skeleton2": 1, "leaf1": 2, "leaf0": 3, "warning": 4}


@dataclass
class PageRequest:
    key: str  # vec_id or file:<file_vec_id> or warn:<id>
    level: int  # 0 best .. 3 coarsest
    text: str
    kind: str = "leaf"  # leaf | skeleton | warning
    path: str = ""
    start_line: int = 0

    @property
    def stability(self) -> int:
        if self.kind == "skeleton":
            return STABILITY["skeleton3"] if self.level >= 3 else STABILITY["skeleton2"]
        if self.kind == "warning":
            return STABILITY["warning"]
        return STABILITY["leaf0"] if self.level == 0 else STABILITY["leaf1"]


@dataclass
class Page:
    key: str
    level: int
    kind: str
    text: str
    tokens: int
    epoch: int
    turn: int


@dataclass
class PagingResult:
    pages: list[Page]
    appended: list[str]
    promoted: list[str]
    epoch: int
    turn: int
    compacted: bool
    context: str
    kv_hit_rate: float
    resident_tokens: int


@dataclass
class PagerState:
    epoch: int = 0
    turn: int = 0
    pages: list[dict] = field(default_factory=list)
    last_context: str = ""

    @classmethod
    def from_dict(cls, d: dict | None) -> "PagerState":
        if not d:
            return cls()
        return cls(d.get("epoch", 0), d.get("turn", 0), d.get("pages", []), d.get("last_context", ""))


def order_by_stability(reqs: Sequence[PageRequest]) -> list[PageRequest]:
    return sorted(reqs, key=lambda r: (r.stability, r.path, r.start_line, r.key))


def render_context(pages: Sequence[Page], epoch: int) -> str:
    body = "\n".join(p.text.rstrip("\n") for p in pages)
    return f'<aegis_context epoch="{epoch}">\n{body}\n</aegis_context>\n'


class SemanticPager:
    """Session-scoped append-only page table (state persisted via the store's ``sessions`` table)."""

    def __init__(self, store=None, overflow: float = 2.0, pin_frac: float = 0.75):
        self.store = store
        self.overflow = overflow
        self.pin_frac = pin_frac
        self._mem: dict[str, dict] = {}

    def _load(self, sid: str) -> PagerState:
        d = self.store.load_session("pager:" + sid) if self.store is not None else self._mem.get(sid)
        return PagerState.from_dict(d)

    def _save(self, sid: str, st: PagerState) -> None:
        d = asdict(st)
        if self.store is not None:
            self.store.save_session("pager:" + sid, d)
        else:
            self._mem[sid] = d

    def page(self, session_id: str, requests: Sequence[PageRequest], budget: int) -> PagingResult:
        st = self._load(session_id)
        st.turn += 1
        resident = [Page(**p) for p in st.pages]
        by_key = {p.key: p for p in resident}
        appended: list[str] = []
        promoted: list[str] = []

        new_reqs: list[PageRequest] = []
        for r in requests:
            cur = by_key.get(r.key)
            if cur is None:
                new_reqs.append(r)
            elif r.level < cur.level:  # promotion: keep old page in place, append the richer one
                new_reqs.append(r)
                promoted.append(r.key)
        pages = list(resident)
        for r in order_by_stability(new_reqs):
            pages.append(Page(r.key, r.level, r.kind, r.text, count_tokens(r.text), st.epoch, st.turn))
            appended.append(r.key)

        total = sum(p.tokens for p in pages)
        compacted = False
        cap = self.overflow * budget
        if st.turn > 1 and total > cap:
            compacted = True
            pinned: list[Page] = []
            acc = 0
            for p in pages:  # immutable pinned prefix: leading pages within pin_frac of the cap
                if acc + p.tokens > self.pin_frac * cap:
                    break
                pinned.append(p)
                acc += p.tokens
            have: dict[str, int] = {}
            for p in pinned:
                have[p.key] = min(have.get(p.key, 9), p.level)
            todo = [r for r in requests if r.key not in have or r.level < have[r.key]]
            todo.sort(key=lambda r: 0 if r.kind == "skeleton" else 1)  # stable: keeps relevance order of leaves
            room = cap - acc
            tail: list[Page] = []
            for r in todo:
                t = count_tokens(r.text)
                if t <= room:
                    tail.append(Page(r.key, r.level, r.kind, r.text, t, st.epoch, st.turn))
                    room -= t
            stab = {"skeleton": 0, "leaf": 2, "warning": 4}
            tail.sort(key=lambda p: (stab.get(p.kind, 3) + (1 if p.kind == "leaf" and p.level == 0 else 0)))
            pages = pinned + tail
            appended = [p.key for p in tail]
            total = sum(p.tokens for p in pages)

        context = render_context(pages, st.epoch)
        hit = prefix_token_hit_rate(st.last_context, context) if st.last_context else 0.0
        st.pages = [asdict(p) for p in pages]
        st.last_context = context
        self._save(session_id, st)
        return PagingResult(pages, appended, promoted, st.epoch, st.turn, compacted, context, hit, total)

    def resident(self, session_id: str) -> dict[str, int]:
        """key -> best resident level for the session (empty for a new session)."""
        out: dict[str, int] = {}
        for p in self._load(session_id).pages:
            out[p["key"]] = min(out.get(p["key"], 9), p["level"])
        return out

    def resident_tokens(self, session_id: str) -> int:
        return sum(p["tokens"] for p in self._load(session_id).pages)

    def reset(self, session_id: str) -> None:
        self._save(session_id, PagerState())
