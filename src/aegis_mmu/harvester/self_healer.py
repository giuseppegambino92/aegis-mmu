"""Zero-downtime schema self-healing via maximum-weight bipartite matching.

Each synthesised tool carries a :class:`ContractShadow`: per expected leaf field its canonical
JSONPath, key, parent key, depth, type, sample values, value statistics and a structural
embedding. When a live payload no longer validates, every leaf of the new payload becomes a
candidate and we build the composite similarity matrix

    S_ij = 0.55 * name_ij + 0.30 * value_ij + 0.15 * struct_ij

* ``name``   cosine of hashed key/path embeddings (subwords, char-trigrams, concept lexicon),
* ``value``  type compatibility x distribution compatibility (sample overlap, numeric range
             overlap and magnitude, string shape signatures, length),
* ``struct`` depth proximity, array-membership agreement and sibling-concept overlap,

and solve ``scipy.optimize.linear_sum_assignment(-S)`` (Hungarian / Jonker-Volgenant) for the
optimal one-to-one remapping. Matches below ``min_confidence`` stay unresolved. The healed
payload is rebuilt in the *expected* shape (array items re-zipped by index) and the learned
aliases are written back to the shadow so the next drifted payload takes the O(n) fast path.
"""

from __future__ import annotations

import base64
import math
import re
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from ..embeddings.lexicon import concept_of, subword_tokens
from .taint_graph import canonical_path, flatten_json, last_key

W_NAME, W_VALUE, W_STRUCT = 0.55, 0.30, 0.15
_ISO_DT = re.compile(r"^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}")


def _ltype(v: Any) -> str:
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "bool"
    if isinstance(v, int):
        return "int"
    if isinstance(v, float):
        return "float"
    if isinstance(v, str):
        return "str"
    return "other"


def shape_signature(s: str) -> str:
    """Collapse character classes: 'AZ610' -> 'A9', '2026-12-03T10:15:00Z' -> '9-9-9A9:9:9A'."""
    out: list[str] = []
    for ch in s[:64]:
        c = "A" if ch.isupper() else "a" if ch.islower() else "9" if ch.isdigit() else ch
        if not out or out[-1] != c:
            out.append(c)
    return "".join(out)


def _tokens(path: str) -> list[str]:
    return [p for p in re.split(r"[.\[\]*$\"]+", path) if p]


def _parent(path: str) -> str:
    toks = _tokens(path)
    return toks[-2] if len(toks) >= 2 else ""


def _depth(path: str) -> int:
    return len(_tokens(path))


def _wild(path: str) -> int:
    return path.count("[*]")


def _enc(v: np.ndarray) -> str:
    return base64.b64encode(np.asarray(v, dtype=np.float16).tobytes()).decode()


def _dec(s: str) -> np.ndarray:
    return np.frombuffer(base64.b64decode(s), dtype=np.float16).astype(np.float32)


@dataclass
class FieldShadow:
    path: str
    key: str
    parent: str
    depth: int
    leaf_type: str
    samples: list[Any]
    stats: dict[str, Any]
    siblings: list[str]
    key_emb: str = ""
    path_emb: str = ""
    aliases: list[str] = field(default_factory=list)


@dataclass
class ContractShadow:
    tool_name: str
    fields: list[FieldShadow]
    version: int = 1

    def to_dict(self) -> dict:
        return {"tool_name": self.tool_name, "version": self.version, "fields": [asdict(f) for f in self.fields]}

    @classmethod
    def from_dict(cls, d: dict) -> "ContractShadow":
        return cls(d["tool_name"], [FieldShadow(**f) for f in d["fields"]], d.get("version", 1))

    def paths(self) -> list[str]:
        return [f.path for f in self.fields]


@dataclass
class FieldMapping:
    expected: str
    actual: str | None
    confidence: float
    name_sim: float = 0.0
    value_sim: float = 0.0
    struct_sim: float = 0.0
    changed: bool = False


@dataclass
class SchemaDriftReport:
    tool_name: str
    drift_detected: bool
    healed: bool
    mappings: list[FieldMapping]
    unresolved: list[str]
    new_fields: list[str]
    latency_ms: float
    strategy: str  # exact | alias | hungarian
    matrix: dict | None = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["mappings"] = [asdict(m) for m in self.mappings]
        return d


# ----------------------------------------------------------------------------- leaf statistics


def _collect(payload: Any) -> dict[str, list[Any]]:
    """canonical path -> list of observed values (across array items)."""
    out: dict[str, list[Any]] = defaultdict(list)
    for p, v in flatten_json(payload):
        out[canonical_path(p)].append(v)
    return out


def _stats(values: list[Any]) -> dict[str, Any]:
    nums = [float(v) for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)]
    strs = [v for v in values if isinstance(v, str)]
    st: dict[str, Any] = {}
    if nums:
        st.update(min=min(nums), max=max(nums), mean=sum(nums) / len(nums))
    if strs:
        st["shape"] = Counter(shape_signature(s) for s in strs).most_common(1)[0][0]
        st["len"] = sum(len(s) for s in strs) / len(strs)
        st["datetime"] = all(_ISO_DT.match(s) for s in strs)
    return st


def _type_of(values: list[Any]) -> str:
    types = Counter(_ltype(v) for v in values if v is not None)
    if not types:
        return "null"
    if set(types) <= {"int", "float"} and "float" in types:
        return "float"
    return types.most_common(1)[0][0]


def _name_texts(path: str) -> tuple[str, str]:
    key = last_key(path)
    parent = _parent(path)
    return key, f"{key} {key} {parent} " + " ".join(_tokens(path))


def build_shadow(tool_name: str, payload: Any, engine) -> ContractShadow:
    leaves = _collect(payload)
    by_parent: dict[str, list[str]] = defaultdict(list)
    for p in leaves:
        by_parent[p.rsplit(".", 1)[0]].append(last_key(p))
    paths = list(leaves)
    keys = [_name_texts(p)[0] for p in paths]
    ptxt = [_name_texts(p)[1] for p in paths]
    kemb = engine.embed(keys, use_stopwords=False) if hasattr(engine, "embed") else None
    pemb = engine.embed(ptxt, use_stopwords=False)
    fields = []
    for i, p in enumerate(paths):
        vals = leaves[p]
        uniq = list(dict.fromkeys(v for v in vals if not isinstance(v, (dict, list))))[:8]
        sib = [k for k in by_parent[p.rsplit(".", 1)[0]] if k != last_key(p)]
        fields.append(FieldShadow(p, last_key(p), _parent(p), _depth(p), _type_of(vals), uniq, _stats(vals), sib,
                                  _enc(kemb[i]), _enc(pemb[i])))
    return ContractShadow(tool_name, fields)


# ----------------------------------------------------------------------------- similarity


def _type_compat(a: str, b: str) -> float:
    if a == b:
        return 1.0
    if {a, b} <= {"int", "float"}:
        return 0.8
    if "null" in (a, b):
        return 0.3
    return 0.05


def _num_dist(sa: dict, sb: dict) -> float:
    if "min" not in sa or "min" not in sb:
        return 0.5
    lo, hi = max(sa["min"], sb["min"]), min(sa["max"], sb["max"])
    span = max(sa["max"], sb["max"]) - min(sa["min"], sb["min"])
    overlap = max(0.0, hi - lo) / span if span > 0 else 1.0
    ma, mb = abs(sa["mean"]) + 1e-9, abs(sb["mean"]) + 1e-9
    magnitude = math.exp(-abs(math.log(ma / mb)))
    return 0.4 * overlap + 0.6 * magnitude


@dataclass
class _Cand:
    path: str
    values: list[Any]
    ltype: str
    stats: dict
    strset: set
    sib_concepts: set
    depth: int
    wild: int


def _str_dist(fa: FieldShadow, c: "_Cand", sv_a: set) -> float:
    sa, sb = fa.stats, c.stats
    sv_b = c.strset
    jac = len(sv_a & sv_b) / len(sv_a | sv_b) if sv_a and sv_b else 0.0
    shape = 1.0 if sa.get("shape") and sa.get("shape") == sb.get("shape") else 0.0
    if not shape and sa.get("datetime") and sb.get("datetime"):
        shape = 1.0
    la, lb = sa.get("len", 0.0), sb.get("len", 0.0)
    length = min(la, lb) / max(la, lb) if max(la, lb) > 0 else 1.0
    return max(jac, 0.65 * shape + 0.35 * length)


def _value_sim(fa: FieldShadow, c: _Cand, sv_a: set) -> float:
    tb = c.ltype
    tc = _type_compat(fa.leaf_type, tb)
    if tc < 0.1:
        return tc * 0.35
    if fa.leaf_type in ("int", "float") and tb in ("int", "float"):
        d = _num_dist(fa.stats, c.stats)
        if sv_a & c.strset:
            d = max(d, 0.95)
    elif fa.leaf_type == "str" and tb == "str":
        d = _str_dist(fa, c, sv_a)
    elif fa.leaf_type == tb:
        d = 1.0
    else:
        d = 0.0
    return tc * (0.35 + 0.65 * d)


def _concepts(keys: list[str]) -> set[str]:
    out: set[str] = set()
    for k in keys:
        for t in subword_tokens(k):
            out.add(concept_of(t) or t)
    return out


def _struct_sim(fa: FieldShadow, c: _Cand, ca: set) -> float:
    depth = 1.0 / (1.0 + abs(fa.depth - c.depth))
    arr = 1.0 if (_wild(fa.path) > 0) == (c.wild > 0) else 0.0
    cb = c.sib_concepts
    sib = len(ca & cb) / len(ca | cb) if ca and cb else 0.5
    return 0.35 * depth + 0.35 * arr + 0.30 * sib


# ----------------------------------------------------------------------------- reconstruction


def _ptokens(path: str) -> list[str]:
    """'$.a.b[*].c' -> ['a', 'b', '*', 'c']"""
    out: list[str] = []
    for part in re.finditer(r"\.([^.\[\]]+)|\[(\*|\d+)\]", path):
        out.append(part.group(1).strip('"') if part.group(1) is not None else "*")
    return out


def _resolve_all(obj: Any, toks: list[str], idx: tuple = ()) -> list[tuple[tuple, Any]]:
    if not toks:
        return [(idx, obj)]
    t, rest = toks[0], toks[1:]
    if t == "*":
        if not isinstance(obj, list):
            return []
        out = []
        for i, item in enumerate(obj):
            out.extend(_resolve_all(item, rest, idx + (i,)))
        return out
    if isinstance(obj, dict) and t in obj:
        return _resolve_all(obj[t], rest, idx)
    return []


def _set_path(out: dict, toks: list[str], idx: tuple, value: Any) -> None:
    cur: Any = out
    k = 0
    for j, t in enumerate(toks):
        last = j == len(toks) - 1
        nxt_is_list = not last and toks[j + 1] == "*"
        if t == "*":
            i = idx[k]
            k += 1
            while len(cur) <= i:
                cur.append([] if nxt_is_list else {})
            if last:
                cur[i] = value
            else:
                cur = cur[i]
            continue
        if last:
            cur[t] = value
        else:
            if t not in cur:
                cur[t] = [] if nxt_is_list else {}
            cur = cur[t]


def remap(payload: Any, mapping: dict[str, str | None], expected: list[str]) -> dict:
    out: dict = {}
    for p in expected:
        q = mapping.get(p)
        toks_p = _ptokens(p)
        if q is None:
            continue
        vals = _resolve_all(payload, _ptokens(q))
        for idx, v in vals:
            if len(idx) != _wild(p):
                idx = idx[: _wild(p)] + (0,) * max(0, _wild(p) - len(idx))
            _set_path(out, toks_p, idx, v)
    return out


# ----------------------------------------------------------------------------- healer


class SchemaHealer:
    def __init__(self, engine, min_confidence: float = 0.40):
        self.engine = engine
        self.min_confidence = min_confidence

    def validate(self, payload: Any, shadow: ContractShadow) -> list[str]:
        """Expected paths that are missing or type-incompatible in ``payload``."""
        leaves = _collect(payload)
        bad = []
        for f in shadow.fields:
            vals = leaves.get(f.path)
            if vals is None or _type_compat(f.leaf_type, _type_of(vals)) < 0.5:
                bad.append(f.path)
        return bad

    def heal(self, payload: Any, shadow: ContractShadow) -> tuple[Any, SchemaDriftReport]:
        t0 = time.perf_counter()
        expected = shadow.paths()
        bad = self.validate(payload, shadow)
        if not bad:
            rep = SchemaDriftReport(shadow.tool_name, False, True,
                                    [FieldMapping(p, p, 1.0, 1.0, 1.0, 1.0) for p in expected], [], [],
                                    (time.perf_counter() - t0) * 1e3, "exact")
            return payload, rep
        leaves = _collect(payload)

        # 1. alias fast path (learned from previous healings)
        alias_map: dict[str, str | None] = {}
        for f in shadow.fields:
            if f.path in leaves:
                alias_map[f.path] = f.path
                continue
            hit = next((a for a in f.aliases if a in leaves and _type_compat(f.leaf_type, _type_of(leaves[a])) >= 0.5), None)
            alias_map[f.path] = hit
        if all(v is not None for v in alias_map.values()):
            healed = remap(payload, alias_map, expected)
            maps = [FieldMapping(p, q, 1.0, changed=p != q) for p, q in alias_map.items()]
            rep = SchemaDriftReport(shadow.tool_name, True, True, maps, [], [], (time.perf_counter() - t0) * 1e3, "alias")
            return healed, rep

        # 2. Hungarian matching on the composite similarity matrix
        cands = list(leaves)
        by_parent: dict[str, list[str]] = defaultdict(list)
        for p in cands:
            by_parent[p.rsplit(".", 1)[0]].append(last_key(p))
        ck = self.engine.embed([_name_texts(p)[0] for p in cands], use_stopwords=False)
        cp = self.engine.embed([_name_texts(p)[1] for p in cands], use_stopwords=False)
        ek = np.stack([_dec(f.key_emb) for f in shadow.fields])
        ep = np.stack([_dec(f.path_emb) for f in shadow.fields])
        name = np.clip(0.7 * (ek @ ck.T) + 0.3 * (ep @ cp.T), 0.0, 1.0)
        n, m = len(shadow.fields), len(cands)
        profiles = []
        for p in cands:
            vals = leaves[p]
            sib = [k for k in by_parent[p.rsplit(".", 1)[0]] if k != last_key(p)]
            profiles.append(_Cand(p, vals, _type_of(vals), _stats(vals), {str(v) for v in vals if not isinstance(v, (dict, list))},
                                  _concepts(sib), _depth(p), _wild(p)))
        value = np.zeros((n, m))
        struct = np.zeros((n, m))
        wild_ok = np.zeros((n, m), dtype=bool)
        for i, f in enumerate(shadow.fields):
            sv_a = {str(v) for v in f.samples}
            ca = _concepts(f.siblings)
            fw = _wild(f.path)
            for j, c in enumerate(profiles):
                if fw != c.wild:
                    continue  # array nesting must agree
                wild_ok[i, j] = True
                value[i, j] = _value_sim(f, c, sv_a)
                struct[i, j] = _struct_sim(f, c, ca)
        S = np.where(wild_ok, W_NAME * name + W_VALUE * value + W_STRUCT * struct, 0.0)
        for i, f in enumerate(shadow.fields):
            j = cands.index(f.path) if f.path in leaves else -1
            if j >= 0 and value[i, j] > 0.5:
                S[i, j] = 1.0 + S[i, j]  # unchanged fields dominate
        rows, cols = linear_sum_assignment(-S)
        mapping: dict[str, str | None] = {p: None for p in expected}
        maps: list[FieldMapping] = []
        used: set[int] = set()
        for i, j in zip(rows, cols):
            f = shadow.fields[i]
            conf = float(min(S[i, j], 1.0))
            if conf >= self.min_confidence:
                mapping[f.path] = cands[j]
                used.add(j)
            maps.append(FieldMapping(f.path, cands[j] if conf >= self.min_confidence else None, round(conf, 4),
                                     round(float(name[i, j]), 4), round(float(value[i, j]), 4),
                                     round(float(struct[i, j]), 4), cands[j] != f.path))
        mapped = {m.expected for m in maps}
        maps += [FieldMapping(p, None, 0.0) for p in expected if p not in mapped]
        unresolved = [p for p, q in mapping.items() if q is None]
        healed = remap(payload, mapping, expected)
        top = np.argsort(-S, axis=1)[:, :3]
        matrix = {"rows": expected, "cols": cands,
                  "top": [[(cands[j], round(float(min(S[i, j], 1.0)), 3)) for j in top[i]] for i in range(n)]}
        rep = SchemaDriftReport(shadow.tool_name, True, not unresolved, maps, unresolved,
                                [cands[j] for j in range(m) if j not in used], (time.perf_counter() - t0) * 1e3,
                                "hungarian", matrix)
        for f in shadow.fields:  # learn aliases
            q = mapping.get(f.path)
            if q and q != f.path and q not in f.aliases:
                f.aliases.append(q)
        shadow.version += 1
        return healed, rep
