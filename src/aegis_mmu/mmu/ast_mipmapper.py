"""Holographic AST mipmapping: every source file is pre-computed at four resolutions.

Resolution model
----------------
* Leaf units (functions, methods, config/attribute blocks) carry three renderings:
    - **L0** exact source,
    - **L1** signature + decorators + first docstring line + top-level control-flow skeleton
      (``if/elif/else/for/while/try/except/finally/with/match`` headers and top-level
      ``return``/``raise``/``throw``) with all other statement runs replaced by a fold marker
      ``[~aegis:<vec_id>~]`` (only when the fold saves tokens),
    - **L2** one-line signature with an empty body (no docstring).
* Files carry a skeleton level:
    - **L2** a one-line ``deps:`` import summary + one-line container headers + public signatures,
    - **L3** a single ``[L3 module: top symbols | vec_id]`` descriptor line.

A rendered file is the skeleton with selected leaves spliced in at L0/L1. When every leaf is at
L0 the original source is returned byte-for-byte. Python output is guaranteed to stay parseable
(fold markers are ``...  # comment`` statements); brace-language output stays brace-balanced.

Python is parsed with :mod:`ast`; TypeScript/JavaScript, Kotlin, Swift, Rust, Go, Java, C#, C/C++
are parsed by a string/comment-aware brace lexer plus a declaration state machine.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import re
import textwrap
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Iterable

from ..embeddings.lexicon import split_identifier
from ..tokens import count_tokens, estimate_tokens, line_prefix, line_prefix_estimate

# ----------------------------------------------------------------------------- languages

EXT_LANG = {
    ".py": "python", ".pyi": "python",
    ".ts": "typescript", ".tsx": "typescript", ".mts": "typescript", ".cts": "typescript",
    ".js": "javascript", ".jsx": "javascript", ".mjs": "javascript", ".cjs": "javascript",
    ".kt": "kotlin", ".kts": "kotlin",
    ".swift": "swift",
    ".rs": "rust",
    ".go": "go",
    ".java": "java", ".scala": "java",
    ".cs": "csharp",
    ".c": "c", ".h": "c", ".cc": "cpp", ".cpp": "cpp", ".hpp": "cpp", ".hh": "cpp", ".m": "c",
}
FENCE_LANG = {
    "py": "python", "python": "python", "python3": "python",
    "ts": "typescript", "typescript": "typescript", "tsx": "typescript",
    "js": "javascript", "javascript": "javascript", "jsx": "javascript",
    "kt": "kotlin", "kotlin": "kotlin", "swift": "swift", "rs": "rust", "rust": "rust",
    "go": "go", "golang": "go", "java": "java", "cs": "csharp", "csharp": "csharp",
    "c": "c", "cpp": "cpp", "c++": "cpp",
}
BRACE_LANGS = {"typescript", "javascript", "kotlin", "swift", "rust", "go", "java", "csharp", "c", "cpp"}


def detect_language(path: str | None = None, fence: str | None = None, text: str | None = None) -> str:
    if fence:
        lang = FENCE_LANG.get(fence.strip().lower())
        if lang:
            return lang
    if path:
        lang = EXT_LANG.get(PurePosixPath(path).suffix.lower())
        if lang:
            return lang
    if text is not None:
        if re.search(r"^\s*(def|class)\s+\w+.*:\s*$", text, re.M) and not re.search(r"[{};]\s*$", text, re.M):
            return "python"
        if re.search(r"^\s*fun\s+\w+", text, re.M):
            return "kotlin"
        if re.search(r"^\s*(pub\s+)?fn\s+\w+", text, re.M):
            return "rust"
        if re.search(r"^\s*func\s+\w+", text, re.M):
            return "swift" if "->" in text or "let " in text else "go"
        if re.search(r"^\s*(export\s+)?(function|const|interface|class)\s+\w+", text, re.M):
            return "typescript"
    return "text"


def comment_prefix(lang: str) -> str:
    return "#" if lang in ("python", "text") else "//"


def make_vec_id(*parts: str) -> str:
    return hashlib.blake2b("\x00".join(parts).encode("utf-8"), digest_size=5).hexdigest()


# ----------------------------------------------------------------------------- data model


@dataclass
class CodeUnit:
    vec_id: str
    path: str
    language: str
    qualname: str
    kind: str  # function | method | config | type
    start_line: int  # 1-based inclusive (first decorator / annotation)
    end_line: int
    indent: str
    lead: str  # comment / blank lines directly above the unit
    l0: str
    l1: str
    l2: str
    l3: str
    signature: str
    doc: str
    role: str
    parent: str | None
    is_public: bool
    identifiers: list[str] = field(default_factory=list)
    tokens: dict[int, int] = field(default_factory=dict)
    context: str = ""  # enclosing container header / doc (contextual chunk header for retrieval)
    extras: str = ""  # literals / status codes / exception names / comments from the body (retrieval only)
    calls: list[str] = field(default_factory=list)  # names this unit calls (ast.Call / `name(` patterns)

    @property
    def name(self) -> str:
        return self.qualname.rsplit(".", 1)[-1]

    def level_text(self, level: int) -> str:
        return {0: self.l0, 1: self.l1, 2: self.l2, 3: self.l3}[level]

    def embed_text(self) -> str:
        return f"{self.path} {self.qualname}\n{self.context}\n{self.lead}{self.l0}"

    def composite_text(self, calls: list[str] | None = None, called_by: list[str] | None = None) -> str:
        """Composite indexing document: names split into words, signature + docstring, enclosing
        container, body literals/codes/exceptions/comments, the 1-hop call neighbourhood, then source."""
        module = PurePosixPath(self.path).with_suffix("").as_posix()
        words = " ".join(split_identifier(re.sub(r"[^A-Za-z0-9_]+", "_", f"{module}_{self.qualname}")))
        parts = [f"{self.path} :: {self.qualname} ({words})"]
        if self.signature:
            parts.append(self.signature[:300])
        if self.doc:
            parts.append(self.doc[:400])
        if self.context:
            parts.append(self.context[:300])
        if not self.extras:  # computed lazily: only the (opt-in) composite index needs it
            self.extras = _extract_extras(self.l0, self.language)
        if self.extras:
            parts.append(self.extras)
        if calls or called_by:
            parts.append(f"calls: [{', '.join((calls or [])[:12])}] | called_by: [{', '.join((called_by or [])[:12])}]")
        parts.append(self.lead + self.l0)
        return "\n".join(parts)


@dataclass
class Seg:
    kind: str  # import | leaf | container | text
    start: int  # 0-based inclusive
    end: int
    lead: list[str] = field(default_factory=list)
    unit: CodeUnit | None = None
    header_end: int = -1  # container: last header line (0-based)
    footer: tuple[int, int] | None = None
    footer_lead: list[str] = field(default_factory=list)
    children: list["Seg"] = field(default_factory=list)
    body_indent: str = ""
    doc_range: tuple[int, int] | None = None  # python container docstring lines
    name: str = ""
    compact_header: str = ""  # one-line container header used in pure-L2 renders


@dataclass
class FileMipmap:
    path: str
    language: str
    source: str
    lines: list[str]
    root: list[Seg]
    trailing: list[str]
    units: list[CodeUnit]
    file_vec_id: str
    summary: str
    tokens_source: int = 0
    skeleton: dict[int, str] = field(default_factory=dict)
    skeleton_tokens: dict[int, int] = field(default_factory=dict)
    deps_line: str = ""
    tokens_estimated: bool = False
    _exact_source_tokens: int | None = None

    def exact_tokens(self) -> int:
        """Exact cl100k count of the source (cached); used for every reported metric."""
        if not self.tokens_estimated:
            return self.tokens_source
        if self._exact_source_tokens is None:
            self._exact_source_tokens = count_tokens(self.source)
        return self._exact_source_tokens

    # --------------------------------------------------------------- rendering
    def render(self, levels: dict[str, int] | None = None, skeleton_level: int = 2) -> str:
        """Render the file with per-leaf levels (``vec_id -> 0|1``) on top of an L2/L3 skeleton."""
        levels = levels or {}
        if self.units and all(levels.get(u.vec_id, 9) == 0 for u in self.units):
            return self.source
        out: list[str] = []
        if skeleton_level >= 3:
            out.append(self.l3_line())
        self._deps_done = False
        self._render_children(self.root, levels, skeleton_level, out)
        text = "\n".join(_collapse_blanks(out)).strip("\n")
        return text + "\n" if text else ""

    def l3_line(self) -> str:
        return f"{comment_prefix(self.language)} [L3 {self.summary} | {self.file_vec_id}]"

    def _expanded(self, seg: Seg, levels: dict[str, int]) -> bool:
        if seg.kind == "leaf":
            return levels.get(seg.unit.vec_id, 9) <= 1  # type: ignore[union-attr]
        return any(self._expanded(c, levels) for c in seg.children)

    def _render_children(self, segs: list[Seg], levels: dict[str, int], skel: int, out: list[str]) -> int:
        shown = 0
        any_public = any(s.kind == "leaf" and s.unit.is_public for s in segs)  # type: ignore[union-attr]
        for seg in segs:
            if seg.kind == "import":
                continue  # L2 is an interface skeleton: imports are not part of it
            elif seg.kind == "text":
                if skel <= 2:
                    out.extend(self.lines[seg.start : seg.end + 1])
                    shown += 1
            elif seg.kind == "leaf":
                u = seg.unit
                assert u is not None
                lvl = levels.get(u.vec_id)
                if lvl == 0:
                    out.extend(seg.lead)
                    out.extend(u.l0.split("\n"))
                    shown += 1
                elif lvl == 1:
                    out.extend(_blank_only(seg.lead))
                    out.extend(_doc_summary(seg.lead, self.language, u.indent))
                    out.extend(u.l1.split("\n"))
                    shown += 1
                elif skel <= 2 and u.kind != "config" and (u.is_public or not any_public):
                    if u.l2:
                        out.extend(u.l2.split("\n"))
                        shown += 1
            elif seg.kind == "container":
                expanded = self._expanded(seg, levels)
                if skel <= 2 or expanded:
                    if expanded:
                        out.extend(_blank_only(seg.lead))
                        out.extend(self._container_header(seg))
                    else:
                        header_line = seg.compact_header or self.lines[seg.start]
                        inner_l2: list[str] = []
                        n_inner = self._render_children(seg.children, levels, skel, inner_l2)
                        if not _has_statement(inner_l2):  # no public members: one-line stub
                            if self.language == "python":
                                out.append(header_line + " ...")
                            else:
                                out.append(header_line.rstrip() + " }")
                            shown += 1
                            continue
                        out.append(header_line)
                        out.extend(inner_l2)
                        if seg.footer is not None:
                            out.append(self.lines[seg.footer[0]][: len(self.lines[seg.footer[0]]) - len(self.lines[seg.footer[0]].lstrip())] + "}")
                        shown += 1 + n_inner
                        continue
                    inner: list[str] = []
                    n = self._render_children(seg.children, levels, skel, inner)
                    out.extend(inner)
                    if self.language == "python" and not _has_statement(inner):
                        out.append(seg.body_indent + "...")
                    if seg.footer is not None:
                        out.extend(self.lines[seg.footer[0] : seg.footer[1] + 1])
                    shown += 1 + n
        return shown

    def _container_header(self, seg: Seg) -> list[str]:
        if seg.doc_range is None:
            return self.lines[seg.start : seg.header_end + 1]
        head = self.lines[seg.start : seg.doc_range[0]]
        doc = self.lines[seg.doc_range[0] : seg.doc_range[1] + 1]
        return head + _compact_docstring(doc, seg.body_indent)

    def render_page(self, unit: CodeUnit, level: int) -> str:
        """Stand-alone, dedented page for one leaf (used by the KV paginator)."""
        doc = _doc_summary(unit.lead.split("\n"), self.language, unit.indent)
        body = {0: unit.lead + unit.l0, 1: "".join(d + "\n" for d in doc) + unit.l1, 2: unit.l2, 3: unit.l3}[level]
        body = textwrap.dedent(body).strip("\n")
        c = comment_prefix(self.language)
        return f"{c} {self.path} :: {unit.qualname} [L{level}] lines {unit.start_line}-{unit.end_line} vec_id={unit.vec_id}\n{body}\n"

    def uniform(self, level: int) -> str:
        if level == 0:
            return self.source
        if level == 1:
            return self.render({u.vec_id: 1 for u in self.units}, 2)
        return self.skeleton[level]

    def unit(self, vec_id: str) -> CodeUnit | None:
        for u in self.units:
            if u.vec_id == vec_id:
                return u
        return None


def _has_statement(lines: list[str]) -> bool:
    return any(ln.strip() and not ln.strip().startswith("#") for ln in lines)


def _blank_only(lead: list[str]) -> list[str]:
    return [""] if any(not ln.strip() for ln in lead) else []


def _doc_summary(lead: list[str], lang: str, indent: str) -> list[str]:
    """First sentence of a leading doc comment as a single-line comment (L1 keeps only the summary)."""
    if lang == "python":
        return []  # python docstrings live inside the body and are compacted there
    doc = _doc_from_comments(lead)
    if not doc:
        return []
    first = re.split(r"(?<=[.!?])\s", doc)[0]
    words = first.split()
    return [f"{indent}{comment_prefix(lang)} " + " ".join(words[:16])]


def _doc_lead(lead: list[str], as_text: bool = False):
    """Keep doc-comment / annotation lines of a lead, collapse blanks."""
    kept: list[str] = []
    blank = False
    for ln in lead:
        s = ln.strip()
        if not s:
            blank = True
            continue
        if s.startswith(("///", "/**", "*", "//", "#", "@")):
            kept.append(ln)
    res = ([""] if blank else []) + kept
    if as_text:
        return "\n".join(res) + ("\n" if res else "")
    return res


def _collapse_blanks(lines: list[str]) -> list[str]:
    out: list[str] = []
    for ln in lines:
        if not ln.strip():
            if out and not out[-1].strip():
                continue
            out.append("")
        else:
            out.append(ln)
    return out


def _compact_docstring(doc_lines: list[str], indent: str) -> list[str]:
    if len(doc_lines) <= 1:
        return doc_lines
    first = doc_lines[0].strip()
    m = re.match(r'^([rubfRUBF]*)("""|\'\'\')(.*)$', first)
    if not m:
        return doc_lines
    prefix, q, rest = m.groups()
    rest = rest.strip()
    if not rest:
        for ln in doc_lines[1:]:
            if ln.strip() and ln.strip() not in ('"""', "'''"):
                rest = ln.strip()
                break
    rest = rest.replace(q, "").rstrip("\\")
    return [f"{indent}{prefix}{q}{rest}{q}"]


# ----------------------------------------------------------------------------- role inference

_VERB_ROLES = [
    (("get", "fetch", "load", "find", "read", "lookup", "list", "query", "search", "resolve"), "reads/looks up"),
    (("set", "update", "save", "write", "store", "persist", "put", "upsert", "patch"), "writes/updates"),
    (("create", "build", "make", "new", "init", "construct", "register", "add", "insert"), "creates"),
    (("delete", "remove", "drop", "evict", "purge", "clear", "cancel", "revoke"), "removes"),
    (("validate", "check", "verify", "ensure", "is", "has", "can", "should", "assert"), "checks"),
    (("compute", "calc", "calculate", "apply", "estimate", "score", "aggregate", "sum", "total"), "computes"),
    (("convert", "to", "parse", "format", "serialize", "deserialize", "encode", "decode", "render", "normalize", "map"), "transforms"),
    (("send", "notify", "emit", "publish", "dispatch", "post", "enqueue", "schedule"), "dispatches"),
    (("handle", "on", "process", "run", "execute", "perform", "dispatch"), "handles"),
    (("test",), "tests"),
    (("refund", "charge", "pay", "capture", "authorize", "settle", "reserve", "release"), "performs"),
]


def infer_role(qualname: str, doc: str, body: str, kind: str) -> str:
    if doc:
        first = re.split(r"(?<=[.!?])\s", doc.strip().split("\n\n")[0].replace("\n", " "))[0]
        words = first.split()
        s = " ".join(words[:14]) + (" …" if len(words) > 14 else "")
        return s.rstrip(".")
    name = qualname.rsplit(".", 1)[-1]
    words = split_identifier(name.strip("_")) or [name]
    role = ""
    if name in ("__init__", "init", "constructor"):
        role = f"initialises {qualname.rsplit('.', 1)[0]}"
    elif kind == "config":
        role = "constants/fields"
    else:
        for verbs, label in _VERB_ROLES:
            if words[0] in verbs:
                obj = " ".join(words[1:]) or (qualname.split(".")[0] if "." in qualname else "")
                role = f"{label} {obj}".strip()
                break
        if not role:
            role = " ".join(words)
    flags = []
    if re.search(r"\bawait\b|\basync\b|\bsuspend\b", body):
        flags.append("async")
    if re.search(r"\b(raise|throw|throws)\b", body):
        flags.append("raises")
    if re.search(r"\b(SELECT|INSERT|UPDATE|DELETE)\b", body):
        flags.append("sql")
    if re.search(r"\b(lock|Lock|mutex|Mutex|synchronized|atomic)\b", body):
        flags.append("sync")
    return role + (f" [{', '.join(flags)}]" if flags else "")


_IDENT_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")


_STR_LIT = re.compile(r'"([^"\\\n]{3,80})"|\'([^\'\\\n]{3,80})\'')
_CODE_LIT = re.compile(r"(?<![\w.])\d{3,5}(?![\w.])")
_EXC_NAME = re.compile(r"\b[A-Z]\w*(?:Error|Exception|Failure|Fault)\b")
_HASH_COMMENT = re.compile(r"#\s?(.{4,160})$", re.M)
_SLASH_COMMENT = re.compile(r"//\s?(.{4,160})$|/\*+\s?(.{4,300}?)\*/", re.M | re.S)


def _extract_extras(l0: str, lang: str) -> str:
    """Retrieval-side summary of a body: string literals, numeric codes, exception names, comments."""
    lits = list(dict.fromkeys((a or b).strip() for a, b in _STR_LIT.findall(l0)))[:12]
    codes = list(dict.fromkeys(_CODE_LIT.findall(l0)))[:8]
    excs = list(dict.fromkeys(_EXC_NAME.findall(l0)))[:8]
    if lang == "python":
        comments = [c.strip() for c in _HASH_COMMENT.findall(l0)]
    else:
        comments = [(a or b).strip() for a, b in _SLASH_COMMENT.findall(l0)]
    comments = [c for c in dict.fromkeys(comments) if c][:8]
    parts = []
    if lits:
        parts.append("literals: " + " | ".join(lits))
    if codes:
        parts.append("codes: " + " ".join(codes))
    if excs:
        parts.append("raises: " + " ".join(excs))
    if comments:
        parts.append("comments: " + " | ".join(comments))
    return "\n".join(parts)


_SAL_PATTERNS = [
    (re.compile(r"\b(?:self|this)\.\w+\s*(?:[-+*/|&]|\?\?)?=(?!=)"), 1.0),   # attribute / state mutation
    (re.compile(r"\w\[[^\]\n]+\]\s*(?:[-+*/]?)=(?!=)"), 1.0),               # subscript write
    (re.compile(r"^\s*(?:with|async with)\b|\.lock\(|\block\b|synchronized|Mutex"), 1.0),  # locks / contexts
    (re.compile(r"(?<![<>=!-])(?:<=|>=|<|>)(?![<>=])"), 0.5),                 # boundary comparisons
    (re.compile(r"(?<![/%])(?://|%)(?![/=])"), 0.5),                           # modulo / floor division
    (re.compile(r"\b(?:ttl|expires?_?\w*|deadline|timeout|now|window|elapsed|seconds|minutes)\b", re.I), 0.5),
]


def structural_salience(l0: str) -> float:
    """Bug-proneness prior: state mutation, subscript writes, locking, boundary comparisons,
    modulo/division and time arithmetic, per body line (wrappers / boilerplate score ~0)."""
    lines = max(1, l0.count("\n"))
    score = sum(w * len(rx.findall(l0)) for rx, w in _SAL_PATTERNS)
    return score / lines


def _identifiers(text: str) -> list[str]:
    return list(dict.fromkeys(_IDENT_RE.findall(text)))


def _fold_marker(lang: str, indent: str, vec_id: str, a: int, b: int, saved: int, first: bool = True) -> str:
    """Fold token. The first fold of a unit carries ``[~aegis:<vec_id>~]`` (expand with
    ``aegis_expand_node``); later folds in the same unit are a bare ``~`` to keep L1 small."""
    tag = f"[~aegis:{vec_id}~]" if first else "~"
    if not first and lang != "python":
        return f"{indent}/* … */"
    if lang == "python":
        return f"{indent}...  # {tag}"
    return f"{indent}/* {tag} */"


# ============================================================================= Python


_PY_CONTROL = (ast.If, ast.For, ast.AsyncFor, ast.While, ast.Try, ast.With, ast.AsyncWith)
if hasattr(ast, "TryStar"):
    _PY_CONTROL = _PY_CONTROL + (ast.TryStar,)  # type: ignore[attr-defined]
if hasattr(ast, "Match"):
    _PY_CONTROL = _PY_CONTROL + (ast.Match,)  # type: ignore[attr-defined]


def _node_start(node: ast.AST) -> int:
    decs = getattr(node, "decorator_list", None)
    if decs:
        return min(d.lineno for d in decs)
    return node.lineno  # type: ignore[attr-defined]


def _docstring_node(body: list[ast.stmt]) -> ast.Expr | None:
    if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant) \
            and isinstance(body[0].value.value, str):
        return body[0]
    return None


class _PyL1:
    """Control-flow skeleton renderer for a Python function body."""

    def __init__(self, lines: list[str], vec_id: str, max_depth: int, P=None):
        self.lines = lines
        self.P = P
        self.vec_id = vec_id
        self.max_depth = max_depth
        self.out: list[str] = []
        self.last_emitted = 0  # 1-based line number
        self.marked = False

    def emit_lines(self, a: int, b: int) -> None:
        a = max(a, self.last_emitted + 1)
        if a > b:
            return
        self.out.extend(self.lines[a - 1 : b])
        self.last_emitted = b

    def fold(self, a: int, b: int, indent: str) -> None:
        a = max(a, self.last_emitted + 1)
        if a > b:
            return
        saved = int(self.P[b] - self.P[a - 1]) if self.P is not None else count_tokens("\n".join(self.lines[a - 1 : b]))
        marker = _fold_marker("python", indent, self.vec_id, a, b, saved, first=not self.marked)
        if saved <= count_tokens(marker):  # folding must never cost tokens
            self.out.extend(self.lines[a - 1 : b])
        else:
            self.out.append(marker)
            self.marked = True
        self.last_emitted = b

    def _indent_of(self, lineno: int) -> str:
        ln = self.lines[lineno - 1]
        return ln[: len(ln) - len(ln.lstrip())]

    def _find_kw(self, kw: str, lo: int, hi: int) -> int | None:
        pat = re.compile(rf"^\s*{kw}\b\s*:")
        for ln in range(lo, hi + 1):
            if 1 <= ln <= len(self.lines) and pat.match(self.lines[ln - 1]):
                return ln
        return None

    def block(self, stmts: list[ast.stmt], depth: int) -> None:
        run: list[ast.stmt] = []

        def flush() -> None:
            if run:
                self.fold(_node_start(run[0]), run[-1].end_lineno or run[-1].lineno, self._indent_of(run[0].lineno))
                run.clear()

        for st in stmts:
            if st.lineno <= self.last_emitted and not run:
                continue
            if isinstance(st, _PY_CONTROL) and depth < self.max_depth:
                flush()
                self.control(st, depth)
            elif depth == 0 and (isinstance(st, (ast.Return, ast.Raise)) or (
                isinstance(st, ast.Expr) and isinstance(st.value, (ast.Yield, ast.YieldFrom))
            )):
                flush()
                self.emit_lines(st.lineno, st.end_lineno or st.lineno)
            elif isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                flush()
                body0 = st.body[0]
                if body0.lineno == st.lineno:
                    self.emit_lines(_node_start(st), st.end_lineno or st.lineno)
                else:
                    self.emit_lines(_node_start(st), body0.lineno - 1)
                    self.fold(body0.lineno, st.end_lineno or body0.lineno, self._indent_of(body0.lineno))
            else:
                run.append(st)
        flush()

    def _header_then_body(self, header_line: int, body: list[ast.stmt], depth: int) -> None:
        if not body:
            return
        if body[0].lineno == header_line:  # `if x: return y` on one line
            self.emit_lines(header_line, body[-1].end_lineno or body[-1].lineno)
            return
        self.emit_lines(header_line, body[0].lineno - 1)
        self.block(body, depth + 1)

    def _else(self, prev_end: int, orelse: list[ast.stmt], depth: int, kw: str = "else") -> None:
        if not orelse:
            return
        ln = self._find_kw(kw, prev_end + 1, orelse[0].lineno)
        if ln is None:
            return
        self._header_then_body(ln, orelse, depth)

    def control(self, st: ast.stmt, depth: int) -> None:
        if isinstance(st, ast.If):
            node: ast.stmt = st
            self._header_then_body(node.lineno, node.body, depth)
            while True:
                orelse = node.orelse  # type: ignore[attr-defined]
                if not orelse:
                    break
                body_end = node.body[-1].end_lineno or node.body[-1].lineno  # type: ignore[attr-defined]
                first = orelse[0]
                if len(orelse) == 1 and isinstance(first, ast.If) and self.lines[first.lineno - 1].lstrip().startswith("elif"):
                    self._header_then_body(first.lineno, first.body, depth)
                    node = first
                    continue
                self._else(body_end, orelse, depth)
                break
        elif isinstance(st, (ast.For, ast.AsyncFor, ast.While)):
            self._header_then_body(st.lineno, st.body, depth)
            self._else(st.body[-1].end_lineno or st.body[-1].lineno, st.orelse, depth)
        elif isinstance(st, (ast.With, ast.AsyncWith)):
            self._header_then_body(st.lineno, st.body, depth)
        elif isinstance(st, ast.Try) or (hasattr(ast, "TryStar") and isinstance(st, ast.TryStar)):  # type: ignore[attr-defined]
            self._header_then_body(st.lineno, st.body, depth)
            end = st.body[-1].end_lineno or st.lineno
            for h in st.handlers:
                self._header_then_body(h.lineno, h.body, depth)
                end = h.body[-1].end_lineno or h.lineno
            if st.orelse:
                self._else(end, st.orelse, depth)
                end = st.orelse[-1].end_lineno or end
            if st.finalbody:
                self._else(end, st.finalbody, depth, kw="finally")
        elif hasattr(ast, "Match") and isinstance(st, ast.Match):  # type: ignore[attr-defined]
            self.emit_lines(st.lineno, st.cases[0].pattern.lineno - 1 if st.cases else st.lineno)
            for case in st.cases:
                self._header_then_body(case.pattern.lineno, case.body, depth)
        else:  # pragma: no cover
            self.emit_lines(st.lineno, st.end_lineno or st.lineno)


class _PythonMipmapper:
    def __init__(self, path: str, source: str, max_depth: int, P=None):
        self.path = path
        self.source = source
        self.lines = source.split("\n")
        self.max_depth = max_depth
        self.units: list[CodeUnit] = []
        self.P = P if P is not None else line_prefix(source)

    def run(self) -> tuple[list[Seg], int]:
        tree = ast.parse(self.source)
        calls = []  # one walk of the whole module: (lineno, called name); exact, ignores strings/comments
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                f = node.func
                name = f.id if isinstance(f, ast.Name) else f.attr if isinstance(f, ast.Attribute) else None
                if name:
                    calls.append((node.lineno, name))
        calls.sort()  # ast.walk is breadth-first; the per-function bisect needs line order
        self._call_lines = [c[0] for c in calls]
        self._call_names = [c[1] for c in calls]
        segs, last = self._segments(tree.body, None, 0)
        return segs, last

    # ------------------------------------------------------------- helpers
    def _text(self, a: int, b: int) -> str:
        return "\n".join(self.lines[a - 1 : b])

    def _indent(self, lineno: int) -> str:
        ln = self.lines[lineno - 1]
        return ln[: len(ln) - len(ln.lstrip())]

    def _lead(self, prev_end: int, start: int) -> list[str]:
        return self.lines[prev_end:start - 1]

    def _segments(self, body: list[ast.stmt], parent: str | None, prev_end: int) -> tuple[list[Seg], int]:
        segs: list[Seg] = []
        config_run: list[ast.stmt] = []
        config_lead_start = prev_end

        def flush_config() -> None:
            nonlocal config_lead_start
            if not config_run:
                return
            seg = self._config_seg(config_run, parent, config_lead_start)
            segs.append(seg)
            config_run.clear()

        for st in body:
            start = _node_start(st)
            end = st.end_lineno or st.lineno
            if isinstance(st, (ast.Import, ast.ImportFrom)) and parent is None:
                flush_config()
                segs.append(Seg("import", start - 1, end - 1, lead=self._lead(prev_end, start)))
            elif isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef)):
                flush_config()
                seg = Seg("leaf", start - 1, end - 1, lead=self._lead(prev_end, start))
                seg.unit = self._function_unit(st, parent, "\n".join(seg.lead))
                segs.append(seg)
            elif isinstance(st, ast.ClassDef):
                flush_config()
                segs.append(self._class_seg(st, parent, prev_end))
            else:
                if not config_run:
                    config_lead_start = prev_end
                config_run.append(st)
            prev_end = end
        flush_config()
        return segs, prev_end

    def _config_seg(self, stmts: list[ast.stmt], parent: str | None, lead_start: int) -> Seg:
        start = _node_start(stmts[0])
        end = stmts[-1].end_lineno or stmts[-1].lineno
        lead = self._lead(lead_start, start)
        l0 = self._text(start, end)
        l1_parts: list[str] = []
        l2_parts: list[str] = []
        for st in stmts:
            s, e = _node_start(st), st.end_lineno or st.lineno
            ind = self._indent(st.lineno)
            text = self._text(s, e)
            short = self._config_l2(st, ind)
            if e - s + 1 <= 6:
                l1_parts.append(text)
            else:
                saved = int(self.P[e] - self.P[s - 1])
                l1_parts.append((short or f"{ind}...") + "  # [~aegis:{VID}~]")
            if short:
                l2_parts.append(short)
        qual_base = f"{parent}.<fields>" if parent else f"<module:{start}>"
        vec_id = make_vec_id(self.path, qual_base, "config", l0)
        l1 = "\n".join(l1_parts).replace("{VID}", vec_id)
        ind = self._indent(stmts[0].lineno)
        is_doc = _docstring_node(stmts) is not None and len(stmts) == 1
        role = "module docstring" if is_doc and parent is None else ("fields" if parent else "module constants/statements")
        l3 = f"{ind}# [L3 {qual_base}: {role} | {vec_id}]"
        unit = CodeUnit(
            vec_id=vec_id, path=self.path, language="python", qualname=qual_base, kind="config",
            start_line=start, end_line=end, indent=ind, lead="\n".join(lead) + ("\n" if lead else ""),
            l0=l0, l1=l1, l2="\n".join(l2_parts), l3=l3, signature="", doc="", role=role, parent=parent,
            is_public=True, identifiers=_identifiers(l0),
        )
        self.units.append(unit)
        return Seg("leaf", start - 1, end - 1, lead=lead, unit=unit)

    def _seg(self, node: ast.AST) -> str:
        """Source text of ``node`` from cached lines (ast column offsets are UTF-8 byte offsets)."""
        l0, l1 = node.lineno, node.end_lineno or node.lineno  # type: ignore[attr-defined]
        c0, c1 = node.col_offset, node.end_col_offset  # type: ignore[attr-defined]
        if l0 == l1:
            return self.lines[l0 - 1].encode("utf-8")[c0:c1].decode("utf-8", "replace")
        first = self.lines[l0 - 1].encode("utf-8")[c0:].decode("utf-8", "replace")
        last = self.lines[l1 - 1].encode("utf-8")[:c1].decode("utf-8", "replace")
        return "\n".join([first] + self.lines[l0 : l1 - 1] + [last])

    def _config_l2(self, st: ast.stmt, ind: str) -> str:
        if isinstance(st, ast.AnnAssign):
            tgt = self._seg(st.target)
            ann = self._seg(st.annotation)
            if "\n" in ann:
                ann = " ".join(ann.split())
            return f"{ind}{tgt}: {ann}" + (" = ..." if st.value is not None else "")
        if isinstance(st, ast.Assign):
            tg = " = ".join(self._seg(t) for t in st.targets)
            if "\n" in tg:
                return ""
            val = self._seg(st.value)
            if "\n" in val or len(val) > 60:
                val = "..."
            return f"{ind}{tg} = {val}"
        doc = _docstring_node([st])
        if doc is not None:
            return "\n".join(_compact_docstring(self._text(st.lineno, st.end_lineno or st.lineno).split("\n"), ind))
        return ""

    def _function_unit(self, fn: ast.FunctionDef | ast.AsyncFunctionDef, parent: str | None, lead: str) -> CodeUnit:
        start = _node_start(fn)
        end = fn.end_lineno or fn.lineno
        qual = f"{parent}.{fn.name}" if parent else fn.name
        kind = "method" if parent else "function"
        l0 = self._text(start, end)
        vec_id = make_vec_id(self.path, qual, kind, l0)
        ind = self._indent(fn.lineno)
        body0 = fn.body[0]
        body0_start = _node_start(body0)
        doc_node = _docstring_node(fn.body)
        doc = ast.get_docstring(fn) or ""
        body_indent = self._indent(body0.lineno) if body0.lineno != fn.lineno else ind + "    "
        if body0.lineno == fn.lineno:  # one-liner
            l1 = l2 = l0
            signature = self.lines[fn.lineno - 1].strip()
        else:
            header = self._text(start, body0_start - 1)
            signature = " ".join(self._text(fn.lineno, body0_start - 1).split())
            r = _PyL1(self.lines, vec_id, self.max_depth, self.P)
            r.out = header.split("\n")
            r.last_emitted = body0_start - 1
            rest = fn.body
            if doc_node is not None:
                r.out.extend(_compact_docstring(self._text(doc_node.lineno, doc_node.end_lineno or doc_node.lineno).split("\n"), body_indent))
                r.last_emitted = doc_node.end_lineno or doc_node.lineno
                rest = fn.body[1:]
            if getattr(self, "l1_mode", "signature") == "signature":
                if rest:  # whole body (guards included) behind one expandable fold
                    r.out.append(_fold_marker("python", body_indent, vec_id, 0, 0, 0))
            else:
                r.block(rest, 0)
            if not _has_statement(r.out[len(header.split("\n")):]):
                r.out.append(body_indent + "...")
            l1 = "\n".join(r.out)
            l2 = _py_signature_line(fn, ind, self._seg)
        role = infer_role(qual, doc, l0, kind)
        l3 = f"{ind}# [L3 {qual}: {role} | {vec_id}]"
        public = not fn.name.startswith("_") or (fn.name.startswith("__") and fn.name.endswith("__"))
        import bisect

        lo = bisect.bisect_left(self._call_lines, fn.lineno)
        hi = bisect.bisect_right(self._call_lines, end)
        calls = dict.fromkeys(n for n in self._call_names[lo:hi] if n != fn.name)
        unit = CodeUnit(
            vec_id=vec_id, path=self.path, language="python", qualname=qual, kind=kind,
            start_line=start, end_line=end, indent=ind, lead=lead + ("\n" if lead else ""),
            l0=l0, l1=l1, l2=l2, l3=l3, signature=signature, doc=doc, role=role, parent=parent,
            is_public=public, identifiers=_identifiers(l0), calls=list(calls),
        )
        self.units.append(unit)
        return unit

    def _class_seg(self, cls: ast.ClassDef, parent: str | None, prev_end: int) -> Seg:
        start = _node_start(cls)
        end = cls.end_lineno or cls.lineno
        qual = f"{parent}.{cls.name}" if parent else cls.name
        body0 = cls.body[0]
        seg = Seg("container", start - 1, end - 1, lead=self._lead(prev_end, start), name=qual)
        if body0.lineno == cls.lineno:  # `class A: pass`
            seg.kind = "leaf"
            l0 = self._text(start, end)
            vec_id = make_vec_id(self.path, qual, "type", l0)
            ind = self._indent(cls.lineno)
            seg.unit = CodeUnit(vec_id, self.path, "python", qual, "type", start, end, ind,
                                "\n".join(seg.lead) + ("\n" if seg.lead else ""), l0, l0, l0,
                                f"{ind}# [L3 {qual}: type | {vec_id}]", l0.strip(), "", "type",
                                parent, not cls.name.startswith("_"), _identifiers(l0))
            self.units.append(seg.unit)
            return seg
        seg.body_indent = self._indent(body0.lineno)
        doc_node = _docstring_node(cls.body)
        header_end = _node_start(body0) - 1
        rest = cls.body
        if doc_node is not None:
            seg.doc_range = (doc_node.lineno - 1, (doc_node.end_lineno or doc_node.lineno) - 1)
            header_end = doc_node.end_lineno or doc_node.lineno
            rest = cls.body[1:]
        seg.header_end = header_end - 1
        bases = [" ".join(self._seg(b).split()) for b in cls.bases] + [" ".join(self._seg(k).split()) for k in cls.keywords]
        seg.compact_header = f"{self._indent(cls.lineno)}class {cls.name}" + (f"({', '.join(bases)})" if bases else "") + ":"
        children, _ = self._segments(rest, qual, header_end)
        seg.children = children
        ctx = "\n".join(self._container_ctx_lines(seg))
        _set_context(children, ctx)
        return seg

    def _container_ctx_lines(self, seg: Seg) -> list[str]:
        head = self.lines[seg.start : (seg.doc_range[0] if seg.doc_range else seg.header_end + 1)]
        if seg.doc_range:
            head += [" ".join(ln.strip() for ln in self.lines[seg.doc_range[0] : seg.doc_range[1] + 1])[:400]]
        return head


# ============================================================================= brace languages


@dataclass
class LineInfo:
    start_depth: int
    end_depth: int
    events: list[tuple[int, str]]
    code: str
    paren_delta: int
    unterminated: bool = False
    nc: str = ""  # original text minus comments (strings intact)
    starts_in_code: bool = True  # lexer was in plain-code state at the start of this line
    nc_open: list = field(default_factory=list)  # positions in ``nc`` of each '{' event


_RUST_CHAR = re.compile(r"'(\\.|\\u\{[0-9a-fA-F]+\}|[^\\'])'")


_REGEX_PREV = set("(,=:[!&|?{};+-*%<>~^")
_REGEX_KW = ("return", "typeof", "case", "do", "else", "in", "of", "void", "yield", "await", "delete", "throw")


def _regex_context(code: list[str]) -> bool:
    """A ``/`` starts a JS regex literal when it sits in operand position."""
    prev = "".join(code).rstrip()
    if not prev:
        return True
    if prev[-1] in _REGEX_PREV:
        return True
    return any(prev.endswith(k) and (len(prev) == len(k) or not (prev[-len(k) - 1].isalnum() or prev[-len(k) - 1] in "_$"))
               for k in _REGEX_KW)


def _scan_js_regex(line: str, i: int) -> int | None:
    """Index just past the regex literal starting at ``line[i] == '/'`` (flags included), or None."""
    j, n, in_class = i + 1, len(line), False
    while j < n:
        ch = line[j]
        if ch == "\\":
            j += 2
            continue
        if ch == "[":
            in_class = True
        elif ch == "]":
            in_class = False
        elif ch == "/" and not in_class:
            j += 1
            while j < n and line[j].isalpha():
                j += 1
            return j
        j += 1
    return None


_CODE_SPECIAL = re.compile(r"[{}()\[\]\"'`/]")
_BLOCK_SPECIAL = re.compile(r"\*/|/\*")
_STR_SPECIAL = {'"': re.compile(r'[\\"]'), "'": re.compile(r"[\\']"), "`": re.compile(r"[\\`$]"),
                "#": re.compile(r"[\\#]")}


def lex_brace(lines: list[str], lang: str) -> list[LineInfo]:
    """Per-line brace depth / events, skipping strings, chars, comments and template literals."""
    infos: list[LineInfo] = []
    depth = 0
    state: tuple | None = None  # ("block",) | ("str", quote, triple)
    tmpl: list[int] = []
    block_nest = 0
    for line in lines:
        i, n = 0, len(line)
        code: list[str] = []
        events: list[tuple[int, str]] = []
        start = depth
        paren = 0
        spans: list[tuple[int, int]] = []  # comment character ranges (removed from ``nc``)
        in_code_at_start = state is None and not tmpl
        cstart: int | None = 0 if (state is not None and state[0] == "block") else None
        while i < n:
            if state is None:  # code: jump to the next significant character at C speed
                m = _CODE_SPECIAL.search(line, i)
                if m is None:
                    code.append(line[i:])
                    break
                if m.start() > i:
                    code.append(line[i : m.start()])
                    i = m.start()
            elif state[0] == "block":
                m = _BLOCK_SPECIAL.search(line, i)
                if m is None:
                    break
                i = m.start()
            else:
                m = _STR_SPECIAL[state[1]].search(line, i)
                if m is None:
                    break
                i = m.start()
            c = line[i]
            if state is not None and state[0] == "block":
                if line.startswith("*/", i):
                    block_nest -= 1
                    i += 2
                    if block_nest <= 0:
                        state = None
                        block_nest = 0
                        spans.append((cstart or 0, i))
                        cstart = None
                    continue
                if line.startswith("/*", i) and lang in ("rust", "swift", "kotlin"):
                    block_nest += 1
                    i += 2
                    continue
                i += 1
                continue
            if state is not None and state[0] == "str":
                _, q, triple = state
                if c == "\\" and not (lang == "rust" and q == "#"):
                    i += 2
                    continue
                if triple and line.startswith(q * 3, i):
                    state = None
                    code.append(q * 3)
                    i += 3
                    continue
                if not triple and c == q:
                    state = None
                    code.append(c)
                    i += 1
                    continue
                if q == "`" and line.startswith("${", i):
                    state = None
                    tmpl.append(0)
                    code.append("${")
                    i += 2
                    continue
                i += 1
                continue
            # ---- code
            if line.startswith("//", i):
                spans.append((i, n))
                break
            if line.startswith("/*", i):
                state = ("block",)
                block_nest = 1
                cstart = i
                i += 2
                continue
            if c == "/" and lang in ("typescript", "javascript") and _regex_context(code):
                j = _scan_js_regex(line, i)
                if j is not None:  # regex literal: its slashes/braces are not code
                    code.append("/r/")
                    i = j
                    continue
            if c in "\"'`":
                if c == "'" and lang == "rust":
                    m = _RUST_CHAR.match(line, i)
                    if m:
                        code.append("' '")
                        i = m.end()
                    else:  # lifetime
                        code.append(c)
                        i += 1
                    continue
                if c == "'" and lang == "swift":
                    code.append(c)
                    i += 1
                    continue
                if c == "`" and lang not in ("typescript", "javascript"):
                    code.append(c)
                    i += 1
                    continue
                if c == '"' and line.startswith('"""', i) and lang in ("kotlin", "swift", "java", "csharp"):
                    state = ("str", '"', True)
                    code.append('"""')
                    i += 3
                    continue
                state = ("str", c, False)
                code.append(c)
                i += 1
                continue
            if c == "{":
                if tmpl:
                    tmpl[-1] += 1
                events.append((i, "{"))
                depth += 1
            elif c == "}":
                if tmpl and tmpl[-1] == 0:
                    tmpl.pop()
                    state = ("str", "`", False)
                    code.append("}")
                    i += 1
                    continue
                if tmpl:
                    tmpl[-1] -= 1
                events.append((i, "}"))
                depth -= 1
            elif c in "([":
                paren += 1
            elif c in ")]":
                paren -= 1
            code.append(c)
            i += 1
        unterminated = False
        if state is not None and state[0] == "str" and not state[2] and state[1] != "`":
            state = None  # unterminated single-line string: recover
            unterminated = True
        if state is not None and state[0] == "block":
            spans.append((cstart or 0, n))
        if spans:
            keep, prev = [], 0
            for a, b in spans:
                keep.append(line[prev:a])
                prev = b
            keep.append(line[prev:])
            nc = "".join(keep)
            nc_open = [col - sum(min(b, col) - a for a, b in spans if a < col) for col, ev in events if ev == "{"]
        else:
            nc = line
            nc_open = [col for col, ev in events if ev == "{"]
        infos.append(LineInfo(start, depth, events, "".join(code), paren, unterminated, nc=nc, nc_open=nc_open,
                              starts_in_code=in_code_at_start))
    return infos


def _prefix_nonneg(text: str) -> bool:
    depth = 0
    for ch in text:
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth < 0:
                return False
    return True


def brace_balanced(text: str, lang: str) -> bool:
    if not any(ch in text for ch in "\"'`/#"):  # fast path: no strings / chars / comments to skip
        return (text.count("{") == text.count("}") and text.count("(") + text.count("[") == text.count(")") + text.count("]")
                and _prefix_nonneg(text))
    lines = text.split("\n")
    infos = lex_brace(lines, lang)
    depth = 0
    for inf in infos:
        for _, ev in inf.events:
            depth += 1 if ev == "{" else -1
            if depth < 0:
                return False
    if depth != 0:
        return False
    total_paren = sum(i.paren_delta for i in infos)
    return total_paren == 0


_MODS = r"(?:(?:public|private|protected|internal|fileprivate|open|static|final|abstract|override|async|readonly|" \
        r"export|default|declare|sealed|data|inner|enum|annotation|value|inline|suspend|operator|infix|tailrec|" \
        r"external|actual|expect|lateinit|mutating|nonmutating|indirect|convenience|required|dynamic|lazy|weak|" \
        r"unowned|optional|virtual|synchronized|native|transient|volatile|partial|unsafe|extern|const|pub(?:\([^)]*\))?|" \
        r"@\w+(?:\([^)]*\))?)\s+)*"
_CONTROL_WORDS = {"if", "for", "while", "switch", "catch", "return", "function", "else", "do", "try", "new", "throw",
                  "await", "typeof", "delete", "super", "this", "when", "guard", "match", "loop", "unsafe", "sizeof",
                  "case", "yield", "import", "export", "require", "defer", "go", "select", "foreach", "using", "lock"}

_CONTAINER_RES = {
    "typescript": [re.compile(r"^" + _MODS + r"(class|interface|enum|namespace|module)\s+([A-Za-z_$][\w$]*)")],
    "kotlin": [re.compile(r"^" + _MODS + r"(class|interface|object)\s+([A-Za-z_]\w*)"),
               re.compile(r"^" + _MODS + r"(companion\s+object)\b\s*([A-Za-z_]\w*)?")],
    "swift": [re.compile(r"^" + _MODS + r"(class|struct|enum|protocol|extension|actor)\s+([A-Za-z_][\w.]*)")],
    "rust": [re.compile(r"^" + _MODS + r"(struct|enum|trait|mod|union)\s+([A-Za-z_]\w*)"),
             re.compile(r"^" + _MODS + r"(impl)(?:<[^{]*?>)?\s+(?:[\w:<>,\s&']+?\s+for\s+)?([A-Za-z_][\w:]*)")],
    "go": [re.compile(r"^type\s+([A-Za-z_]\w*)\s+(struct|interface)\b")],
    "java": [re.compile(r"^" + _MODS + r"(class|interface|enum|record)\s+([A-Za-z_]\w*)")],
    "csharp": [re.compile(r"^" + _MODS + r"(class|interface|enum|struct|record|namespace)\s+([A-Za-z_][\w.]*)")],
    "cpp": [re.compile(r"^(?:template\s*<[^>]*>\s*)?(class|struct|namespace)\s+([A-Za-z_]\w*)")],
}
_CONTAINER_RES["javascript"] = _CONTAINER_RES["typescript"]
_CONTAINER_RES["c"] = [re.compile(r"^(struct|enum|union)\s+([A-Za-z_]\w*)")]

_FUNC_RES = {
    "typescript": [
        re.compile(r"^" + _MODS + r"function\s*\*?\s*([A-Za-z_$][\w$]*)"),
        re.compile(r"^" + _MODS + r"(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*(?::[^=]*)?=\s*(?:async\s+)?(?:function\b|\([^)]*\)?\s*(?::\s*[^=]+)?=>|\(|[A-Za-z_$][\w$]*\s*=>)"),
    ],
    "kotlin": [re.compile(r"^" + _MODS + r"fun\s+(?:<[^>]*>\s*)?(?:[\w.<>?]+\.)?([A-Za-z_]\w*)\s*\("),
               re.compile(r"^" + _MODS + r"(constructor)\s*\(")],
    "swift": [re.compile(r"^" + _MODS + r"(?:class\s+|static\s+)?func\s+([A-Za-z_]\w*|[^\s(]+)\s*(?:<[^>]*>)?\s*\("),
              re.compile(r"^" + _MODS + r"(init|deinit)\b[?!]?\s*(?:<[^>]*>)?\s*[({]")],
    "rust": [re.compile(r"^" + _MODS + r"(?:async\s+|const\s+|unsafe\s+|extern\s+\"[^\"]*\"\s+)*fn\s+([A-Za-z_]\w*)")],
    "go": [re.compile(r"^func\s+(?:\(\s*\w*\s*\*?\s*([A-Za-z_]\w*)[^)]*\)\s*)?([A-Za-z_]\w*)\s*[(\[]")],
    "java": [],
    "csharp": [],
    "c": [re.compile(r"^(?:static\s+|inline\s+|extern\s+|const\s+)*[A-Za-z_][\w\s\*&:<>,]*?[\s\*&]([A-Za-z_]\w*)\s*\([^;]*$")],
}
_FUNC_RES["javascript"] = _FUNC_RES["typescript"]
_FUNC_RES["cpp"] = _FUNC_RES["c"]
_CLASS_METHOD_RE = re.compile(
    r"^" + _MODS + r"(?:get\s+|set\s+|\*\s*)?(?:[\w<>\[\],.?]+\s+)*?([A-Za-z_$#][\w$]*)\s*(?:<[^>]*>)?\s*\(")
_CLASS_ARROW_RE = re.compile(r"^" + _MODS + r"([A-Za-z_$#][\w$]*)\s*(?::[^=]*)?=\s*(?:async\s*)?\([^)]*\)?\s*(?::\s*[^=]+)?=>")
_ANNOTATION_RE = re.compile(r"^(@[\w.]+(\(.*\))?|#!?\[.*\])$")
_IMPORT_RE = re.compile(r"^(import\b|use\s|package\s|#include|#import|using\s|extern\s+crate|from\s+\S+\s+import|"
                        r"export\s+\*\s+from|export\s+\{[^}]*\}\s+from|const\s+\w+\s*=\s*require\()")
_CALL_RE = re.compile(r"([A-Za-z_$][\w$]*)\s*(?:<[^<>()]*>)?\s*\(")
_CTRL_RE = re.compile(r"^(?:\}\s*)?(if|else|for|while|do|switch|when|match|try|catch|finally|guard|loop|select|foreach|unsafe)\b")
_RET_RE = re.compile(r"^(return|throw|yield|panic!|fatalError)\b")


class _BraceMipmapper:
    def __init__(self, path: str, source: str, lang: str, max_depth: int, P=None):
        self.path = path
        self.source = source
        self.lang = lang
        self.lines = source.split("\n")
        self.infos = lex_brace(self.lines, lang)
        self.P = P if P is not None else line_prefix(source)
        self.max_depth = max_depth
        self.units: list[CodeUnit] = []
        self.c = comment_prefix(lang)

    def run(self) -> tuple[list[Seg], int]:
        segs, last = self._parse(0, len(self.lines), 0, None, False, -1)
        return segs, last + 1

    # ------------------------------------------------------------- structure
    def _code(self, i: int) -> str:
        return self.infos[i].code.strip()

    def _stmt_end(self, i: int, b: int, d: int) -> int:
        """Last line of the statement starting at line i (depth/paren balanced)."""
        paren = 0
        j = i
        while j < b:
            paren += self.infos[j].paren_delta
            code = self._code(j)
            if self.infos[j].end_depth <= d and paren <= 0:
                nxt = self._next_code_line(j + 1, b)
                if nxt is None:
                    return j
                ncode = self._code(nxt)
                cont = code.endswith(("(", "=", "+", "-", "&&", "||", "->", "=>", ".", "|")) \
                    or (code.endswith(",") and paren > 0) \
                    or ncode.startswith((".", "?.", "?:", "&&", "||", "+", "-", ")", "]", "=>", "->", ":", "where", "{"))
                if not cont or code.endswith(";"):
                    return j
            j += 1
        return b - 1

    def _next_code_line(self, j: int, b: int) -> int | None:
        while j < b:
            if self._code(j):
                return j
            j += 1
        return None

    def _match_decl(self, code: str, in_class: bool) -> tuple[str, str] | None:
        for rx in _CONTAINER_RES.get(self.lang, []):
            m = rx.match(code)
            if m:
                g = [x for x in m.groups() if x]
                if self.lang == "go":
                    return "container", m.group(1)
                kw, name = (g[0], g[1]) if len(g) >= 2 else (g[0], "Companion")
                if kw.startswith("companion"):
                    name = m.group(2) or "Companion"
                if self.lang == "rust" and kw == "impl":
                    name = name.split("::")[-1]
                return "container", name
        for rx in _FUNC_RES.get(self.lang, []):
            m = rx.match(code)
            if m:
                if self.lang == "go":
                    recv, name = m.group(1), m.group(2)
                    return "function", f"{recv}.{name}" if recv else name
                name = [x for x in m.groups() if x][-1]
                if name in ("if", "for", "while", "switch", "catch", "return"):
                    continue
                return "function", name
        if in_class and self.lang in ("typescript", "javascript", "java", "csharp", "cpp", "kotlin", "swift"):
            for rx in (_CLASS_ARROW_RE, _CLASS_METHOD_RE):
                m = rx.match(code)
                if m and m.group(1) not in _CONTROL_WORDS and not code.startswith(("return", "await", "new ")):
                    if rx is _CLASS_METHOD_RE and ("=" in code.split("(")[0] or self.lang in ("kotlin", "swift")):
                        continue
                    return "function", m.group(1)
        if self.lang in ("java", "csharp") and in_class:
            m = re.match(r"^" + _MODS + r"(?:<[^>]*>\s*)?[\w<>\[\],.?]+\s+([A-Za-z_]\w*)\s*\(", code)
            if m and m.group(1) not in _CONTROL_WORDS:
                return "function", m.group(1)
        return None

    def _block_bounds(self, i: int, b: int, d: int) -> tuple[int, int] | None:
        """(open_line, close_line) of the declaration block starting at i, or None if body-less."""
        paren = 0
        for k in range(i, min(b, i + 16)):
            inf = self.infos[k]
            depth = inf.start_depth
            for _, ev in inf.events:
                if ev == "{" and depth == d:
                    return k, self._close_of(k, b, d)
                depth += 1 if ev == "{" else -1
            paren += inf.paren_delta
            code = self._code(k)
            if paren <= 0 and code:
                nxt = self._next_code_line(k + 1, b)
                ncode = self._code(nxt) if nxt is not None else ""
                cont = code.endswith((",", "(", ":", "->", "=", "&&", "|", "where", "throws", "<")) or \
                    ncode.startswith(("{", ":", "->", "where", "throws", "=>", ".", ",", ")", "&", "|"))
                if code.endswith(";") or not cont:
                    return None
        return None

    def _close_of(self, open_line: int, b: int, d: int) -> int:
        for k in range(open_line, b):
            if self.infos[k].end_depth <= d:
                return k
        return b - 1

    def _lead(self, prev_end: int, start: int) -> list[str]:
        return self.lines[prev_end + 1 : start]

    def _parse(self, a: int, b: int, d: int, parent: str | None, in_class: bool, prev_end: int) -> tuple[list[Seg], int]:
        segs: list[Seg] = []
        config: list[tuple[int, int]] = []
        config_lead_from = prev_end
        annotations_start: int | None = None
        i = a

        def flush_config() -> None:
            if config:
                segs.append(self._config_seg(config, parent, config_lead_from))
                config.clear()

        while i < b:
            code = self._code(i)
            if not code or self.infos[i].start_depth != d:
                i += 1
                continue
            if _ANNOTATION_RE.match(code):
                if annotations_start is None:
                    annotations_start = i
                i += 1
                continue
            decl = self._match_decl(code, in_class)
            start = annotations_start if annotations_start is not None else i
            annotations_start = None
            if decl is None:
                end = self._stmt_end(i, b, d)
                if d == 0 and _IMPORT_RE.match(code):
                    flush_config()
                    segs.append(Seg("import", start, end, lead=self._lead(prev_end, start)))
                else:
                    if not config:
                        config_lead_from = prev_end
                    config.append((start, end))
                prev_end = end
                i = end + 1
                continue
            kind, name = decl
            qual = f"{parent}.{name}" if parent else name
            bounds = self._block_bounds(i, b, d)
            flush_config()
            if bounds is None:
                end = self._stmt_end(i, b, d)
                seg = Seg("leaf", start, end, lead=self._lead(prev_end, start))
                seg.unit = self._leaf_unit(qual, "type" if kind == "container" else ("method" if in_class else "function"),
                                           start, end, i, None, parent, seg.lead)
                segs.append(seg)
                prev_end = end
                i = end + 1
                continue
            open_line, close = bounds
            if kind == "container" and close > open_line:
                seg = Seg("container", start, close, lead=self._lead(prev_end, start), header_end=open_line, name=qual)
                seg.compact_header = self._indent(i) + self._collapsed(i, open_line) + " {"
                children, last = self._parse(open_line + 1, close, d + 1, qual, True, open_line)
                seg.children = children
                _set_context(children, "\n".join(seg.lead[-6:] + self.lines[start : open_line + 1]))
                seg.footer = (close, close)
                seg.footer_lead = self.lines[last + 1 : close]
                segs.append(seg)
            else:
                seg = Seg("leaf", start, close, lead=self._lead(prev_end, start))
                k = "type" if kind == "container" else ("method" if in_class else "function")
                if self.lang == "go" and "." in name:
                    k = "method"
                seg.unit = self._leaf_unit(qual, k, start, close, i, open_line, parent, seg.lead)
                segs.append(seg)
            prev_end = close
            i = close + 1
        flush_config()
        return segs, prev_end

    # ------------------------------------------------------------- units
    def _text(self, a: int, b: int) -> str:
        return "\n".join(self.lines[a : b + 1])

    def _indent(self, i: int) -> str:
        ln = self.lines[i]
        return ln[: len(ln) - len(ln.lstrip())]

    def _is_public(self, sig: str, name: str, parent: str | None = None) -> bool:
        if re.search(r"\b(private|fileprivate)\b", sig) or name.startswith(("#", "_")):
            return False
        if self.lang == "rust":
            return bool(re.match(r"^\s*(#\[.*\]\s*)*pub\b", sig)) or "impl" in sig
        if self.lang == "go":
            return name.split(".")[-1][:1].isupper()
        if self.lang in ("typescript", "javascript") and parent is None:
            return bool(re.search(r"\bexport\b", sig))  # non-exported top-level symbols are module-private
        return True

    def _config_seg(self, ranges: list[tuple[int, int]], parent: str | None, lead_from: int) -> Seg:
        start, end = ranges[0][0], ranges[-1][1]
        l0 = self._text(start, end)
        qual = f"{parent}.<fields>" if parent else f"<module:{start + 1}>"
        vec_id = make_vec_id(self.path, qual, "config", l0)
        l1_parts, l2_parts = [], []
        for s, e in ranges:
            text = self._text(s, e)
            first = self._first_line_closed(s)
            if e - s + 1 <= 6:
                l1_parts.append(text)
            else:
                l1_parts.append(first + " " + _fold_marker(self.lang, "", vec_id, s + 1, e + 1, int(self.P[e + 1] - self.P[s])).strip())
            l2_parts.append(first if e > s else text)
        ind = self._indent(start)
        lead = self._lead(lead_from, start)
        role = "fields" if parent else "module declarations"
        unit = CodeUnit(vec_id, self.path, self.lang, qual, "config", start + 1, end + 1, ind,
                        "\n".join(lead) + ("\n" if lead else ""), l0, "\n".join(l1_parts), "\n".join(l2_parts),
                        f"{ind}{self.c} [L3 {qual}: {role} | {vec_id}]", "", "", role, parent, True,
                        _identifiers(l0))
        self.units.append(unit)
        return Seg("leaf", start, end, lead=lead, unit=unit)

    def _first_line_closed(self, i: int) -> str:
        """First line of a statement with any unclosed brackets closed (keeps brace balance)."""
        line = self.lines[i].rstrip()
        stack: list[str] = []
        pairs = {"{": "}", "(": ")", "[": "]"}
        for ch in self.infos[i].code:
            if ch in pairs:
                stack.append(pairs[ch])
            elif ch in ")]}" and stack and stack[-1] == ch:
                stack.pop()
        if not stack:
            return line
        return line + " /* … */ " + "".join(reversed(stack))

    def _leaf_unit(self, qual: str, kind: str, start: int, end: int, decl_line: int, open_line: int | None,
                   parent: str | None, lead: list[str]) -> CodeUnit:
        l0 = self._text(start, end)
        vec_id = make_vec_id(self.path, qual, kind, l0)
        ind = self._indent(decl_line)
        doc = _doc_from_comments(lead)
        if open_line is None:
            l1 = l0
            l2 = ind + " ".join(" ".join(self.infos[k].nc.split()) for k in range(decl_line, end + 1)).strip()
            if not brace_balanced(l2, self.lang):
                l2 = l0
            signature = " ".join(self._text(decl_line, end).split())
        else:
            signature = " ".join(self._text(decl_line, open_line).split())
            if getattr(self, "l1_mode", "signature") == "signature" and end > open_line:
                col = self._open_col(open_line)
                l1 = "\n".join(self.lines[start:open_line] + [self.lines[open_line][: col + 1].rstrip(),
                                 _fold_marker(self.lang, ind + "    ", vec_id, 0, 0, 0), ind + "}"])
                if not brace_balanced(l1, self.lang):
                    l1 = self._l1(start, open_line, end, vec_id)
            else:
                l1 = self._l1(start, open_line, end, vec_id)
            # L2: one-line, comment-free signature with an empty body (annotations dropped)
            l2 = ind + self._collapsed(decl_line, open_line) + " { … }"
            if not brace_balanced(l2, self.lang):
                l2 = "\n".join(self.lines[start:open_line]) + ("\n" if open_line > start else "") + \
                    self._first_line_closed(open_line)
        role = infer_role(qual, doc, l0, kind)
        short = qual.rsplit(".", 1)[-1]
        body_from = (open_line + 1) if open_line is not None else end + 1
        calls = list(dict.fromkeys(
            m for k in range(body_from, end + 1) for m in _CALL_RE.findall(self.infos[k].code)
            if m not in _CONTROL_WORDS and m != short))
        unit = CodeUnit(vec_id, self.path, self.lang, qual, kind, start + 1, end + 1, ind,
                        "\n".join(lead) + ("\n" if lead else ""), l0, l1, l2,
                        f"{ind}{self.c} [L3 {qual}: {role} | {vec_id}]", signature, doc, role, parent,
                        self._is_public(self._text(start, decl_line), qual.rsplit(".", 1)[-1], parent), _identifiers(l0))
        unit.calls = calls
        self.units.append(unit)
        return unit

    def _collapsed(self, a: int, open_line: int) -> str:
        """Comment-free text from line ``a`` up to (excluding) the first ``{`` on ``open_line``, on one line."""
        parts = [self.infos[k].nc for k in range(a, open_line)]
        inf = self.infos[open_line]
        parts.append(inf.nc[: inf.nc_open[0]] if inf.nc_open else inf.nc)
        return " ".join(" ".join(p.split()) for p in parts if p.strip())

    def _open_col(self, open_line: int) -> int:
        inf = self.infos[open_line]
        depth = inf.start_depth
        target = None
        for col, ev in inf.events:
            if ev == "{":
                if target is None:
                    target = col
                    break
            depth += 1 if ev == "{" else -1
        return target if target is not None else len(self.lines[open_line])

    def _valid_lines(self, emitted: list[int]) -> bool:
        """Balance proof for a render made of original lines + comment-only fold markers: every
        emitted line starts and ends in plain-code lexer state, braces never go negative and both
        braces and parens net to zero. Avoids re-lexing the rendered text."""
        depth = paren = 0
        last = len(self.infos) - 1
        for k in emitted:
            inf = self.infos[k]
            if not inf.starts_in_code or inf.unterminated or (k < last and not self.infos[k + 1].starts_in_code):
                return False
            for _, ev in inf.events:
                depth += 1 if ev == "{" else -1
                if depth < 0:
                    return False
            paren += inf.paren_delta
        return depth == 0 and paren == 0

    def _l1(self, start: int, open_line: int, close: int, vec_id: str) -> str:
        out = self.lines[start : open_line + 1]
        if close == open_line:
            return "\n".join(out)
        emitted = list(range(start, open_line + 1))
        marked = [False]
        base = self.infos[open_line].end_depth
        stack: list[bool] = []
        run: list[int] = []

        def flush() -> None:
            if run:
                a, b = run[0], run[-1]
                txt = "\n".join(self.lines[a : b + 1])
                if txt.strip():
                    n_run = int(self.P[b + 1] - self.P[a])
                    marker = _fold_marker(self.lang, self._indent(a), vec_id, a + 1, b + 1, n_run, first=not marked[0])
                    if n_run <= count_tokens(marker):
                        out.extend(self.lines[a : b + 1])
                        emitted.extend(range(a, b + 1))
                    else:
                        out.append(marker)
                        marked[0] = True
                run.clear()

        for i in range(open_line + 1, close):
            inf = self.infos[i]
            code = inf.code.strip()
            if not code:
                if run:
                    run.append(i)
                continue
            rel = inf.start_depth - base
            first_close = bool(inf.events) and inf.events[0][1] == "}" and code.startswith("}")
            all_kept = all(stack)
            if first_close:
                keep = stack[-1] if stack else True
                keep = keep and all(stack[:-1])
            else:
                keep = all_kept and rel < self.max_depth and (_CTRL_RE.match(code) is not None or _RET_RE.match(code) is not None)
            # update stack
            for _, ev in inf.events:
                if ev == "{":
                    stack.append(keep)
                elif stack:
                    stack.pop()
            if keep:
                flush()
                out.append(self.lines[i])
                emitted.append(i)
            else:
                run.append(i)
        flush()
        out.append(self.lines[close])
        emitted.append(close)
        text = "\n".join(out)
        if not self._valid_lines(emitted):
            body = "\n".join(self.lines[open_line + 1 : close])
            text = "\n".join(self.lines[start : open_line + 1] + [
                _fold_marker(self.lang, self._indent(open_line) + "    ", vec_id, open_line + 2, close, count_tokens(body)),
                self.lines[close]])
        return text


def _set_context(segs: list[Seg], ctx: str) -> None:
    for c in segs:
        if c.kind == "leaf" and c.unit is not None:
            c.unit.context = (ctx + "\n" + c.unit.context).strip() if c.unit.context else ctx
        elif c.kind == "container":
            _set_context(c.children, ctx)


def _doc_from_comments(lead: list[str]) -> str:
    out = []
    for ln in lead:
        s = ln.strip()
        if s.startswith(("///", "//!", "/**", "*", "//")) and not s.startswith("*/"):
            s = re.sub(r"^(///?!?|/\*\*|\*)\s?", "", s).strip()
            if s and not s.startswith("@"):
                out.append(s)
    return " ".join(out)


# ============================================================================= plain text / fallback


def _text_mipmap(path: str, source: str, lang: str) -> tuple[list[Seg], list[CodeUnit]]:
    lines = source.split("\n")
    end = len(lines) - 1
    while end > 0 and not lines[end].strip():
        end -= 1
    l0 = "\n".join(lines[: end + 1])
    vec_id = make_vec_id(path, "<text>", "config", l0)
    c = comment_prefix(lang)
    head = "\n".join(lines[: min(8, end + 1)])
    l1 = head + (f"\n{c} [~aegis:{vec_id}~]" if end >= 8 else "")
    l2 = lines[0] if lines else ""
    unit = CodeUnit(vec_id, path, lang, "<text>", "config", 1, end + 1, "", "", l0, l1, l2,
                    f"{c} [L3 {path}: text | {vec_id}]", "", "", "text", None, True, _identifiers(l0))
    return [Seg("leaf", 0, end, unit=unit)], [unit]


# ============================================================================= public API


def _summary(units: list[CodeUnit], path: str) -> str:
    """``module: top-3 public symbols`` for the single-line L3 descriptor."""
    module = PurePosixPath(path).stem
    names: list[str] = []
    for u in units:
        if u.kind == "config" or not u.is_public:
            continue
        top = u.qualname.split(".")[0]  # class name for methods, function name otherwise
        if top not in names:
            names.append(top)
    if not names:
        names = list(dict.fromkeys(u.qualname.split(".")[0] for u in units if u.kind != "config"))
    more = f" +{len(names) - 3}" if len(names) > 3 else ""
    return f"{module}: {', '.join(names[:3])}{more}" if names else module


_IMPORT_TARGET = re.compile(r"""["']([^"']+)["']""")


def _deps_line(root: list[Seg], lines: list[str], lang: str) -> str:
    """Collapse every import statement into one ``deps: a, b, c`` comment (the module dependency graph)."""
    mods: list[str] = []
    for seg in root:
        if seg.kind != "import":
            continue
        text = " ".join(ln.strip() for ln in lines[seg.start : seg.end + 1])
        if text.startswith("package "):
            continue
        if lang == "python":
            m = re.match(r"from\s+(\S+)\s+import", text)
            found = [m.group(1)] if m else [x.strip().split(" as ")[0] for x in text[len("import "):].split(",")]
        else:
            q = _IMPORT_TARGET.search(text)
            if q:
                found = [q.group(1)]
            else:
                t = re.sub(r"^(import|use|#include|#import|using|extern\s+crate)\s+(static\s+)?", "", text).rstrip(";").strip()
                found = [re.sub(r"::\{.*\}$|\.\*$", "", t).strip("<> ")]
        for f in found:
            if f and f not in mods and f != "__future__":
                mods.append(f)
    if not mods:
        return ""
    more = f" +{len(mods) - 8}" if len(mods) > 8 else ""
    return f"{comment_prefix(lang)} deps: {', '.join(mods[:8])}{more}"


def _py_signature_line(fn, indent: str, seg) -> str:
    """One-line Python stub built by slicing the source spans of the argument/return nodes (no
    ``ast.unparse``/``deepcopy``): ``self``/``cls`` dropped, defaults longer than 12 chars shown as
    ``...``, semantic decorators only."""
    keep = ("property", "staticmethod", "classmethod", "abstractmethod", "overload")
    flat = lambda n: " ".join(seg(n).split())  # noqa: E731
    decos = []
    for d in fn.decorator_list:
        txt = flat(d)
        if txt.split(".")[-1] in keep or txt.endswith((".setter", ".getter", ".deleter")):
            decos.append(f"{indent}@{txt}")
    a = fn.args
    positional = list(a.posonlyargs) + list(a.args)
    defaults = [None] * (len(positional) - len(a.defaults)) + list(a.defaults)
    parts: list[str] = []
    for i, (arg, dflt) in enumerate(zip(positional, defaults)):
        if i == 0 and arg.arg in ("self", "cls"):
            continue
        txt = flat(arg)
        if dflt is not None:
            d = flat(dflt)
            txt += "=" + (d if len(d) <= 12 else "...")
        parts.append(txt)
        if a.posonlyargs and arg is a.posonlyargs[-1]:
            parts.append("/")
    if a.vararg is not None:
        parts.append("*" + flat(a.vararg))
    elif a.kwonlyargs:
        parts.append("*")
    for arg, dflt in zip(a.kwonlyargs, a.kw_defaults):
        txt = flat(arg)
        if dflt is not None:
            d = flat(dflt)
            txt += "=" + (d if len(d) <= 12 else "...")
        parts.append(txt)
    if a.kwarg is not None:
        parts.append("**" + flat(a.kwarg))
    ret = f" -> {flat(fn.returns)}" if fn.returns is not None else ""
    head = "async def" if isinstance(fn, ast.AsyncFunctionDef) else "def"
    return "\n".join(decos + [f"{indent}{head} {fn.name}({', '.join(parts)}){ret}: ..."])


def mipmap_file(path: str, source: str, language: str | None = None, max_depth: int = 1,
                l1_mode: str = "signature", exact_tokens: bool = True) -> FileMipmap:
    """Parse ``source`` and pre-compute all four resolution levels.

    ``exact_tokens=False`` (proxy first-sight path) sizes units/levels with the O(1) estimator
    instead of cl100k BPE; reported token metrics are always computed exactly on rendered text."""
    lang = language or detect_language(path=path, text=source)
    source = source.replace("\r\n", "\n")
    units: list[CodeUnit]
    trailing: list[str]
    lines = source.split("\n")
    P = line_prefix(source) if exact_tokens else line_prefix_estimate(source)
    tok = count_tokens if exact_tokens else estimate_tokens
    try:
        if lang == "python":
            mm = _PythonMipmapper(path, source, max_depth, P)
            mm.l1_mode = l1_mode
            root, last = mm.run()
            units = mm.units
            trailing = lines[last:]
        elif lang in BRACE_LANGS:
            bm = _BraceMipmapper(path, source, lang, max_depth, P)
            bm.l1_mode = l1_mode
            root, last = bm.run()
            units = bm.units
            trailing = lines[last:]
        else:
            root, units = _text_mipmap(path, source, lang)
            trailing = []
    except (SyntaxError, ValueError, RecursionError):
        root, units = _text_mipmap(path, source, lang)
        trailing = []
    if not units and source.strip():
        root, units = _text_mipmap(path, source, lang)
    fm = FileMipmap(path=path, language=lang, source=source, lines=lines, root=root, trailing=trailing,
                    units=units, file_vec_id=make_vec_id(path, "<file>", source), summary=_summary(units, path),
                    deps_line=_deps_line(root, lines, lang))
    fm.tokens_source = int(P[-1])
    for u in units:
        t0 = int(P[u.end_line] - P[u.start_line - 1]) if u.l0 is not None and u.end_line <= len(lines) else count_tokens(u.l0)
        t1 = t0 if u.l1 == u.l0 else tok(u.l1)
        if t1 >= t0:
            u.l1, t1 = u.l0, t0
        t2 = tok(u.l2)
        if t2 > t1:
            u.l2, t2 = u.l1, t1
        u.tokens = {0: t0, 1: t1, 2: t2, 3: tok(u.l3)}
    fm.skeleton = {2: fm.render({}, 2), 3: fm.l3_line() + "\n"}
    fm.skeleton_tokens = {k: tok(v) for k, v in fm.skeleton.items()}
    fm.tokens_estimated = not exact_tokens
    return fm


def syntax_ok(text: str, language: str) -> bool:
    """Parseability check used by tests/benchmarks: ``ast.parse`` for Python, brace balance otherwise."""
    if language == "python":
        try:
            ast.parse(text)
            return True
        except SyntaxError:
            return False
    if language in BRACE_LANGS:
        return brace_balanced(text, language)
    return True


def iter_leaf_units(files: Iterable[FileMipmap]) -> Iterable[CodeUnit]:
    for f in files:
        yield from f.units
