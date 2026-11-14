"""Locate code inside Anthropic / OpenAI chat payloads and rewrite it in place.

Recognised code carriers:
* fenced blocks  ```lang [path]  …  ```   (also ```lang:path and ~~~ fences),
* ``<file path="…">…</file>`` and ``<document><source>…</source><document_content>…`` tags,
* tool results produced by file-reading tools (``cat -n`` / ``N→`` numbered listings or raw text),
  whose path is recovered from the matching ``tool_use`` / ``tool_calls`` input.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Iterator

from .ast_mipmapper import detect_language

FENCE_RE = re.compile(r"(?P<fence>`{3,}|~{3,})(?P<info>[^\n`]*)\n(?P<code>.*?)\n[ \t]*(?P=fence)[ \t]*(?=\n|$)", re.S)
FILE_TAG_RE = re.compile(r"<file\s+(?:path|name)=\"(?P<path>[^\"]+)\"[^>]*>\n?(?P<code>.*?)\n?</file>", re.S)
DOC_TAG_RE = re.compile(r"<source>(?P<path>[^<]+)</source>\s*<document_content>\n?(?P<code>.*?)\n?</document_content>", re.S)
CATN_LINE = re.compile(r"^\s*(\d+)(?:\t|→)(.*)$")
PATHLIKE = re.compile(r"[\w./-]+\.[A-Za-z0-9]{1,6}$")
PATH_KEYS = ("file_path", "path", "filename", "file", "target_file", "notebook_path")


@dataclass
class Slot:
    path: tuple  # key/index path inside the message dict
    text: str
    tool_path: str | None = None


@dataclass
class FoundCode:
    start: int
    end: int
    path: str
    language: str
    code: str
    style: str  # fence | file_tag | doc_tag | tool
    fence: str = ""
    info: str = ""


# ----------------------------------------------------------------------------- helpers

def strip_cache_control(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: strip_cache_control(v) for k, v in obj.items() if k != "cache_control"}
    if isinstance(obj, list):
        return [strip_cache_control(v) for v in obj]
    return obj


def canonical_hash(msg: dict) -> str:
    data = json.dumps(strip_cache_control(msg), sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.blake2b(data.encode("utf-8"), digest_size=12).hexdigest()


def get_in(obj: Any, path: tuple) -> Any:
    for p in path:
        obj = obj[p]
    return obj


def set_in(obj: Any, path: tuple, value: Any) -> None:
    for p in path[:-1]:
        obj = obj[p]
    obj[path[-1]] = value


def tool_paths(messages: list[dict], fmt: str) -> dict[str, str]:
    """Map tool-call ids to the file path they read (from tool_use / tool_calls inputs)."""
    out: dict[str, str] = {}
    for m in messages:
        if m.get("role") != "assistant":
            continue
        if fmt == "anthropic" and isinstance(m.get("content"), list):
            for b in m["content"]:
                if isinstance(b, dict) and b.get("type") == "tool_use":
                    inp = b.get("input") or {}
                    for k in PATH_KEYS:
                        if isinstance(inp.get(k), str):
                            out[b.get("id", "")] = inp[k]
                            break
        elif fmt == "openai":
            for tc in m.get("tool_calls") or []:
                fn = tc.get("function") or {}
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                except (ValueError, TypeError):
                    args = {}
                if isinstance(args, dict):
                    for k in PATH_KEYS:
                        if isinstance(args.get(k), str):
                            out[tc.get("id", "")] = args[k]
                            break
    return out


def iter_slots(msg: dict, fmt: str, tpaths: dict[str, str]) -> Iterator[Slot]:
    content = msg.get("content")
    tool_path = None
    if fmt == "openai" and msg.get("role") == "tool":
        tool_path = tpaths.get(msg.get("tool_call_id", ""))
    if isinstance(content, str):
        yield Slot(("content",), content, tool_path)
        return
    if not isinstance(content, list):
        return
    for i, b in enumerate(content):
        if not isinstance(b, dict):
            continue
        t = b.get("type")
        if t in ("text", "input_text") and isinstance(b.get("text"), str):
            yield Slot(("content", i, "text"), b["text"], tool_path)
        elif t == "tool_result":
            tp = tpaths.get(b.get("tool_use_id", ""))
            c = b.get("content")
            if isinstance(c, str):
                yield Slot(("content", i, "content"), c, tp)
            elif isinstance(c, list):
                for j, bb in enumerate(c):
                    if isinstance(bb, dict) and bb.get("type") == "text" and isinstance(bb.get("text"), str):
                        yield Slot(("content", i, "content", j, "text"), bb["text"], tp)


def _parse_info(info: str) -> tuple[str, str]:
    info = info.strip()
    if not info:
        return "", ""
    if ":" in info.split()[0] and not info.startswith("http"):
        lang, _, path = info.split()[0].partition(":")
        return lang, path
    parts = info.split()
    lang, path = "", ""
    for p in parts:
        p = p.strip("{}").replace("title=", "").replace("file=", "").strip("\"'")
        if PATHLIKE.search(p) or "/" in p:
            path = p
        elif not lang:
            lang = p
    return lang, path


_CATN_STRIP = re.compile(r"(?m)^[ \t]*\d+(?:\t|→)")


def _strip_catn(text: str) -> str | None:
    """Strip ``cat -n`` / ``N→`` numbering when (nearly) every non-empty line carries it."""
    nonempty = sum(1 for ln in text.split("\n") if ln.strip())
    if nonempty < 3:
        return None
    stripped, n = _CATN_STRIP.subn("", text)
    if n < 0.8 * nonempty:
        return None
    return stripped.rstrip("\n").rstrip()


def find_code(slot: Slot, min_chars: int = 200) -> list[FoundCode]:
    text = slot.text
    found: list[FoundCode] = []
    for m in FENCE_RE.finditer(text):
        code = m.group("code")
        if len(code) < min_chars:
            continue
        lang_tag, path = _parse_info(m.group("info"))
        lang = detect_language(path=path or None, fence=lang_tag or None, text=code)
        if lang == "text":
            continue
        found.append(FoundCode(m.start(), m.end(), path or f"snippet_{hashlib.blake2b(code.encode(), digest_size=3).hexdigest()}",
                               lang, code, "fence", m.group("fence"), m.group("info")))
    for rx, style in ((FILE_TAG_RE, "file_tag"), (DOC_TAG_RE, "doc_tag")):
        for m in rx.finditer(text):
            if any(f.start <= m.start() < f.end for f in found):
                continue
            code = m.group("code")
            path = m.group("path").strip()
            lang = detect_language(path=path, text=code)
            if len(code) < min_chars or lang == "text":
                continue
            found.append(FoundCode(m.start("code"), m.end("code"), path, lang, code, style))
    if not found and slot.tool_path:
        code = _strip_catn(text)
        style = "tool"
        if code is None:
            code = text
        lang = detect_language(path=slot.tool_path, text=code)
        if lang != "text" and len(code) >= min_chars:
            found.append(FoundCode(0, len(text), slot.tool_path, lang, code, style))
    found.sort(key=lambda f: f.start)
    return found


def replace_found(text: str, found: FoundCode, rendered: str, note: str) -> str:
    body = rendered.rstrip("\n")
    if found.style == "fence":
        new = f"{found.fence}{found.info}\n{body}\n{found.fence}"
    elif found.style == "tool":
        new = f"{note}\n{body}"
    else:
        new = body
    return text[: found.start] + new + text[found.end :]


def natural_text(msg: dict, fmt: str) -> str:
    """User-authored text of a message with code blocks and tool results removed."""
    content = msg.get("content")
    parts: list[str] = []
    if isinstance(content, str):
        parts.append(content)
    elif isinstance(content, list):
        for b in content:
            if isinstance(b, dict) and b.get("type") in ("text", "input_text") and isinstance(b.get("text"), str):
                parts.append(b["text"])
    text = "\n".join(parts)
    text = FENCE_RE.sub(" ", text)
    text = FILE_TAG_RE.sub(" ", text)
    text = re.sub(r"<system-reminder>.*?</system-reminder>", " ", text, flags=re.S)
    text = re.sub(r"<aegis_[a-z_]+[^>]*>.*?</aegis_[a-z_]+>", " ", text, flags=re.S)
    return text.strip()


def append_text(msg: dict, fmt: str, text: str) -> dict:
    msg = copy.deepcopy(msg)
    c = msg.get("content")
    if isinstance(c, str) or c is None:
        if fmt == "anthropic":
            msg["content"] = ([{"type": "text", "text": c}] if c else []) + [{"type": "text", "text": text}]
        else:
            msg["content"] = (c + "\n\n" if c else "") + text
    elif isinstance(c, list):
        c.append({"type": "text", "text": text})
    return msg


def system_text(body: dict, fmt: str) -> str:
    if fmt == "anthropic":
        s = body.get("system")
        if isinstance(s, str):
            return s
        if isinstance(s, list):
            return "\n".join(b.get("text", "") for b in s if isinstance(b, dict))
        return ""
    return "\n".join(m.get("content", "") if isinstance(m.get("content"), str) else ""
                     for m in body.get("messages", []) if m.get("role") in ("system", "developer"))


def message_plain(msg: dict) -> str:
    c = msg.get("content")
    if isinstance(c, str):
        return c
    out: list[str] = []
    if isinstance(c, list):
        for b in c:
            if not isinstance(b, dict):
                continue
            if b.get("type") in ("text", "input_text"):
                out.append(b.get("text", ""))
            elif b.get("type") == "tool_result":
                cc = b.get("content")
                if isinstance(cc, str):
                    out.append(cc)
                elif isinstance(cc, list):
                    out.extend(x.get("text", "") for x in cc if isinstance(x, dict))
            elif b.get("type") == "tool_use":
                out.append(json.dumps(b.get("input", {}), sort_keys=True))
    for tc in msg.get("tool_calls") or []:
        out.append(json.dumps(tc.get("function", {}), sort_keys=True))
    return "\n".join(out)


def prompt_text(body: dict, fmt: str) -> str:
    """Linearised prompt (system -> messages) used for prefix-cache accounting."""
    parts = [system_text(body, fmt)] if fmt == "anthropic" else []
    for m in body.get("messages", []):
        parts.append(f"\n\n{m.get('role', '')}: {message_plain(m)}")
    return "".join(parts)
