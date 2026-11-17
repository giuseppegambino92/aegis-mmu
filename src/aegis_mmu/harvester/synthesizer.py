"""Compile a minimal execution chain into a deterministic tool.

Output of :func:`synthesize`:

* :class:`ToolSpec` – declarative steps (method, URL, query/headers/body templates, extractions),
  typed inputs (secrets become required, env-backed inputs and are never persisted),
* :class:`~aegis_mmu.harvester.self_healer.ContractShadow` of the target response,
* a stand-alone async Python client (httpx + pydantic) with generated Pydantic input/output models.

:class:`ToolRuntime` executes a spec (zero LLM tokens), validates the final payload against the
shadow contract and transparently heals schema drift.
"""

from __future__ import annotations

import json
import keyword
import re
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field

from .self_healer import ContractShadow, SchemaDriftReport, SchemaHealer, build_shadow
from .taint_graph import MinimalChain, TaintGraph

SECRET_RE = re.compile(r"(pass(word|wd)?|secret|api[_-]?key|otp|pin|private[_-]?key|client[_-]?secret)$", re.I)
KEEP_HEADERS = {"accept", "content-type", "authorization", "origin"}
DROP_HEADERS = {"cookie", "user-agent", "accept-language", "accept-encoding", "content-length", "host", "referer",
                "connection", "pragma", "cache-control", "te", "priority", "dnt"}
PLACEHOLDER = re.compile(r"\{\{(input|var)\.([A-Za-z_]\w*)\}\}")


class InputSpec(BaseModel):
    name: str
    type: str = "str"
    default: Any = None
    required: bool = False
    secret: bool = False
    env: str | None = None
    location: str = ""


class ExtractSpec(BaseModel):
    var: str
    source: str  # body | header | cookie
    path: str


class StepSpec(BaseModel):
    har_index: int
    method: str
    url: str
    query: list[list[str]] = Field(default_factory=list)
    headers: dict[str, str] = Field(default_factory=dict)
    body_kind: str | None = None  # json | form | raw
    body: Any = None
    extract: list[ExtractSpec] = Field(default_factory=list)


class ToolSpec(BaseModel):
    name: str
    description: str
    inputs: list[InputSpec]
    steps: list[StepSpec]
    target: str
    target_path: str
    origins: list[str]


@dataclass
class HarvestedTool:
    spec: ToolSpec
    shadow: ContractShadow
    code: str
    chain: MinimalChain

    def input_schema(self) -> dict:
        return input_json_schema(self.spec)


# ----------------------------------------------------------------------------- helpers


def _ident(name: str) -> str:
    s = re.sub(r"\W+", "_", name).strip("_").lower() or "value"
    if s[0].isdigit():
        s = "v_" + s
    if keyword.iskeyword(s):
        s += "_"
    return s


def _ptype(v: Any) -> str:
    if isinstance(v, bool):
        return "bool"
    if isinstance(v, int):
        return "int"
    if isinstance(v, float):
        return "float"
    if isinstance(v, str) and re.fullmatch(r"-?\d{1,9}", v):
        return "int"
    return "str"


def _coerce(v: Any, t: str) -> Any:
    if v is None:
        return None
    try:
        return {"int": int, "float": float, "bool": lambda x: str(x).lower() in ("1", "true", "yes"), "str": str}[t](v)
    except (ValueError, TypeError):
        return v


def input_json_schema(spec: ToolSpec) -> dict:
    jt = {"str": "string", "int": "integer", "float": "number", "bool": "boolean"}
    props, req = {}, []
    for i in spec.inputs:
        p: dict[str, Any] = {"type": jt.get(i.type, "string"), "description": f"{i.location}" + (f" (secret; falls back to ${i.env})" if i.secret else "")}
        if i.default is not None and not i.secret:
            p["default"] = i.default
        props[i.name] = p
        if i.required and not i.secret:
            req.append(i.name)
    return {"type": "object", "properties": props, "required": req, "additionalProperties": False}


# ----------------------------------------------------------------------------- synthesis


def synthesize(chain: MinimalChain, tool_name: str, engine, description: str | None = None) -> HarvestedTool:
    tool_name = _ident(tool_name)
    graph = TaintGraph(chain.txns)
    steps_t = chain.steps()
    chain_set = set(chain.chain)
    # variables: values produced inside the chain and consumed by a later chain step
    consumed: dict[str, tuple[int, str, str]] = {}  # value -> (producer idx, location, var name)
    var_names: set[str] = set()
    for e in chain.edges:
        if e.src in chain_set and e.dst in chain_set and e.value not in consumed:
            src_loc = e.src_loc
            base = _ident(src_loc.split(":", 1)[1].rsplit(".", 1)[-1].split("[")[0] if src_loc.startswith("body:") else src_loc.split(":", 1)[1])
            name = base
            k = 2
            while name in var_names:
                name = f"{base}_{k}"
                k += 1
            var_names.add(name)
            consumed[e.value] = (e.src, src_loc, name)
    values_sorted = sorted(consumed.items(), key=lambda kv: -len(kv[0]))

    def templ(s: str) -> tuple[str, bool]:
        changed = False
        for val, (_src, _loc, name) in values_sorted:
            if val in s:
                s = s.replace(val, "{{var.%s}}" % name)
                changed = True
        return s, changed

    inputs: dict[str, InputSpec] = {}

    def add_input(raw_name: str, value: Any, location: str, secret: bool = False) -> str:
        name = _ident(raw_name)
        if name in inputs and inputs[name].location != location:
            name = f"{name}_{len(inputs)}"
        t = _ptype(value)
        inputs[name] = InputSpec(name=name, type=t, default=None if secret else _coerce(value, t), required=secret,
                                 secret=secret, env=f"{tool_name}_{name}".upper() if secret else None, location=location)
        return "{{input.%s}}" % name

    steps: list[StepSpec] = []
    for t in steps_t:
        is_target = t.index == chain.target_index
        url, _ = templ(t.origin + t.path)
        query = []
        for k, v in t.query:
            tv, ch = templ(v)
            if not ch:
                tv = add_input(k, v, f"step{t.index}.query.{k}") if is_target else v
            query.append([k, tv])
        headers: dict[str, str] = {}
        for k, v in t.req_headers.items():
            if k in DROP_HEADERS or k.startswith(("sec-", ":")):
                continue
            tv, ch = templ(v)
            if ch or k in KEEP_HEADERS or k.startswith("x-"):
                headers[k] = tv
        cookie_parts = []
        for k, v in t.req_cookies.items():
            tv, ch = templ(v)
            if ch:
                cookie_parts.append(f"{k}={tv}")
        if cookie_parts:
            headers["cookie"] = "; ".join(cookie_parts)
        body_kind, body = None, None
        if t.req_json is not None:
            body_kind = "json"
            body = _templ_json(t.req_json, templ, lambda key, val, loc: add_input(key, val, loc, bool(SECRET_RE.search(key))),
                               f"step{t.index}.body")
        elif t.req_body:
            if "x-www-form-urlencoded" in (t.req_mime or ""):
                from urllib.parse import parse_qsl

                body_kind = "form"
                body = {}
                for k, v in parse_qsl(t.req_body, keep_blank_values=True):
                    tv, ch = templ(v)
                    body[k] = tv if ch else add_input(k, v, f"step{t.index}.form.{k}", bool(SECRET_RE.search(k)))
            else:
                body_kind = "raw"
                body, _ = templ(t.req_body)
        extract = [ExtractSpec(var=name, source=loc.split(":", 1)[0], path=loc.split(":", 1)[1])
                   for val, (src, loc, name) in consumed.items() if src == t.index]
        steps.append(StepSpec(har_index=t.index, method=t.method, url=url, query=query, headers=headers,
                              body_kind=body_kind, body=body, extract=extract))
    target_txn = chain.txns[chain.target_index]
    desc = description or (f"Fetch `{chain.target}` from {target_txn.origin}{target_txn.path} via a {len(steps)}-step "
                           f"replayed chain (harvested by aegis-mmu from a {len(chain.txns)}-request capture).")
    spec = ToolSpec(name=tool_name, description=desc, inputs=list(inputs.values()), steps=steps, target=chain.target,
                    target_path=chain.target_path, origins=sorted({t.origin for t in steps_t}))
    shadow = build_shadow(tool_name, target_txn.resp_json, engine)
    code = generate_client_code(spec, target_txn.resp_json, shadow)
    del graph
    return HarvestedTool(spec, shadow, code, chain)


def _templ_json(obj: Any, templ, add_input, loc: str) -> Any:
    if isinstance(obj, dict):
        return {k: _templ_json_leaf(v, templ, add_input, f"{loc}.{k}", k) if not isinstance(v, (dict, list))
                else _templ_json(v, templ, add_input, f"{loc}.{k}") for k, v in obj.items()}
    if isinstance(obj, list):
        return [_templ_json(v, templ, add_input, f"{loc}[{i}]") if isinstance(v, (dict, list))
                else _templ_json_leaf(v, templ, add_input, f"{loc}[{i}]", "item") for i, v in enumerate(obj)]
    return obj


def _templ_json_leaf(v: Any, templ, add_input, loc: str, key: str) -> Any:
    sv = v if isinstance(v, str) else json.dumps(v)
    tv, ch = templ(sv)
    if ch:
        return tv
    return add_input(key, v, loc)


# ----------------------------------------------------------------------------- runtime


def render_template(obj: Any, inputs: dict[str, Any], vars_: dict[str, Any]) -> Any:
    if isinstance(obj, str):
        m = PLACEHOLDER.fullmatch(obj)
        if m:
            src = inputs if m.group(1) == "input" else vars_
            return src.get(m.group(2))
        return PLACEHOLDER.sub(lambda mm: str((inputs if mm.group(1) == "input" else vars_).get(mm.group(2), "")), obj)
    if isinstance(obj, dict):
        return {k: render_template(v, inputs, vars_) for k, v in obj.items()}
    if isinstance(obj, list):
        return [render_template(v, inputs, vars_) for v in obj]
    return obj


def resolve_path(obj: Any, path: str) -> Any:
    for part in re.finditer(r"\.([^.\[\]]+)|\[(\d+)\]", path):
        if part.group(1) is not None:
            if not isinstance(obj, dict):
                return None
            obj = obj.get(part.group(1).strip('"'))
        else:
            i = int(part.group(2))
            if not isinstance(obj, list) or i >= len(obj):
                return None
            obj = obj[i]
    return obj


class ToolExecutionError(RuntimeError):
    pass


async def execute_spec(spec: ToolSpec, inputs: dict[str, Any], *, client=None,
                       origin_overrides: dict[str, str] | None = None, timeout: float = 30.0) -> tuple[Any, list[dict]]:
    """Run every step of ``spec``; returns (final JSON payload, step log)."""
    import os

    import httpx

    resolved: dict[str, Any] = {}
    for i in spec.inputs:
        v = inputs.get(i.name)
        if v is None and i.secret and i.env:
            v = os.environ.get(i.env)
        if v is None:
            v = i.default
        if v is None and (i.required or i.secret):
            raise ToolExecutionError(f"missing required input {i.name!r}" + (f" (or env ${i.env})" if i.env else ""))
        resolved[i.name] = _coerce(v, i.type)
    own = client is None
    client = client or httpx.AsyncClient(timeout=timeout, follow_redirects=True)
    vars_: dict[str, Any] = {}
    log: list[dict] = []
    payload: Any = None
    try:
        for st in spec.steps:
            url = render_template(st.url, resolved, vars_)
            for src, dst in (origin_overrides or {}).items():
                if url.startswith(src):
                    url = dst + url[len(src):]
            params = [(k, str(render_template(v, resolved, vars_))) for k, v in st.query]
            headers = {k: str(render_template(v, resolved, vars_)) for k, v in st.headers.items()}
            kw: dict[str, Any] = {}
            if st.body_kind == "json":
                kw["json"] = render_template(st.body, resolved, vars_)
            elif st.body_kind == "form":
                kw["data"] = render_template(st.body, resolved, vars_)
            elif st.body_kind == "raw":
                kw["content"] = str(render_template(st.body, resolved, vars_))
            r = await client.request(st.method, url, params=params, headers=headers, **kw)
            log.append({"step": st.har_index, "method": st.method, "url": str(r.request.url).split("?")[0], "status": r.status_code})
            if r.status_code >= 400:
                raise ToolExecutionError(f"step {st.har_index} {st.method} {url} -> HTTP {r.status_code}: {r.text[:200]}")
            data = None
            try:
                data = r.json()
            except ValueError:
                data = None
            for ex in st.extract:
                if ex.source == "body":
                    vars_[ex.var] = resolve_path(data, ex.path)
                elif ex.source == "header":
                    vars_[ex.var] = r.headers.get(ex.path)
                elif ex.source == "cookie":
                    vars_[ex.var] = r.cookies.get(ex.path) or client.cookies.get(ex.path)
                if vars_.get(ex.var) is None:
                    raise ToolExecutionError(f"step {st.har_index}: could not extract {ex.var} from {ex.source}:{ex.path}")
            payload = data
    finally:
        if own:
            await client.aclose()
    return payload, log


class ToolRuntime:
    """Executes registered tools from the store and heals drifted responses on the fly."""

    def __init__(self, store, healer: SchemaHealer | None = None):
        self.store = store
        self.healer = healer or SchemaHealer(store.engine, store.settings.heal_min_confidence)

    def register(self, tool: HarvestedTool) -> None:
        self.store.save_contract(tool.spec.name, tool.spec.model_dump(), tool.shadow.to_dict(), tool.code)

    def heal(self, tool_name: str, payload: Any) -> tuple[Any, SchemaDriftReport]:
        c = self.store.get_contract(tool_name)
        if c is None:
            raise KeyError(f"unknown tool {tool_name!r}")
        shadow = ContractShadow.from_dict(c["shadow"])
        healed, report = self.healer.heal(payload, shadow)
        if report.drift_detected:
            self.store.log_drift(tool_name, report.to_dict())
            if report.healed:
                self.store.update_shadow(tool_name, shadow.to_dict())
        return healed, report

    async def call(self, tool_name: str, inputs: dict[str, Any], *, client=None,
                   origin_overrides: dict[str, str] | None = None) -> dict:
        c = self.store.get_contract(tool_name)
        if c is None:
            raise KeyError(f"unknown tool {tool_name!r}")
        spec = ToolSpec(**c["spec"])
        payload, log = await execute_spec(spec, inputs, client=client, origin_overrides=origin_overrides)
        healed, report = self.heal(tool_name, payload)
        return {"data": healed, "steps": log, "drift": report.to_dict() if report.drift_detected else None}


# ----------------------------------------------------------------------------- code generation


def _model_sources(name: str, sample: Any, out: list[str], seen: set[str]) -> str:
    """Emit Pydantic classes for ``sample``; returns the type annotation for it."""
    if isinstance(sample, dict):
        cls = name
        while cls in seen:
            cls += "_"
        seen.add(cls)
        lines = [f"class {cls}(BaseModel):", "    model_config = ConfigDict(extra=\"allow\", populate_by_name=True)"]
        for k, v in sample.items():
            fname = _ident(k)
            ann = _model_sources(cls + "".join(p.capitalize() for p in _ident(k).split("_")), v, out, seen)
            if v is None:
                lines.append(f"    {fname}: Any = Field(default=None, alias={k!r})" if fname != k else f"    {fname}: Any = None")
            else:  # observed fields are required: a drifted payload must fail validation, not pass as nulls
                lines.append(f"    {fname}: {ann} = Field(alias={k!r})" if fname != k else f"    {fname}: {ann}")
        out.append("\n".join(lines))
        return cls
    if isinstance(sample, list):
        if sample and isinstance(sample[0], dict):
            merged: dict = {}
            for it in sample:
                if isinstance(it, dict):
                    for k, v in it.items():
                        merged.setdefault(k, v)
            return f"list[{_model_sources(name + 'Item', merged, out, seen)}]"
        inner = _model_sources(name + "Item", sample[0], out, seen) if sample else "Any"
        return f"list[{inner}]"
    if isinstance(sample, bool):
        return "bool"
    if isinstance(sample, int):
        return "int"
    if isinstance(sample, float):
        return "float"
    if isinstance(sample, str):
        return "str"
    return "Any"


_RUNTIME_TEMPLATE = '''

PLACEHOLDER = re.compile(r"\\{\\{(input|var)\\.([A-Za-z_]\\w*)\\}\\}")


def _render(obj: Any, inputs: dict, vars_: dict) -> Any:
    if isinstance(obj, str):
        m = PLACEHOLDER.fullmatch(obj)
        if m:
            return (inputs if m.group(1) == "input" else vars_).get(m.group(2))
        return PLACEHOLDER.sub(lambda mm: str((inputs if mm.group(1) == "input" else vars_).get(mm.group(2), "")), obj)
    if isinstance(obj, dict):
        return {k: _render(v, inputs, vars_) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_render(v, inputs, vars_) for v in obj]
    return obj


def _resolve(obj: Any, path: str) -> Any:
    for part in re.finditer(r"\\.([^.\\[\\]]+)|\\[(\\d+)\\]", path):
        if part.group(1) is not None:
            obj = obj.get(part.group(1)) if isinstance(obj, dict) else None
        else:
            i = int(part.group(2))
            obj = obj[i] if isinstance(obj, list) and i < len(obj) else None
    return obj


async def run(inputs: {INPUT} | dict | None = None, *, client: httpx.AsyncClient | None = None,
              origin_overrides: dict[str, str] | None = None) -> {OUTPUT}:
    """Execute the harvested chain and return the validated (and, if aegis-mmu is installed, self-healed) output."""
    if inputs is None:
        inputs = {INPUT}()
    elif isinstance(inputs, dict):
        inputs = {INPUT}(**inputs)
    values = inputs.model_dump()
    own = client is None
    client = client or httpx.AsyncClient(timeout=30.0, follow_redirects=True)
    vars_: dict[str, Any] = {}
    payload: Any = None
    try:
        for st in SPEC["steps"]:
            url = _render(st["url"], values, vars_)
            for src, dst in (origin_overrides or {}).items():
                if url.startswith(src):
                    url = dst + url[len(src):]
            kw: dict[str, Any] = {}
            if st["body_kind"] == "json":
                kw["json"] = _render(st["body"], values, vars_)
            elif st["body_kind"] == "form":
                kw["data"] = _render(st["body"], values, vars_)
            elif st["body_kind"] == "raw":
                kw["content"] = str(_render(st["body"], values, vars_))
            r = await client.request(st["method"], url,
                                     params=[(k, str(_render(v, values, vars_))) for k, v in st["query"]],
                                     headers={k: str(_render(v, values, vars_)) for k, v in st["headers"].items()}, **kw)
            r.raise_for_status()
            data = r.json() if r.headers.get("content-type", "").startswith("application/json") else None
            for ex in st["extract"]:
                if ex["source"] == "body":
                    vars_[ex["var"]] = _resolve(data, ex["path"])
                elif ex["source"] == "header":
                    vars_[ex["var"]] = r.headers.get(ex["path"])
                else:
                    vars_[ex["var"]] = r.cookies.get(ex["path"]) or client.cookies.get(ex["path"])
            payload = data
    finally:
        if own:
            await client.aclose()
    try:
        return {OUTPUT}.model_validate(payload)
    except ValidationError:
        try:  # optional zero-downtime healing when aegis-mmu is available
            from aegis_mmu.embeddings.providers import FastLocalEngine
            from aegis_mmu.harvester.self_healer import ContractShadow, SchemaHealer

            healed, _report = SchemaHealer(FastLocalEngine()).heal(payload, ContractShadow.from_dict(SHADOW))
            return {OUTPUT}.model_validate(healed)
        except ImportError:
            raise


if __name__ == "__main__":
    import asyncio

    print(asyncio.run(run()).model_dump_json(indent=2))
'''


def generate_client_code(spec: ToolSpec, sample_response: Any, shadow: ContractShadow) -> str:
    cls_base = "".join(p.capitalize() for p in spec.name.split("_"))
    in_cls, out_cls = f"{cls_base}Input", f"{cls_base}Output"
    lines = [f"    {i.name}: {i.type} | None = " +
             (f"Field(default_factory=lambda: os.environ.get({i.env!r}), description='secret: set ${i.env}')" if i.secret
              else f"{i.default!r}") for i in spec.inputs]
    in_src = f"class {in_cls}(BaseModel):\n    \"\"\"Inputs of the harvested `{spec.name}` tool.\"\"\"\n\n" + ("\n".join(lines) or "    pass")
    models: list[str] = []
    _model_sources(out_cls, sample_response if isinstance(sample_response, dict) else {"value": sample_response}, models, set())
    shadow_slim = {"tool_name": shadow.tool_name, "version": shadow.version, "fields": [
        {k: v for k, v in f.__dict__.items()} for f in shadow.fields]}
    header = (
        f'"""Auto-generated by aegis-mmu from a HAR capture — {spec.name}.\n\n{spec.description}\n\n'
        f"Target field: {spec.target} ({spec.target_path}).  Steps: "
        + " -> ".join(f"{s.method} {s.url.split('?')[0]}" for s in spec.steps)
        + '\n"""\n\nimport json\nimport os\nimport re\nfrom typing import Any\n\n'
        "import httpx\nfrom pydantic import BaseModel, ConfigDict, Field, ValidationError\n\n"
    )
    body = (
        header
        + f"SPEC: dict[str, Any] = json.loads({json.dumps(json.dumps(spec.model_dump(), separators=(',', ':')))})\n"
        + f"SHADOW: dict[str, Any] = json.loads({json.dumps(json.dumps(shadow_slim, separators=(',', ':')))})\n\n\n"
        + in_src + "\n\n\n" + "\n\n\n".join(models)
        + _RUNTIME_TEMPLATE.replace("{INPUT}", in_cls).replace("{OUTPUT}", out_cls)
    )
    return body
