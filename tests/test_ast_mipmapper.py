import ast
import re
import textwrap

import pytest

from aegis_mmu.mmu.ast_mipmapper import brace_balanced, detect_language, lex_brace, mipmap_file, syntax_ok
from aegis_mmu.tokens import count_tokens

from .conftest import SAMPLES

SAMPLE_FILES = sorted(p.name for p in SAMPLES.iterdir())
FOLD_RE = re.compile(r"\[~aegis:([0-9a-f]{10})~\]")


def _fm(name, l1_mode="signature"):
    return mipmap_file(name, (SAMPLES / name).read_text(), l1_mode=l1_mode)


@pytest.mark.parametrize("name", SAMPLE_FILES)
def test_all_l0_roundtrips_exactly(name):
    fm = _fm(name)
    assert fm.render({u.vec_id: 0 for u in fm.units}) == fm.source


@pytest.mark.parametrize("name", SAMPLE_FILES)
@pytest.mark.parametrize("mode", ["signature", "skeleton"])
def test_every_level_is_syntactically_valid(name, mode):
    fm = _fm(name, mode)
    for level in (1, 2, 3):
        assert syntax_ok(fm.uniform(level), fm.language), f"L{level} of {name} broke syntax"
    for u in fm.units:  # stand-alone pages
        for level in (0, 1, 2):
            page = fm.render_page(u, level)
            assert syntax_ok(page, fm.language), f"page {u.qualname} L{level}"


@pytest.mark.parametrize("name", SAMPLE_FILES)
def test_levels_are_monotonically_smaller(name):
    fm = _fm(name)
    t0, t1 = fm.tokens_source, count_tokens(fm.uniform(1))
    t2, t3 = fm.skeleton_tokens[2], fm.skeleton_tokens[3]
    assert t0 >= t1 > t2 > t3
    for u in fm.units:
        assert u.tokens[0] >= u.tokens[1] >= u.tokens[2]


def test_python_units_signatures_and_types_preserved():
    fm = _fm("billing.py")
    names = {u.qualname: u for u in fm.units}
    assert {"Invoice.total_with_tax", "Invoice._audit", "charge_with_retry", "_fmt", "Invoice.<fields>"} <= set(names)
    l2 = fm.uniform(2)
    assert "    def total_with_tax(country: str) -> int: ..." in l2  # self dropped, types kept
    assert "async def charge_with_retry(client, invoice: Invoice, attempts: int=3) -> dict: ..." in l2
    assert "_audit" not in l2 and "amount_cents" not in l2  # private helpers and fields stripped at L2
    assert '"""' not in l2  # no docstrings at L2
    assert "import" not in l2 and "deps" not in l2  # L2 is an interface skeleton without imports
    l1 = fm.uniform(1)
    assert "amount_cents: int" in l1  # field contracts are still available at L1


def test_signature_mode_l1_is_declaration_plus_summary():
    fm = _fm("billing.py")
    u = next(x for x in fm.units if x.qualname == "charge_with_retry")
    assert u.l1 == ("async def charge_with_retry(client, invoice: Invoice, attempts: int = 3) -> dict:\n"
                    f"    ...  # [~aegis:{u.vec_id}~]")
    t = next(x for x in fm.units if x.qualname == "Invoice.total_with_tax")
    assert t.l1.split("\n")[1].strip() == '"""Return the gross amount in cents, rounded half-even."""'
    assert t.l1.count('"""Return the gross') == 1 and t.l1.endswith(f"...  # [~aegis:{t.vec_id}~]")
    add = next(x for x in _fm("cartStore.ts").units if x.qualname == "CartStore.addLine")
    assert add.l1 == f"  async addLine(sku: string, quantity: number): Promise<void> {{\n      /* [~aegis:{add.vec_id}~] */\n  }}"


def test_python_l1_keeps_control_flow_and_returns():
    fm = _fm("billing.py", "skeleton")
    u = next(x for x in fm.units if x.qualname == "Invoice.total_with_tax")
    assert '"""Return the gross amount in cents, rounded half-even."""' in u.l1
    assert "if rate is None:" in u.l1 and 'raise KeyError(f"no tax rate for {country}")' in u.l1
    assert "return int(gross.quantize" in u.l1
    assert u.tokens[1] <= u.tokens[0]
    big = next(x for x in fm.units if x.qualname == "charge_with_retry")
    assert FOLD_RE.search(big.l1) and big.tokens[1] < 0.6 * big.tokens[0]
    assert "payload = resp.json()" not in big.l1
    for m in FOLD_RE.finditer(u.l1):
        assert m.group(1) == u.vec_id
    ast.parse(textwrap.dedent(u.l1))


def test_python_l1_try_except_finally_headers():
    fm = mipmap_file("b.py", (SAMPLES / "billing.py").read_text(), max_depth=3, l1_mode="skeleton")
    u = next(x for x in fm.units if x.qualname == "charge_with_retry")
    for header in ("for attempt in range(attempts):", "try:", "if resp.status_code == 409:", "elif resp.status_code >= 500:",
                   "else:", "except ConnectionError as exc:", "finally:", 'raise RuntimeError("charge failed") from last_error'):
        assert header in u.l1, header
    ast.parse(u.l1)


def test_typescript_units_and_l2():
    fm = _fm("cartStore.ts", "skeleton")
    q = {u.qualname for u in fm.units}
    assert {"cartTotal", "CartStore.subscribe", "CartStore.addLine", "CartStore.emit", "formatPrice"} <= q
    l2 = fm.uniform(2)
    assert "  async addLine(sku: string, quantity: number): Promise<void> { … }" in l2
    assert "private emit" not in l2
    add = next(u for u in fm.units if u.qualname == "CartStore.addLine")
    assert "try {" in add.l1 and "} catch (e) {" in add.l1 and "} finally {" in add.l1
    assert brace_balanced(add.l1, "typescript")


def test_kotlin_swift_rust_units():
    kt = {u.qualname for u in _fm("CheckoutViewModel.kt").units}
    assert {"CheckoutViewModel.submit", "CheckoutViewModel.applyDiscount", "CheckoutState"} <= kt
    sw = {u.qualname: u for u in _fm("SessionManager.swift", "skeleton").units}
    assert {"SessionManager.init", "SessionManager.accessToken", "SessionManager.refresh", "TokenStore.save"} <= set(sw)
    assert "guard let token = store.load() else {" in sw["SessionManager.accessToken"].l1
    assert sw["SessionManager.accessToken"].doc.startswith("Returns a valid access token")
    rs = {u.qualname: u for u in _fm("discount.rs", "skeleton").units}
    assert {"DiscountEngine.new", "DiscountEngine.discount_for", "clamp"} <= set(rs)
    assert "if let Some(p) = self.overrides.get(sku) {" in rs["DiscountEngine.discount_for"].l1


def test_lexer_ignores_braces_in_strings_comments_chars():
    src = 'let a = "{"; // }\nlet c = \'{\';\n/* { */ fn x<\'a>(s: &\'a str) { }\n'
    infos = lex_brace(src.split("\n"), "rust")
    assert sum(1 for i in infos for _, e in i.events if e == "{") == 1
    assert infos[-1].end_depth == 0
    ts = "const s = `a ${ {x: 1}.x } }`;\nfunction f() {\n}\n"
    assert brace_balanced(ts, "typescript")
    # regex literals: '//' inside a regex is not a comment, '{' inside it is not code
    rx = 'function g(u) {\n  if (!/^https?:\\/\\//.test(u)) {\n    return u.replace(/\\/+$/, "").split(/[{}]/);\n  }\n  return x / 2 / y;\n}\n'
    infos = lex_brace(rx.split("\n"), "typescript")
    assert [i.end_depth for i in infos][:5] == [1, 2, 2, 1, 1] and brace_balanced(rx, "typescript")


def test_l3_summary_and_vec_ids_are_deterministic():
    a, b = _fm("billing.py"), _fm("billing.py")
    assert [u.vec_id for u in a.units] == [u.vec_id for u in b.units]
    assert a.skeleton[3] == f"# [L3 billing: Invoice, charge_with_retry | {a.file_vec_id}]\n"
    for u in a.units:
        assert u.l3.strip().startswith("# [L3 ") and u.vec_id in u.l3


def test_language_detection():
    assert detect_language("x.kt") == "kotlin" and detect_language(fence="rs") == "rust"
    assert detect_language(text="def f(x):\n    return x\n") == "python"
    assert detect_language(text="fun main() {\n}\n") == "kotlin"


def test_mixed_resolution_render_is_valid(codebase):
    for path, src in codebase:
        fm = mipmap_file(path, src)
        levels = {u.vec_id: (i % 3) for i, u in enumerate(fm.units) if i % 3 < 2}
        for skel in (2, 3):
            out = fm.render(levels, skel)
            assert syntax_ok(out, fm.language), (path, skel)
