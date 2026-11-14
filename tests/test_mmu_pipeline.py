import itertools
import json

import pytest

from aegis_mmu.demo import agent_session_requests, catn
from aegis_mmu.mmu.allocator import allocate, compute_budget
from aegis_mmu.mmu.ast_mipmapper import mipmap_file, syntax_ok
from aegis_mmu.mmu.kv_paginator import PageRequest, SemanticPager, render_context
from aegis_mmu.mmu.prompt_codec import prompt_text
from aegis_mmu.tokens import common_prefix_len, count_tokens

from .conftest import DATASETS, SAMPLES

QUERIES = json.loads((DATASETS / "queries.json").read_text())


def test_fold_codebase_compresses_and_keeps_target_at_l0(mmu, codebase):
    q = QUERIES[0]
    res = mmu.fold(codebase, q["query"], verify=True)
    assert res.tokens_original > 12_000
    assert res.savings_pct > 75
    assert all(f.syntax_ok for f in res.files)
    assert res.tokens_folded <= res.budget * 1.05
    tgt = next(u for u in mmu.store.find_units(q["target"]["path"], q["target"]["symbol"]))
    f = next(f for f in res.files if f.path == q["target"]["path"])
    assert f.leaf_levels.get(tgt.vec_id) == 0
    assert tgt.l0 in f.text
    assert res.hits[0]["qualname"] == q["target"]["symbol"]


def test_fold_is_deterministic(mmu, codebase):
    a = mmu.fold(codebase, QUERIES[2]["query"]).context
    b = mmu.fold(codebase, QUERIES[2]["query"]).context
    assert a == b


def test_small_inputs_pass_through_unfolded(mmu):
    src = (SAMPLES / "billing.py").read_text()
    res = mmu.fold([("billing.py", src)], "tax rounding", budget=10_000)
    assert res.files[0].text == src and res.savings_pct == 0


def test_expand_returns_l0(mmu, codebase):
    res = mmu.fold(codebase, QUERIES[1]["query"])
    vid = next(v for f in res.files for v, lv in f.leaf_levels.items() if lv == 1) if any(
        lv == 1 for f in res.files for lv in f.leaf_levels.values()) else res.hits[0]["vec_id"]
    node = mmu.expand(vid)
    assert node and node["source"].strip() and node["tokens"] > 0
    assert mmu.expand(res.files[0].file_vec_id)["kind"] == "file"


def test_knapsack_dp_is_optimal_vs_bruteforce():
    fm = mipmap_file("billing.py", (SAMPLES / "billing.py").read_text())
    rel = {u.vec_id: r for u, r in zip(fm.units, [0.1, 0.0, 0.9, 1.0, 0.05, 0.7, 0.3])}
    budget = fm.skeleton_tokens[3] + 260
    alloc = allocate([fm], rel, budget)
    from aegis_mmu.mmu.allocator import L1_FIDELITY, L2_FILE_BASE, L2_FILE_VALUE

    best = 0.0
    leaves = [u for u in fm.units if rel.get(u.vec_id, 0) > 1e-4]
    fr = max(rel.values())
    for skel in (2, 3):
        for choice in itertools.product((None, 1, 0), repeat=len(leaves)):
            cost = fm.skeleton_tokens[3] + (fm.skeleton_tokens[2] - fm.skeleton_tokens[3] if skel == 2 else 0)
            val = (L2_FILE_VALUE * fr + L2_FILE_BASE) if skel == 2 else 0.0
            for u, c in zip(leaves, choice):
                if c is not None:
                    cost += u.tokens[c]
                    val += rel[u.vec_id] * (1.0 if c == 0 else L1_FIDELITY)
            if cost <= budget:
                best = max(best, val)
    assert alloc.utility == pytest.approx(best, rel=1e-6)
    assert alloc.estimated_tokens <= budget


def test_budget_policy(settings):
    assert compute_budget(100, settings) == settings.min_budget
    assert compute_budget(20_000, settings) == int(0.22 * 20_000)
    assert compute_budget(10**6, settings) == settings.token_budget


def test_semantic_pager_is_append_only(store):
    pager = SemanticPager(store)
    r1 = pager.page("s", [PageRequest("file:a", 2, "skeleton A", "skeleton", "a"), PageRequest("u1", 1, "u1 L1", "leaf", "a", 3)], 1000)
    r2 = pager.page("s", [PageRequest("file:a", 2, "skeleton A", "skeleton", "a"), PageRequest("u1", 1, "u1 L1", "leaf", "a", 3)], 1000)
    assert r2.kv_hit_rate == pytest.approx(1.0) and r2.appended == []
    # promotion u1 L1 -> L0, degraded request for file:a (L3) keeps the resident L2 page in place
    r3 = pager.page("s", [PageRequest("file:a", 3, "L3 A", "skeleton", "a"), PageRequest("u1", 0, "u1 L0 full", "leaf", "a", 3),
                          PageRequest("u2", 0, "u2 L0", "leaf", "b", 1)], 1000)
    assert r3.promoted == ["u1"] and r3.appended == ["u1", "u2"]
    prev_body = r2.context.rsplit("</aegis_context>", 1)[0]
    assert r3.context.startswith(prev_body)
    assert [p.key for p in r3.pages][:2] == ["file:a", "u1"]
    assert 0.4 <= r3.kv_hit_rate < 1.0


def test_pager_compacts_on_overflow(store):
    pager = SemanticPager(store, overflow=1.2)
    pager.page("c", [PageRequest("a", 0, "x " * 200)], 200)
    r = pager.page("c", [PageRequest("b", 0, "y " * 200)], 200)
    # "a" alone exceeds the pinnable 75% of the cap, so nothing is pinned and the tail is rebuilt
    assert r.compacted and r.epoch == 0 and [p.key for p in r.pages] == ["b"]


def test_compaction_never_touches_the_pinned_prefix(store):
    pager = SemanticPager(store, overflow=2.0, pin_frac=0.75)
    budget = 100
    base = [PageRequest(f"s{i}", 2, f"skeleton {i} " + "w " * 18, "skeleton", f"f{i}") for i in range(6)]
    r1 = pager.page("p", base, budget)
    r2 = pager.page("p", base + [PageRequest("u1", 0, "leaf one " + "x " * 30, "leaf")], budget)
    r3 = pager.page("p", base + [PageRequest("u2", 0, "leaf two " + "y " * 40, "leaf")], budget)
    assert r3.compacted and r3.epoch == r1.epoch
    pinned = [p for p in r3.pages if p.turn == 1]
    assert [p.key for p in pinned] == [p.key for p in r1.pages][: len(pinned)] and len(pinned) >= 5
    head = r1.context.rsplit("</aegis_context>", 1)[0]
    pinned_text = render_context(pinned, r1.epoch).rsplit("</aegis_context>", 1)[0]
    assert head.startswith(pinned_text.rstrip("\n")) and r3.context.startswith(pinned_text.rstrip("\n"))
    assert r3.kv_hit_rate > 0.5 and "u2" in [p.key for p in r3.pages]


def _session(codebase):
    session = json.loads((DATASETS / "session.json").read_text())
    return agent_session_requests(session["turns"], dict(codebase))


def test_rewrite_request_is_turn_anchored_prefix_stable(mmu, codebase):
    reqs = _session(codebase)
    prev = None
    for i, req in enumerate(reqs):
        rr = mmu.rewrite_request(req, "anthropic", session_id="t1")
        cur = prompt_text(rr.body, "anthropic")
        assert rr.tokens_rewritten < rr.tokens_original
        if prev is not None:
            assert cur.startswith(prev), f"turn {i}: previous prompt is not a prefix"
            assert rr.kv_hit_rate == pytest.approx(count_tokens(prev) / count_tokens(cur), rel=0.02)
        prev = cur
    assert not rr.reset


def test_rewrite_tolerates_moving_cache_control(mmu, codebase):
    reqs = _session(codebase)[:2]
    r0 = json.loads(json.dumps(reqs[0]))
    r0["messages"][-1]["content"][-1]["cache_control"] = {"type": "ephemeral"}
    a = mmu.rewrite_request(r0, "anthropic", session_id="cc")
    r1 = json.loads(json.dumps(reqs[1]))
    r1["messages"][-1]["content"][-1]["cache_control"] = {"type": "ephemeral"}
    b = mmu.rewrite_request(r1, "anthropic", session_id="cc")
    assert not b.reset
    assert b.body["messages"][-1]["content"][-1]["cache_control"] == {"type": "ephemeral"}
    assert "cache_control" not in json.dumps(b.body["messages"][2])  # moved off the old turn
    pa = prompt_text(a.body, "anthropic")
    assert prompt_text(b.body, "anthropic").startswith(pa)


def test_edited_history_resets_session(mmu, codebase):
    reqs = _session(codebase)[:2]
    mmu.rewrite_request(reqs[0], "anthropic", session_id="r")
    edited = json.loads(json.dumps(reqs[1]))
    edited["messages"][0]["content"][0]["text"] = "something else entirely"
    assert mmu.rewrite_request(edited, "anthropic", session_id="r").reset


def test_rewritten_tool_result_is_folded_and_valid(mmu, codebase):
    req = _session(codebase)[0]
    rr = mmu.rewrite_request(req, "anthropic", session_id="v")
    tr = rr.body["messages"][2]["content"][0]["content"]
    assert tr.startswith("[aegis-mmu: holographic view of ledgerline/payments/refunds.py")
    body = tr.split("\n", 1)[1]
    assert syntax_ok(body, "python")
    assert "def compute_refund" in body and ("[~aegis:" in body or "[L3 " in body)
    assert req["messages"][2]["content"][0]["content"].startswith("     1\t")  # input not mutated


def test_negative_rag_warns_and_injects(mmu, codebase):
    neg = json.loads((DATASETS / "negative_rag.json").read_text())
    for m in neg["memories"]:
        mmu.negative.record(task=m["task"], code=m["code"], error=m["error"], kind=m["kind"], title=m["title"], why=m["why"], remedy=m["remedy"])
    pos = neg["probes"][4]
    w = mmu.negative.check(pos["task"], pos["code"])
    assert w and "utcnow" in w[0].title and w[0].similarity >= mmu.negative.threshold
    assert not mmu.negative.check("Add a dark-mode toggle to the settings screen")
    req = {"model": "m", "system": "s", "messages": [{"role": "user", "content": pos["task"]}]}
    rr = mmu.rewrite_request(req, "anthropic", session_id="neg")
    last = rr.body["messages"][-1]["content"]
    assert isinstance(last, list) and "<aegis_negative_memory_warning" in last[-1]["text"]
    assert "datetime.now(timezone.utc)" in last[-1]["text"]
    rr2 = mmu.rewrite_request({**req, "messages": req["messages"] + [{"role": "assistant", "content": "ok"},
                                                                     {"role": "user", "content": pos["task"]}]},
                              "anthropic", session_id="neg")
    assert "<aegis_negative_memory_warning" not in json.dumps(rr2.body["messages"][-1])  # warned once per session
    assert "<aegis_negative_memory_warning" in json.dumps(rr2.body["messages"][0])  # history replayed byte-identically


def test_failed_tool_result_is_auto_recorded(mmu):
    req = {"model": "m", "messages": [
        {"role": "user", "content": "run the migration script"},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "Bash", "input": {"command": "python migrate.py --all"}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "is_error": True,
                                      "content": "Traceback (most recent call last):\n  File \"migrate.py\", line 3\nsqlite3.OperationalError: database is locked"}]}]}
    mmu.rewrite_request(req, "anthropic", session_id="auto")
    fails = mmu.store.list_failures()
    assert len(fails) == 1 and "database is locked" in fails[0]["error"] and "migrate.py" in fails[0]["code"]


def test_openai_format_rewrite(mmu, codebase):
    path, src = next((p, s) for p, s in codebase if p.endswith("ledger.py"))
    body = {"model": "gpt", "messages": [
        {"role": "system", "content": "be terse"},
        {"role": "user", "content": f"Fix the cutoff bug in balance_as_of.\n```python {path}\n{src}\n```"}]}
    rr = mmu.rewrite_request(body, "openai", session_id="oa")
    out = rr.body["messages"][1]["content"]
    assert f"```python {path}" in out and "def balance_as_of" in out and len(out) < len(body["messages"][1]["content"])
    assert rr.tokens_rewritten < rr.tokens_original
