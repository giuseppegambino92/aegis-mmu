"""Concept bridging (SE thesaurus), concept-expanded BM25, 1-hop graph propagation, threshold calibration."""

import numpy as np
import pytest

from aegis_mmu.config import AegisSettings
from aegis_mmu.db.store import AegisStore
from aegis_mmu.embeddings.lexicon import concepts_of, expand_term
from aegis_mmu.embeddings.providers import FastLocalEngine
from aegis_mmu.mmu.negative_rag import calibrate_threshold
from aegis_mmu.mmu.pipeline import SemanticMMU
from aegis_mmu.tokens import count_tokens, line_prefix

LIMITER = '''
import time


class Gatekeeper:
    def admit(self, tenant: str, now: float) -> bool:
        slot = self._slot(tenant, now)
        used = self.counters.incr(slot)
        return used <= self.allowance

    def _slot(self, tenant: str, now: float) -> str:
        return f"{tenant}:{int(now // 60)}"


def render_banner(title: str) -> str:
    return f"<h1>{title}</h1>"
'''

HANDLER = '''
from gate import Gatekeeper


def handle_write(request, gatekeeper: Gatekeeper):
    """Reject writes with HTTP 429 Too Many Requests when the tenant is over its allowance."""
    if not gatekeeper.admit(request.tenant, request.now):
        return {"status": 429, "error": "too many requests"}
    return {"status": 200}
'''


def test_concepts_and_expansion_are_multi_membership():
    assert "se_rate_limiting" in concepts_of("429")
    assert {"auth", "se_auth_failure", "se_rate_limiting"} <= set(concepts_of("token"))
    assert "se_cache_key" in concepts_of("memoized")  # via the stemmer
    assert "bucket" in expand_term("429") and "jwt" in expand_term("401")
    assert expand_term("zebra") == ()


def test_engine_bridges_symptom_and_mechanism():
    e = FastLocalEngine()
    q = e.embed_query("clients receive 429 errors")
    mech = e.embed(["def refill_bucket(self, window): self.tokens = min(self.burst, self.tokens + rate)"])[0]
    other = e.embed(["def render_banner(title): return f'<h1>{title}</h1>'"])[0]
    assert float(q @ mech) > float(q @ other) + 0.1


def test_concept_expanded_bm25_finds_zero_overlap_unit(mmu):
    mmu.ingest("lim.py", "def refill(self):\n    self.bucket_level = min(self.burst_cap, self.bucket_level + 1)\n")
    mmu.ingest("ui.py", "def banner(title):\n    return title.upper()\n")
    assert mmu.store.bm25("429 errors from the api", 5) == []
    hits = mmu.store.bm25("429 errors from the api", 5, expand=True)
    assert hits and hits[0][0] == mmu.store.find_units("lim.py", "refill")[0].vec_id


def test_propagation_formula(mmu):
    mmu.ingest("gate.py", LIMITER)
    admit = mmu.store.find_units("gate.py", "Gatekeeper.admit")[0].vec_id
    slot = mmu.store.find_units("gate.py", "Gatekeeper._slot")[0].vec_id
    banner = mmu.store.find_units("gate.py", "render_banner")[0].vec_id
    assert slot in mmu.store.neighbors(admit) and banner not in mmu.store.neighbors(admit)
    cand = [admit, slot, banner]
    out = mmu.retriever._propagate(cand, np.array([1.0, 0.2, 0.5], dtype=np.float32), 0.35)
    np.testing.assert_allclose(out, [1.0 + 0.35 * 0.2, 0.2 + 0.35 * 1.0, 0.5], rtol=1e-6)


DISTRACTORS = '''
def write_request_log(requests, path):
    """Write many requests to the log file."""
    with open(path, "a") as fh:
        for r in requests:
            fh.write(repr(r))


def too_many_open_files(limit_hint):
    return limit_hint > 1024
'''


def test_graph_propagation_lifts_the_called_helper(settings):
    def run(alpha):
        s = settings.model_copy(update={"graph_alpha": alpha, "callee_activation": False, "file_prior_weight": 0.0, "sibling_caller_weight": 0.0, "salience_weight": 0.0})
        st = AegisStore(":memory:", settings=s)
        m = SemanticMMU(st, settings=s)
        m.ingest("gate.py", LIMITER)
        m.ingest("api.py", HANDLER)
        m.ingest("log.py", DISTRACTORS)
        hits = m.retriever.search("writes fail with 429 too many requests").hits
        admit = st.find_units("gate.py", "Gatekeeper.admit")[0].vec_id
        top = hits[0]
        assert st.get_unit(top.vec_id).qualname == "handle_write"  # the symptom matches the handler
        h = next(x for x in hits if x.vec_id == admit)
        return [x.vec_id for x in hits].index(admit) + 1, h.score, top.score

    rank0, s0, _ = run(0.0)
    rank1, s1, _ = run(0.35)
    # the helper the handler calls gains ~alpha x the handler's (pre-propagation) score; ranks never get worse
    assert s1 - s0 == pytest.approx(0.35, abs=0.02)
    assert rank1 <= rank0


def test_edges_are_scoped_to_same_file_or_imports(mmu):
    mmu.ingest("gate.py", LIMITER)
    mmu.ingest("api.py", HANDLER)  # imports gate -> edge allowed
    mmu.ingest("other.py", "def audit(g):\n    return g.admit('t', 0)\n")  # mentions admit, no import
    admit = mmu.store.find_units("gate.py", "Gatekeeper.admit")[0].vec_id
    handler = mmu.store.find_units("api.py", "handle_write")[0].vec_id
    audit = mmu.store.find_units("other.py", "audit")[0].vec_id
    assert admit in mmu.store.neighbors(handler)
    assert admit not in mmu.store.neighbors(audit)


def test_negative_threshold_self_calibrates():
    cal = calibrate_threshold(FastLocalEngine())
    assert cal["source"] == "calibrated"
    assert cal["threshold"] == pytest.approx(max(0.25, cal["null_mean"] + 5 * cal["null_std"]), abs=1e-3)

    class Broken:
        name = "broken"

        def embed(self, texts):
            raise ConnectionError("unreachable")

    assert calibrate_threshold(Broken())["threshold"] == 0.75


def test_line_prefix_matches_direct_counts():
    src = LIMITER + HANDLER
    P = line_prefix(src)
    assert int(P[-1]) == count_tokens(src)
    lines = src.split("\n")
    direct = count_tokens("\n".join(lines[3:12]))
    assert abs(int(P[12] - P[3]) - direct) <= 2


def test_callees_are_extracted_and_resolved(mmu):
    mmu.ingest("gate.py", LIMITER)
    admit = mmu.store.find_units("gate.py", "Gatekeeper.admit")[0].vec_id
    slot = mmu.store.find_units("gate.py", "Gatekeeper._slot")[0].vec_id
    assert mmu.store.callees(admit) == {slot}
    assert mmu.store.callees(slot) == set()


def test_downward_callee_activation(settings):
    def run(on):
        s = settings.model_copy(update={"callee_activation": on, "graph_alpha": 0.0, "file_prior_weight": 0.0, "sibling_caller_weight": 0.0, "salience_weight": 0.0})
        st = AegisStore(":memory:", settings=s)
        m = SemanticMMU(st, settings=s)
        m.ingest("gate.py", LIMITER)
        m.ingest("log.py", DISTRACTORS)
        hits = m.retriever.search("tenant over its allowance is admitted anyway").hits
        ids = [h.vec_id for h in hits]
        admit = st.find_units("gate.py", "Gatekeeper.admit")[0].vec_id
        slot = st.find_units("gate.py", "Gatekeeper._slot")[0].vec_id
        sc = {h.vec_id: h.score for h in hits}
        return ids, admit, slot, sc

    ids, admit, slot, sc = run(True)
    assert ids[0] == admit and ids[1] == slot  # helper right behind the entry point that calls it
    assert sc[slot] < sc[admit]  # capped below its parent
    _, _, _, sc0 = run(False)
    assert sc[slot] > sc0[slot] + 0.2  # activation transfers ~gamma1 of the parent's score


def test_co_promotion_brings_same_file_helpers_to_l0(settings):
    s = settings.model_copy(update={"callee_activation": False, "file_prior_weight": 0.0, "sibling_caller_weight": 0.0, "salience_weight": 0.0})
    m = SemanticMMU(AegisStore(":memory:", settings=s), settings=s)
    filler = "\n".join(f"def unrelated_{i}(x):\n    return x * {i} + {i}\n" for i in range(60))
    res = m.fold([("gate.py", LIMITER + filler)], "tenant allowance admission counters", budget=260)
    lv = res.files[0].leaf_levels
    admit = m.store.find_units("gate.py", "Gatekeeper.admit")[0].vec_id
    slot = m.store.find_units("gate.py", "Gatekeeper._slot")[0].vec_id
    assert lv.get(admit) == 0 and lv.get(slot) == 0


HUB = '''
def _log(msg):
    print(msg)


def _key(user, day):
    _log("key")
    return f"{user}:{day}"


def charge(user, day, amount):
    _log("charge")
    return {"key": _key(user, day), "amount": amount}


def refund(user, day, amount):
    _log("refund")
    return {"amount": -amount}


def audit(user):
    _log("audit")
    return user
'''


def test_exclusive_helper_activation_skips_hub_utilities(settings):
    s = settings.model_copy(update={"callee_activation": True, "callee_max_in_degree": 2, "graph_alpha": 0.0, "file_prior_weight": 0.0, "sibling_caller_weight": 0.0, "salience_weight": 0.0})
    m = SemanticMMU(AegisStore(":memory:", settings=s), settings=s)
    m.ingest("pay.py", HUB)
    unit = {u.qualname: u.vec_id for u in m.store.find_units("pay.py")}
    assert m.store.same_file_in_degree(unit["_log"]) == 4 and m.store.same_file_in_degree(unit["_key"]) == 1
    hits = m.retriever.search("charge amount for a user on a day").hits
    sc = {h.vec_id: h.score for h in hits}
    top = hits[0].vec_id
    assert top == unit["charge"]
    assert sc[unit["_key"]] >= 0.94 * sc[top]  # exclusive helper activated
    assert sc.get(unit["_log"], 0.0) < 0.9 * sc[top]  # hub utility (4 callers) is not


def test_structural_salience_prefers_mutation_over_wrappers():
    from aegis_mmu.mmu.ast_mipmapper import structural_salience

    mutating = "def bump(self, k, now):\n    with self._lock:\n        self.count += 1\n        self.seen[k] = now\n        return self.count % 7 >= 3\n"
    wrapper = "def get_name(self):\n    return describe(self)\n"
    assert structural_salience(mutating) > 0.5 > structural_salience(wrapper) == 0.0


def test_hierarchical_prior_formula_and_top_file_expansion(settings):
    s = settings.model_copy(update={"file_prior_weight": 0.25, "sibling_caller_weight": 0.15, "salience_weight": 0.0,
                                    "callee_activation": False, "stage1_candidates": 2})
    m = SemanticMMU(AegisStore(":memory:", settings=s), settings=s)
    m.ingest("gate.py", LIMITER)
    m.ingest("log.py", DISTRACTORS)
    unit = {u.qualname: u.vec_id for u in m.store.find_units("gate.py")}
    hits = m.retriever.search("tenant allowance admission counters").hits
    ids = [h.vec_id for h in hits]
    assert unit["Gatekeeper._slot"] in ids  # pulled in from the top file even though stage 1 kept only 2
    s0 = settings.model_copy(update={"callee_activation": False, "stage1_candidates": 2, "file_prior_weight": 0.0,
                                     "sibling_caller_weight": 0.0, "salience_weight": 0.0})
    m0 = SemanticMMU(AegisStore(":memory:", settings=s0), settings=s0)
    m0.ingest("gate.py", LIMITER)
    m0.ingest("log.py", DISTRACTORS)
    base = {h.vec_id: h.score for h in m0.retriever.search("tenant allowance admission counters").hits}
    sc = {h.vec_id: h.score for h in hits}
    adm = unit["Gatekeeper.admit"]
    if adm in base:  # admit is the top-file leader: +0.25 * its own score, no caller of its own
        assert sc[adm] == pytest.approx(base[adm] * 1.25, rel=1e-3)
