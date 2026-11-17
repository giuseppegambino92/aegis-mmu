"""aegis-mmu benchmark suite: Naive Vector RAG vs aegis-mmu across five scenarios.

Run:  python -m evals.benchmark_suite      (writes evals/latest_report.json and evals/latest_report.md)

Everything is offline and deterministic (FastLocalEngine, heuristic BPE token counter unless
``tiktoken`` is installed). Latencies are wall-clock on the machine running the suite.
"""

from __future__ import annotations

import ast
import copy
import json
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aegis_mmu import __version__  # noqa: E402
from aegis_mmu.config import AegisSettings  # noqa: E402
from aegis_mmu.db.store import AegisStore  # noqa: E402
from aegis_mmu.demo import agent_session_requests, catn  # noqa: E402
from aegis_mmu.embeddings.colbert import batch_maxsim  # noqa: E402
from aegis_mmu.embeddings.matryoshka import binary_quantize, cosine_topk, hamming_topk, mrl_slice  # noqa: E402
from aegis_mmu.embeddings.providers import FastLocalEngine  # noqa: E402
from aegis_mmu.harvester.self_healer import ContractShadow, SchemaHealer, build_shadow  # noqa: E402
from aegis_mmu.harvester.synthesizer import ToolRuntime, synthesize  # noqa: E402
from aegis_mmu.harvester.taint_graph import extract_minimal_chain  # noqa: E402
from aegis_mmu.mmu.ast_mipmapper import brace_balanced, mipmap_file  # noqa: E402
from aegis_mmu.mmu.pipeline import SemanticMMU  # noqa: E402
from aegis_mmu.mmu.prompt_codec import prompt_text  # noqa: E402
from aegis_mmu.runtime import tune_gc  # noqa: E402
from aegis_mmu.tokens import common_prefix_len, count_tokens, prefix_token_hit_rate, tokenizer_name  # noqa: E402

DS = ROOT / "evals" / "datasets"
NEURAL = {"embedding_provider": "ollama", "embedding_model": "nomic-embed-text",
          "embedding_cache_path": str(ROOT / "evals" / ".cache" / "embeddings.db")}


NEURAL_RESULTS = ROOT / "evals" / ".cache" / "neural_results.json"


def dataset_fingerprint() -> str:
    """Hash of every benchmark input + the retrieval-relevant source, so cached neural results are
    only reused for exactly the same data and code."""
    import hashlib

    h = hashlib.blake2b(digest_size=12)
    for p in sorted(list((DS / "codebase").rglob("*")) + [DS / "queries.json", DS / "queries_heldout.json",
                    DS / "queries_heldout_v2.json", DS / "queries_heldout_v3.json",
                    DS / "queries_heldout_v4.json", DS / "queries_heldout_v5.json"] + sorted((ROOT / "src" / "aegis_mmu").rglob("*.py"))):
        if p.is_file():
            h.update(str(p.relative_to(ROOT)).encode())
            h.update(p.read_bytes())
    return h.hexdigest()


def neural_available() -> bool:
    try:
        from aegis_mmu.embeddings.providers import OllamaEngine

        return OllamaEngine(cache_path=None).available()
    except Exception:  # noqa: BLE001
        return False


def wait_for_quiet(max_load: float, timeout_s: float = 1800.0) -> list[float]:
    """Block until the 1-minute load average drops below ``max_load`` (locale-independent: uses
    ``os.getloadavg()``, never parses ``uptime``/``sysctl`` text with comma decimals)."""
    import os

    t0 = time.time()
    while os.getloadavg()[0] >= max_load and time.time() - t0 < timeout_s:
        time.sleep(5)
    return [round(x, 2) for x in os.getloadavg()]


def pct(xs, q):
    return float(np.percentile(np.asarray(xs, dtype=np.float64), q)) if xs else 0.0


def lat_stats(xs) -> dict:
    return {"p50_ms": round(pct(xs, 50), 4), "p95_ms": round(pct(xs, 95), 4), "p99_ms": round(pct(xs, 99), 4), "n": len(xs)}


def timeit(fn, n: int, warmup: int = 3) -> list[float]:
    for _ in range(warmup):
        fn()
    out = []
    for _ in range(n):
        t = time.perf_counter()
        fn()
        out.append((time.perf_counter() - t) * 1e3)
    return out


def fresh_mmu(**over) -> SemanticMMU:
    s = AegisSettings(db_path=":memory:").model_copy(update=over)
    return SemanticMMU(AegisStore(":memory:", settings=s), settings=s)


def load_codebase(name: str = "codebase"):
    root = DS / name
    return [(str(p.relative_to(root)), p.read_text()) for p in sorted(root.rglob("*")) if p.is_file()]


# ----------------------------------------------------------------------------- naive vector RAG baseline


class NaiveRAG:
    """Fixed-size line windows (~256 tokens) embedded with the same engine; dense top-k by cosine."""

    def __init__(self, files, engine, window_tokens: int = 256):
        self.engine = engine
        self.chunks: list[dict] = []
        for path, src in files:
            lines = src.split("\n")
            buf, start = [], 0
            for i, ln in enumerate(lines):
                buf.append(ln)
                if count_tokens("\n".join(buf)) >= window_tokens:
                    self._add(path, buf, start)
                    buf, start = [], i + 1
            if any(x.strip() for x in buf):
                self._add(path, buf, start)
        self.E = engine.embed([f"{c['path']}\n{c['text']}" for c in self.chunks])

    def _add(self, path, buf, start):
        text = "\n".join(buf)
        self.chunks.append({"path": path, "text": text, "start": start, "tokens": count_tokens(text)})

    def retrieve(self, query: str, budget: int) -> list[dict]:
        q = self.engine.embed_query(query)
        order = np.argsort(-(self.E @ q), kind="stable")
        out, used = [], 0
        for i in order:
            c = self.chunks[int(i)]
            if used + c["tokens"] > budget:
                continue
            out.append(c)
            used += c["tokens"]
        return out


def chunk_syntax_ok(c: dict) -> bool:
    p = c["path"]
    if p.endswith(".py"):
        try:
            ast.parse(c["text"])
            return True
        except SyntaxError:
            return False
    ext = p.rsplit(".", 1)[-1]
    lang = {"ts": "typescript", "kt": "kotlin", "swift": "swift", "rs": "rust"}.get(ext)
    return brace_balanced(c["text"], lang) if lang else True


# ----------------------------------------------------------------------------- scenario 1


def scenario_compression(files, queries, **over) -> dict:
    mmu = fresh_mmu(**over)
    for p, s in files:
        mmu.ingest(p, s)
    naive = NaiveRAG(files, mmu.engine)
    rows, a_red, a_rec3, a_cov, a_syn, n_rec3, n_cov, n_syn, lats, n_lines = [], [], [], [], [], [], [], [], [], []
    a_clu: list[bool] = []
    original = sum(count_tokens(s) for _, s in files)
    for q in queries:
        res = mmu.fold(files, q["query"], record=False, verify=True)
        tgt = mmu.store.find_units(q["target"]["path"], q["target"]["symbol"])[0]
        rank = next((i + 1 for i, h in enumerate(res.hits) if h["vec_id"] == tgt.vec_id), None)
        level = next((f.leaf_levels.get(tgt.vec_id) for f in res.files if f.path == tgt.path), None)
        in_l0 = level == 0
        rec3 = bool(rank and rank <= 3 and in_l0)
        # call-cluster variant: target at L0 and either top-3 itself or a same-file 1/2-hop callee of a top-3 hit
        cluster = rec3
        if in_l0 and not cluster:
            for h in res.hits[:3]:
                c1 = {c for c in mmu.store.callees(h["vec_id"]) if mmu.store.same_file(h["vec_id"], c)}
                c2 = {c2 for c in c1 for c2 in mmu.store.callees(c) if mmu.store.same_file(c, c2)}
                if tgt.vec_id in c1 | c2:
                    cluster = True
                    break
        a_clu.append(cluster)
        syn = all(f.syntax_ok for f in res.files)
        # naive with the same token budget aegis used
        chunks = naive.retrieve(q["query"], res.tokens_folded)
        body = tgt.l0.strip()
        n_hit = [i for i, c in enumerate(chunks) if body in c["text"]]
        covered = set()
        for c in chunks:
            if c["path"] == tgt.path:
                covered.update(range(c["start"] + 1, c["start"] + 1 + c["text"].count("\n") + 1))
        span = set(range(tgt.start_line, tgt.end_line + 1))
        n_lines.append(len(span & covered) / len(span))
        n_rec3.append(bool(n_hit and n_hit[0] < 3))
        n_cov.append(bool(n_hit))
        n_syn.append(sum(chunk_syntax_ok(c) for c in chunks) / max(1, len(chunks)))
        a_red.append(res.savings_pct)
        a_rec3.append(rec3)
        a_cov.append(in_l0)
        a_syn.append(syn)
        lats.append(res.latency_ms)
        rows.append({"id": q["id"], "target": f"{tgt.path.split('/')[-1]}::{tgt.qualname}", "tokens_original": res.tokens_original,
                     "tokens_folded": res.tokens_folded, "reduction_pct": round(res.savings_pct, 2), "target_rank": rank,
                     "target_level": None if level is None else f"L{level}", "l0_recall_at_3": rec3, "cluster_recall_at_3": cluster,
                     "syntax_ok": syn,
                     "latency_ms": round(res.latency_ms, 3), "naive_target_whole_in_context": bool(n_hit)})
    return {
        "codebase_tokens": original, "files": len(files), "units": mmu.store.size, "queries": len(queries),
        "aegis": {"token_reduction_pct": round(statistics.mean(a_red), 2), "l0_recall_at_3": round(sum(a_rec3) / len(a_rec3), 3),
                  "cluster_l0_recall_at_3": round(sum(a_clu) / len(a_clu), 3),
                  "target_at_l0_any_rank": round(sum(a_cov) / len(a_cov), 3),
                  "ast_syntax_preservation": round(sum(a_syn) / len(a_syn), 3), "fold_latency": lat_stats(lats)},
        "naive_rag": {"token_reduction_pct": round(statistics.mean(a_red), 2), "target_whole_in_top3": round(sum(n_rec3) / len(n_rec3), 3),
                      "target_whole_in_context": round(sum(n_cov) / len(n_cov), 3),
                      "target_lines_covered_mean": round(statistics.mean(n_lines), 3),
                      "chunk_syntax_preservation": round(statistics.mean(n_syn), 3), "note": "same token budget as aegis"},
        "per_query": rows,
    }


def scenario_levels(files) -> dict:
    """Per-level size ratios: L1 vs L0 of every function/method; L2/L3 file skeletons vs whole files."""
    t0 = t1 = src = s2 = s3 = 0
    fn_sizes = []
    for p, s in files:
        fm = mipmap_file(p, s)
        src += fm.tokens_source
        s2 += fm.skeleton_tokens[2]
        s3 += fm.skeleton_tokens[3]
        for u in fm.units:
            if u.kind in ("function", "method"):
                t0 += u.tokens[0]
                t1 += u.tokens[1]
                fn_sizes.append(u.tokens[0])
    return {"files": len(files), "tokens": src, "functions": len(fn_sizes),
            "function_tokens_median": float(statistics.median(fn_sizes)) if fn_sizes else 0.0,
            "l1_over_l0": round(t1 / max(1, t0), 4), "l2_over_file": round(s2 / max(1, src), 4), "l3_over_file": round(s3 / max(1, src), 4)}


# ----------------------------------------------------------------------------- scenario 2


def scenario_kv(files, session) -> dict:
    turns = session["turns"]
    fmap = dict(files)
    reqs = agent_session_requests(turns, fmap, system="You are a careful senior engineer working in the ledgerline monorepo.")
    # --- (a) full-conversation prefix accounting through the proxy rewriter (repo page-in enabled)
    mmu = fresh_mmu(repo_page_in=True)
    mmu.index_paths(DS / "codebase")
    a_prompts, a_rates, a_tokens, a_lat = [], [], [], []
    for r in reqs:
        rr = mmu.rewrite_request(r, "anthropic", session_id="bench")
        a_prompts.append(prompt_text(rr.body, "anthropic"))
        a_rates.append(rr.kv_hit_rate)
        a_tokens.append(rr.tokens_rewritten)
        a_lat.append(rr.latency_ms)
    # standard RAG: per-turn top-k chunks re-ranked into the system prompt, tool results verbatim
    naive = NaiveRAG(files, mmu.engine)
    n_prompts, n_rates, n_tokens = [], [], []
    for t, r in enumerate(reqs):
        chunks = naive.retrieve(turns[t]["query"], 2500)
        body = copy.deepcopy(r)
        body["system"] = r["system"] + "\n\n<retrieved_context>\n" + "\n---\n".join(
            f"# {c['path']}:{c['start'] + 1}\n{c['text']}" for c in chunks) + "\n</retrieved_context>"
        pt = prompt_text(body, "anthropic")
        n_rates.append(prefix_token_hit_rate(n_prompts[-1], pt) if n_prompts else 0.0)
        n_prompts.append(pt)
        n_tokens.append(count_tokens(pt))

    def bound(prompts):
        return [0.0] + [count_tokens(prompts[i - 1]) / count_tokens(prompts[i]) for i in range(1, len(prompts))]

    def retention(prompts):
        out = [0.0]
        for i in range(1, len(prompts)):
            lcp = common_prefix_len(prompts[i - 1], prompts[i])
            out.append(count_tokens(prompts[i][:lcp]) / count_tokens(prompts[i - 1]))
        return out

    a_bound, n_bound = bound(a_prompts), bound(n_prompts)
    conv = {
        "aegis": {"kv_hit_rate_per_turn": [round(x, 4) for x in a_rates], "kv_hit_rate_avg_t2_5": round(statistics.mean(a_rates[1:]), 4),
                  "append_only_optimum_per_turn": [round(x, 4) for x in a_bound],
                  "prefix_retention_avg_t2_5": round(statistics.mean(retention(a_prompts)[1:]), 4),
                  "prompt_tokens_per_turn": a_tokens, "rewrite_latency": lat_stats(a_lat[1:])},
        "standard_rag": {"kv_hit_rate_per_turn": [round(x, 4) for x in n_rates], "kv_hit_rate_avg_t2_5": round(statistics.mean(n_rates[1:]), 4),
                         "append_only_optimum_per_turn": [round(x, 4) for x in n_bound],
                         "prefix_retention_avg_t2_5": round(statistics.mean(retention(n_prompts)[1:]), 4),
                         "prompt_tokens_per_turn": n_tokens},
    }
    # --- (b) spec-literal context block: aegis append-only pager vs re-ranked top-k block (same queries)
    mmu2 = fresh_mmu()
    b_rates, b_tokens = [], []
    for t in turns:
        pr = mmu2.context_block("block", files, t["query"])
        b_rates.append(pr.kv_hit_rate)
        b_tokens.append(pr.resident_tokens)
    mmu3 = fresh_mmu(kv_policy="adaptive")
    ad_rates, ad_tokens = [], []
    for t in turns:
        pr = mmu3.context_block("block", files, t["query"])
        ad_rates.append(pr.kv_hit_rate)
        ad_tokens.append(pr.resident_tokens)
    nb_prev, nb_rates, nb_tokens = "", [], []
    for t in turns:
        chunks = naive.retrieve(t["query"], int(statistics.mean(b_tokens)))
        ctx = '<aegis_context epoch="0">\n' + "\n".join(f"# {c['path']}:{c['start'] + 1}\n{c['text']}" for c in chunks) + "\n</aegis_context>\n"
        nb_rates.append(prefix_token_hit_rate(nb_prev, ctx) if nb_prev else 0.0)
        nb_tokens.append(count_tokens(ctx))
        nb_prev = ctx
    block = {"aegis_pager": {"kv_hit_rate_per_turn": [round(x, 4) for x in b_rates], "kv_hit_rate_avg_t2_5": round(statistics.mean(b_rates[1:]), 4),
                             "context_tokens_per_turn": b_tokens},
             "aegis_pager_adaptive": {"kv_hit_rate_per_turn": [round(x, 4) for x in ad_rates],
                                      "kv_hit_rate_avg_t2_5": round(statistics.mean(ad_rates[1:]), 4), "context_tokens_per_turn": ad_tokens},
             "standard_rag": {"kv_hit_rate_per_turn": [round(x, 4) for x in nb_rates], "kv_hit_rate_avg_t2_5": round(statistics.mean(nb_rates[1:]), 4),
                              "context_tokens_per_turn": nb_tokens}}
    return {"turns": len(turns), "conversation": conv, "context_block": block}


# ----------------------------------------------------------------------------- scenario 3


def scenario_negative() -> dict:
    data = json.loads((DS / "negative_rag.json").read_text())
    mmu = fresh_mmu()
    ids = {}
    for m in data["memories"]:
        ids[mmu.negative.record(task=m["task"], code=m["code"], error=m["error"], kind=m["kind"], title=m["title"],
                                why=m["why"], remedy=m["remedy"])] = m["id"]
    thr = mmu.negative.threshold
    calibration = mmu.negative.calibration
    tp = fp = fn = tn = correct = 0
    scores, labels, per = [], [], []
    for p in data["probes"]:
        sc = mmu.negative.scores(p["task"], p["code"])
        top_id, top_s = sc[0]
        fired = top_s >= thr
        positive = p["label"] is not None
        scores.append(top_s)
        labels.append(positive)
        if fired and positive:
            tp += 1
            correct += ids[top_id] == p["label"]
        elif fired:
            fp += 1
        elif positive:
            fn += 1
        else:
            tn += 1
        per.append({"label": p["label"], "top": ids[top_id], "score": round(top_s, 4), "fired": fired})
    pos = [s for s, l in zip(scores, labels) if l]
    neg = [s for s, l in zip(scores, labels) if not l]
    auc = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))
    lat = timeit(lambda: mmu.negative.check(data["probes"][0]["task"], data["probes"][0]["code"]), 200)
    return {"memories": len(data["memories"]), "probes": len(data["probes"]), "threshold": thr, "calibration": calibration,
            "precision": round(tp / max(1, tp + fp), 3), "recall": round(tp / max(1, tp + fn), 3),
            "pattern_attribution_accuracy": round(correct / max(1, tp), 3), "roc_auc": round(auc, 4),
            "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn}, "check_latency": lat_stats(lat), "per_probe": per}


# ----------------------------------------------------------------------------- scenario 4


def scenario_harvest() -> dict:
    import asyncio

    import httpx

    from evals.datasets.skyfare_server import create_app

    truth = json.loads((DS / "har_truth.json").read_text())
    drift_truth = json.loads((DS / "drift_truth.json").read_text())
    v1 = json.loads((DS / "flight_v1.json").read_text())
    v2 = json.loads((DS / "flight_v2_drift.json").read_text())
    lat_slice = timeit(lambda: extract_minimal_chain(DS / "skyfare.har", "price_eur"), 50)
    mc = extract_minimal_chain(DS / "skyfare.har", "price_eur")
    eng = FastLocalEngine()
    healer = SchemaHealer(eng)
    shadow = build_shadow("skyfare_flight_search", v1, eng)
    heal_lat, rep, healed = [], None, None
    for _ in range(3):
        healer.heal(v2, ContractShadow.from_dict(copy.deepcopy(shadow.to_dict())))
    for _ in range(100):
        sh = ContractShadow.from_dict(copy.deepcopy(shadow.to_dict()))
        healed, rep = healer.heal(v2, sh)
        heal_lat.append(rep.latency_ms)
    got = {m.expected: m.actual for m in rep.mappings}
    renamed = [p for p, q in drift_truth.items() if p.rsplit(".", 1)[-1] != q.rsplit(".", 1)[-1]]
    alias_lat = []
    sh = ContractShadow.from_dict(copy.deepcopy(shadow.to_dict()))
    healer.heal(v2, sh)
    for _ in range(100):
        _, r2 = healer.heal(v2, sh)
        alias_lat.append(r2.latency_ms)
    # live end-to-end: synthesised tool vs a server issuing fresh CSRF/JWT, serving the drifted schema
    mmu = fresh_mmu()
    tool = synthesize(mc, "skyfare_flight_search", mmu.engine)
    rt = ToolRuntime(mmu.store)
    rt.register(tool)

    async def live():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(drift=True))) as c:
            return await rt.call("skyfare_flight_search", {"password": "correct-horse-battery-9"}, client=c)

    t0 = time.perf_counter()
    out = asyncio.run(live())
    live_ms = (time.perf_counter() - t0) * 1e3
    expected_first = {"flight_number": "AZ608", "price_eur": 398.4, "seats_available": 5}
    first = out["data"]["data"]["flights"][0]
    return {
        "har_requests": len(mc.txns), "categories": mc.categories, "minimal_chain": mc.chain, "expected_chain": truth["minimal_chain"],
        "chain_exact": mc.chain == truth["minimal_chain"], "noise_pruned_pct": round(100 * mc.noise_reduction, 2),
        "slice_latency": lat_stats(lat_slice),
        "drift": {"contract_fields": len(drift_truth), "renamed_fields": len(renamed), "moved_fields": sum(p != q for p, q in drift_truth.items()),
                  "fields_recovered": sum(got.get(p) == q for p, q in drift_truth.items()),
                  "renamed_recovered": sum(got.get(p) == drift_truth[p] for p in renamed),
                  "recovery_rate": round(sum(got.get(p) == q for p, q in drift_truth.items()) / len(drift_truth), 4),
                  "fresh_values": True, "healed_validates": healer.validate(healed, shadow) == [],
                  "hungarian_latency": lat_stats(heal_lat), "alias_fastpath_latency": lat_stats(alias_lat)},
        "live_replay": {"steps": [s["status"] for s in out["steps"]], "drift_strategy": out["drift"]["strategy"],
                        "healed": out["drift"]["healed"], "values_correct": all(first[k] == v for k, v in expected_first.items()),
                        "wall_ms": round(live_ms, 2)},
    }


# ----------------------------------------------------------------------------- scenario 5


def cold_path(files, trials: int = 90) -> dict:
    """First sight of never-seen files through the proxy rewriter, on an on-disk WAL database that
    already holds the indexed codebase. Every trial is a unique file variant (content hash miss), so
    parsing, mipmapping, embedding, FTS/graph persistence and allocation all run cold."""
    import tempfile

    tmp = tempfile.mkdtemp(prefix="aegis-cold-")
    db = str(Path(tmp) / "cold.db")
    s = AegisSettings(db_path=db)
    mmu = SemanticMMU(AegisStore(db, settings=s), settings=s)
    mmu.index_paths(DS / "codebase")
    tune_gc()  # same serving-process GC policy the proxy / MCP entry points apply after warm-up
    rew, ing, ing_cpu, rew_cpu = [], [], [], []
    for k in range(trials):
        path, src = files[k % len(files)]
        marker = "#" if path.endswith(".py") else "//"
        variant = f"{src.rstrip()}\n{marker} variant {k}\n"
        body = {"model": "m", "system": "You are a coding agent.", "messages": [
            {"role": "user", "content": [{"type": "text", "text": "the refund amount is off by one cent for JPY orders"}]},
            {"role": "assistant", "content": [{"type": "tool_use", "id": "t", "name": "Read", "input": {"file_path": path}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t", "content": catn(variant)}]}]}
        c0 = time.process_time()
        rr = mmu.rewrite_request(body, "anthropic", session_id=f"cold{k}")
        rew_cpu.append((time.process_time() - c0) * 1e3)
        rew.append(rr.latency_ms)
        t, c = time.perf_counter(), time.process_time()
        mmu.ingest("x/" + path, variant + f"{marker} ingest-only\n")
        ing.append((time.perf_counter() - t) * 1e3)
        ing_cpu.append((time.process_time() - c) * 1e3)
    mmu.store.close()
    import gc

    gc.unfreeze()
    sizes = [count_tokens(s) for _, s in files]
    return {"proxy_rewrite_first_sight_cold": lat_stats(rew), "file_ingest_cold": lat_stats(ing),
            "proxy_rewrite_first_sight_cold_cpu": lat_stats(rew_cpu), "file_ingest_cold_cpu": lat_stats(ing_cpu),
            "cold_file_tokens_mean": round(statistics.mean(sizes), 1)}


def neural_scenarios(files, queries) -> dict:
    """Neural Hybrid tier (Ollama nomic-embed-text): recall on dev + both held-out sets, the
    composite-index ablation, and model-call latency with fresh (uncached) queries.

    Two-phase protocol: ``--neural-only`` computes these while Ollama runs and caches them (keyed by
    the dataset/code fingerprint); a later full run with Ollama stopped reuses the cached results so
    the daemon cannot perturb the latency micro-benchmarks."""
    fp = dataset_fingerprint()
    if not neural_available():
        if NEURAL_RESULTS.exists():
            cached = json.loads(NEURAL_RESULTS.read_text())
            if cached.get("fingerprint") == fp:
                return {"neural": cached["neural"] | {"from_cache": True}}
        return {"neural": {"available": False}}
    held = json.loads((DS / "queries_heldout.json").read_text())
    held2 = json.loads((DS / "queries_heldout_v5.json").read_text())
    out = {"available": True, "model": "nomic-embed-text (Ollama, 768-d truncated to 512)"}
    out["dev"] = scenario_compression(files, queries, **NEURAL)
    out["heldout"] = scenario_compression(files, held, **NEURAL)
    out["heldout_v2_seen"] = scenario_compression(files, json.loads((DS / "queries_heldout_v2.json").read_text()), **NEURAL)
    out["heldout_v3_seen"] = scenario_compression(files, json.loads((DS / "queries_heldout_v3.json").read_text()), **NEURAL)
    out["heldout_v2"] = scenario_compression(files, held2, **NEURAL)  # key kept for the report: = blind v5
    out["heldout_composite"] = scenario_compression(files, held2, composite_index=True, **NEURAL)
    out["dev_composite"] = scenario_compression(files, queries, composite_index=True, **NEURAL)
    mmu = fresh_mmu(**NEURAL)
    mmu.ingest_many(files)
    import uuid

    q_lat = []
    for i in range(40):  # fresh text every time -> a real model call, no cache hit
        q = f"{queries[i % len(queries)]['query']} [{uuid.uuid4().hex[:8]}]"
        t = time.perf_counter()
        mmu.retriever.search(q, mmu.store.all_vec_ids())
        q_lat.append((time.perf_counter() - t) * 1e3)
    ing = []
    for k in range(10):
        p, src = files[k % len(files)]
        marker = "#" if p.endswith(".py") else "//"
        t = time.perf_counter()
        mmu.ingest(f"n{k}/" + p, f"{src}\n{marker} {uuid.uuid4().hex}\n")
        ing.append((time.perf_counter() - t) * 1e3)
    out["retrieval_fresh_query"] = lat_stats(q_lat)
    out["file_ingest_uncached"] = lat_stats(ing)
    NEURAL_RESULTS.parent.mkdir(parents=True, exist_ok=True)
    NEURAL_RESULTS.write_text(json.dumps({"fingerprint": fp, "neural": out}))
    return {"neural": out}


def scenario_latency(files, queries) -> dict:
    eng = FastLocalEngine()
    rng = np.random.default_rng(7)
    V = rng.standard_normal((10_000, 512)).astype(np.float32)
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    B = binary_quantize(V)
    M = mrl_slice(V, 64)
    q = V[0] + 0.1 * rng.standard_normal(512).astype(np.float32)
    qb = binary_quantize(q)
    qm = mrl_slice(q, 64)
    mmu = fresh_mmu()
    for p, s in files:
        mmu.ingest(p, s)
    cands = mmu.store.all_vec_ids()[:48]
    toks = mmu.store.token_matrices(cands)
    qt = eng.embed_tokens(queries[0]["query"], is_query=True)
    qi = iter(range(10**9))
    big = max(files, key=lambda f: len(f[1]))
    unit_docs = [u.embed_text() for _, s in files for u in mipmap_file(_, s).units] * 10
    eng.embed(unit_docs[:100])
    t = time.perf_counter()
    eng.embed(unit_docs)
    thr = len(unit_docs) / (time.perf_counter() - t)
    file_docs = [s for _, s in files] * 10
    t = time.perf_counter()
    eng.embed(file_docs)
    thr_files = len(file_docs) / (time.perf_counter() - t)
    unit_tokens = statistics.mean(count_tokens(d) for d in unit_docs[: len(unit_docs) // 10])
    mm = [timeit(lambda f=f: mipmap_file(f[0], f[1]), 10, 1) for f in files]
    session = json.loads((DS / "session.json").read_text())
    reqs = agent_session_requests(session["turns"], dict(files))
    mmu2 = fresh_mmu()
    rew = []
    for k in range(20):  # warm turns: same conversation replayed under fresh session ids
        for r in reqs:
            rr = mmu2.rewrite_request(r, "anthropic", session_id=f"lat{k}")
            if k > 0:
                rew.append(rr.latency_ms)
    v1 = json.loads((DS / "flight_v1.json").read_text())
    v2 = json.loads((DS / "flight_v2_drift.json").read_text())
    healer = SchemaHealer(eng)
    shadow = build_shadow("t", v1, eng)
    return {
        "bq_hamming_top50_10k": lat_stats(timeit(lambda: hamming_topk(qb, B, 50), 500)),
        "mrl64_cosine_top50_10k": lat_stats(timeit(lambda: cosine_topk(qm, M, 50), 500)),
        "fts5_bm25_codebase": lat_stats(timeit(lambda: mmu.store.bm25(queries[next(qi) % len(queries)]["query"], 48, cands + mmu.store.all_vec_ids()), 200)),
        "colbert_maxsim_rerank_48": lat_stats(timeit(lambda: batch_maxsim(qt, toks), 500)),
        "hybrid_retrieval_end_to_end": lat_stats(timeit(lambda: mmu.retriever.search(queries[next(qi) % len(queries)]["query"], mmu.store.all_vec_ids()), 200)),
        "ast_mipmap_per_file_cold": lat_stats([x for xs in mm for x in xs]),
        "ast_mipmap_largest_file": {"path": big[0], "tokens": count_tokens(big[1]), **lat_stats(timeit(lambda: mipmap_file(big[0], big[1]), 20, 2))},
        "fold_pipeline_warm_16k_tokens": lat_stats(timeit(lambda: mmu.fold(files, queries[next(qi) % len(queries)]["query"], record=False), 100)),
        "proxy_rewrite_turn_warm": lat_stats(rew),
        **cold_path(files),
        "hungarian_self_heal": lat_stats([healer.heal(v2, ContractShadow.from_dict(copy.deepcopy(shadow.to_dict())))[1].latency_ms for _ in range(200)]),
        "embedding_throughput_docs_per_s": round(thr, 1),
        "embedding_doc_tokens_mean": round(unit_tokens, 1),
        "embedding_throughput_whole_files_per_s": round(thr_files, 1),
    }


# ----------------------------------------------------------------------------- report


def compliance(r: dict) -> list[dict]:
    """Every design target, its acceptance rule and the measured value -> pass/fail (computed, not typed)."""
    S = r["scenarios"]
    c1, ch, lv, kv, h, lat = S["compression"], S["compression_heldout"], S["levels"]["codebase"], S["kv_cache"], S["harvest"], S["latency"]
    nz = S.get("neural", {})
    neural_rows = []
    if nz.get("available"):
        neural_rows = [
            ("L0 Recall@3 >= 90% — blind held-out v5, Neural Hybrid (nomic-embed-text)", 100 * nz["heldout_v2"]["aegis"]["l0_recall_at_3"], lambda v: v >= 90, "{:.0f}%"),
            ("Call-cluster L0 Recall@3 >= 90% — blind held-out v5, Neural Hybrid", 100 * nz["heldout_v2"]["aegis"]["cluster_l0_recall_at_3"], lambda v: v >= 90, "{:.0f}%"),
            ("L0 Recall@3 >= 90% — dev, Neural Hybrid (nomic-embed-text)", 100 * nz["dev"]["aegis"]["l0_recall_at_3"], lambda v: v >= 90, "{:.0f}%"),
        ]
    rows = neural_rows + [
        ("Token reduction > 75%", c1["aegis"]["token_reduction_pct"], lambda v: v > 75, "{:.1f}%"),
        ("L0 Recall@3 >= 90% — blind held-out v5, FastLocal 0-dep", 100 * ch["aegis"]["l0_recall_at_3"], lambda v: v >= 90, "{:.0f}%"),
        ("Call-cluster L0 Recall@3 >= 90% — blind held-out v5, FastLocal 0-dep", 100 * ch["aegis"]["cluster_l0_recall_at_3"], lambda v: v >= 90, "{:.0f}%"),
        ("Target at full L0 anywhere in context >= 90% — blind held-out v5, FastLocal (not a top-3 metric)", 100 * ch["aegis"]["target_at_l0_any_rank"], lambda v: v >= 90, "{:.0f}%"),
        ("L0 Recall@3 >= 90% — dev, FastLocal 0-dep", 100 * c1["aegis"]["l0_recall_at_3"], lambda v: v >= 90, "{:.0f}%"),
        ("AST syntax preservation = 100%", 100 * min(c1["aegis"]["ast_syntax_preservation"], ch["aegis"]["ast_syntax_preservation"]), lambda v: v == 100, "{:.0f}%"),
        ("L1 ~25% of L0 (accept <= 30%)", 100 * lv["l1_over_l0"], lambda v: v <= 30, "{:.1f}%"),
        ("L2 ~6% of file (accept <= 8%)", 100 * lv["l2_over_file"], lambda v: v <= 8, "{:.1f}%"),
        ("L3 ~1-2% of file (accept <= 2%)", 100 * lv["l3_over_file"], lambda v: v <= 2, "{:.2f}%"),
        ("KV prefix hit > 85% (context block, stable-prefix policy)", 100 * kv["context_block"]["aegis_pager"]["kv_hit_rate_avg_t2_5"], lambda v: v > 85, "{:.1f}%"),
        ("Standard RAG prefix hit < 25% (baseline sanity)", 100 * kv["context_block"]["standard_rag"]["kv_hit_rate_avg_t2_5"], lambda v: v < 25, "{:.1f}%"),
        ("Warm fold pipeline p95 < 10 ms", lat["fold_pipeline_warm_16k_tokens"]["p95_ms"], lambda v: v < 10, "{:.2f} ms"),
        ("Warm proxy rewrite p95 < 10 ms", lat["proxy_rewrite_turn_warm"]["p95_ms"], lambda v: v < 10, "{:.2f} ms"),
        ("Cold first-sight proxy rewrite p95 < 10 ms", lat["proxy_rewrite_first_sight_cold"]["p95_ms"], lambda v: v < 10, "{:.2f} ms"),
        ("Cold per-file ingest p95 < 10 ms", lat["file_ingest_cold"]["p95_ms"], lambda v: v < 10, "{:.2f} ms"),
        ("Cold first-sight rewrite, CPU-time p95 < 10 ms", lat["proxy_rewrite_first_sight_cold_cpu"]["p95_ms"], lambda v: v < 10, "{:.2f} ms"),
        ("1-bit Hamming over 10k vectors < 0.5 ms (p95)", lat["bq_hamming_top50_10k"]["p95_ms"], lambda v: v < 0.5, "{:.3f} ms"),
        ("Embedding > 2,000 docs/s", lat["embedding_throughput_docs_per_s"], lambda v: v > 2000, "{:,.0f}/s"),
        ("HAR minimal chain exact", 100.0 if h["chain_exact"] else 0.0, lambda v: v == 100, "{:.0f}%"),
        ("Drift recovery 100%", 100 * h["drift"]["recovery_rate"], lambda v: v == 100, "{:.0f}%"),
        ("Self-healing < 3 ms (p95)", h["drift"]["hungarian_latency"]["p95_ms"], lambda v: v < 3, "{:.2f} ms"),
    ]
    return [{"target": t, "measured": fmt.format(v), "value": v, "pass": bool(ok(v))} for t, v, ok, fmt in rows]


def to_markdown(r: dict) -> str:
    s1, s2, s3, s4, s5 = (r["scenarios"][k] for k in ("compression", "kv_cache", "negative_rag", "harvest", "latency"))
    a, n = s1["aegis"], s1["naive_rag"]
    comp = r["compliance"]
    out = ["# aegis-mmu benchmark report\n",
           f"_{r['generated_at']} · {r['machine']} · {r['cpu_count']} CPUs · load avg start {r['load_avg_start'][0]} / end "
           f"{r['load_avg_end'][0]} · tokenizer: {r['tokenizer']} · engine: {r['engine']}_\n",
           f"## Target compliance ({sum(c['pass'] for c in comp)}/{len(comp)} pass)", "",
           "| Target (acceptance rule) | Measured | Pass |", "|---|---|---|"]
    out += [f"| {c['target']} | {c['measured']} | {'✅' if c['pass'] else '❌'} |" for c in comp]
    ch = r["scenarios"]["compression_heldout"]
    lv = r["scenarios"]["levels"]
    out += ["", "## 1 · Token compression & fidelity", "",
           f"Codebase: {s1['files']} files / {s1['units']} units / **{s1['codebase_tokens']:,} tokens** (Python, TypeScript, Kotlin, Swift, Rust); {s1['queries']} bug-fix queries.", "",
           "| Metric | Naive vector RAG (256-tok chunks, same budget) | aegis-mmu |", "|---|---|---|",
           f"| Token reduction | {n['token_reduction_pct']:.1f}% | **{a['token_reduction_pct']:.1f}%** |",
           f"| Target function intact in top-3 (L0 Recall@3 for aegis) | {n['target_whole_in_top3'] * 100:.0f}% | **{a['l0_recall_at_3'] * 100:.0f}%** |",
           f"| Target function intact anywhere in context | {n['target_whole_in_context'] * 100:.0f}% | **{a['target_at_l0_any_rank'] * 100:.0f}%** |",
           f"| Target function lines present (mean, partial credit) | {n['target_lines_covered_mean'] * 100:.0f}% | **{a['target_at_l0_any_rank'] * 100:.0f}%** |",
           f"| Syntax preservation | {n['chunk_syntax_preservation'] * 100:.0f}% of chunks parse | **{a['ast_syntax_preservation'] * 100:.0f}%** of folded files parse |",
           f"| Fold latency incl. syntax re-verification of every output file (warm) | — | p50 {a['fold_latency']['p50_ms']:.2f} ms · p95 {a['fold_latency']['p95_ms']:.2f} ms |", "",
           "| Query | Target | Tokens | Reduction | Target rank | Level |", "|---|---|---|---|---|---|"]
    for q in s1["per_query"]:
        out.append(f"| {q['id']} | `{q['target']}` | {q['tokens_original']:,} → {q['tokens_folded']:,} | {q['reduction_pct']:.1f}% | "
                   f"{q['target_rank'] if q['target_rank'] else '>48'} | {q['target_level'] or 'folded'} |")
    ha = ch["aegis"]
    v1 = r["scenarios"]["compression_heldout_v1_seen"]["aegis"]
    v2s = r["scenarios"]["compression_heldout_v2_seen"]["aegis"]
    v3s = r["scenarios"]["compression_heldout_v3_seen"]["aegis"]
    v4s = r["scenarios"]["compression_heldout_v4_seen"]["aegis"]
    out += ["", f"### 1b · Blind held-out v5 ({ch['queries']} bug reports written after the final codebase growth, blind to the retrieval code "
            "and to earlier results; evaluated once, after defaults were frozen on dev)", "",
            f"Seen sets (evaluated in earlier rounds, their per-query results informed later design, so no longer blind), FastLocal: "
            f"held-out v1 L0 Recall@3 {v1['l0_recall_at_3'] * 100:.0f}% (target at L0 {v1['target_at_l0_any_rank'] * 100:.0f}%); "
            f"held-out v2 {v2s['l0_recall_at_3'] * 100:.0f}% (target at L0 {v2s['target_at_l0_any_rank'] * 100:.0f}%, call-cluster {v2s['cluster_l0_recall_at_3'] * 100:.0f}%); "
            f"held-out v3 {v3s['l0_recall_at_3'] * 100:.0f}% (target at L0 {v3s['target_at_l0_any_rank'] * 100:.0f}%, call-cluster {v3s['cluster_l0_recall_at_3'] * 100:.0f}%); "
            f"held-out v4 {v4s['l0_recall_at_3'] * 100:.0f}% (target at L0 {v4s['target_at_l0_any_rank'] * 100:.0f}%, call-cluster {v4s['cluster_l0_recall_at_3'] * 100:.0f}%).", "",
            "| Metric | Naive vector RAG | aegis-mmu |", "|---|---|---|",
            f"| Token reduction | {ch['naive_rag']['token_reduction_pct']:.1f}% | **{ha['token_reduction_pct']:.1f}%** |",
            f"| Target intact in top-3 (L0 Recall@3, strict single symbol) | {ch['naive_rag']['target_whole_in_top3'] * 100:.0f}% | **{ha['l0_recall_at_3'] * 100:.0f}%** |",
            f"| Call-cluster L0 Recall@3 (target at L0 and top-3 itself or a same-file 1/2-hop callee of a top-3 hit) | — | **{ha['cluster_l0_recall_at_3'] * 100:.0f}%** |",
            f"| Target intact anywhere in context | {ch['naive_rag']['target_whole_in_context'] * 100:.0f}% | **{ha['target_at_l0_any_rank'] * 100:.0f}%** |",
            f"| Syntax preservation | {ch['naive_rag']['chunk_syntax_preservation'] * 100:.0f}% of chunks | **{ha['ast_syntax_preservation'] * 100:.0f}%** |", "",
            "| Query | Target | Reduction | Target rank | Level |", "|---|---|---|---|---|"]
    for q in ch["per_query"]:
        out.append(f"| {q['id']} | `{q['target']}` | {q['reduction_pct']:.1f}% | {q['target_rank'] if q['target_rank'] else '>48'} | {q['target_level'] or 'folded'} |")
    ab_h, ab_d = r["scenarios"]["compression_heldout_requested_cfg"]["aegis"], r["scenarios"]["compression_dev_requested_cfg"]["aegis"]
    npri = r["scenarios"]["compression_heldout_no_priors"]["aegis"]
    out += ["", "Ablation — hierarchical priors (FastLocal; frozen defaults = file prior 0.25 + sibling-caller 0.15, chosen on dev MRR):", "",
            "| Config | Dev L0 Recall@3 | Blind held-out v5 L0 Recall@3 | Blind held-out v5 target at L0 |", "|---|---|---|---|",
            f"| Frozen defaults | {r['scenarios']['compression']['aegis']['l0_recall_at_3'] * 100:.0f}% | {ha['l0_recall_at_3'] * 100:.0f}% | {ha['target_at_l0_any_rank'] * 100:.0f}% |",
            f"| + structural salience 0.08 (requested; lowered dev MRR) | {ab_d['l0_recall_at_3'] * 100:.0f}% | {ab_h['l0_recall_at_3'] * 100:.0f}% | {ab_h['target_at_l0_any_rank'] * 100:.0f}% |",
            f"| no hierarchical priors | — | {npri['l0_recall_at_3'] * 100:.0f}% | {npri['target_at_l0_any_rank'] * 100:.0f}% |",
            f"| + per-file diversity cap (2 full, 0.5x after; requested; lowered dev L0-anywhere 0.80 -> 0.70) | — | "
            f"{r['scenarios']['compression_heldout_divcap']['aegis']['l0_recall_at_3'] * 100:.0f}% | "
            f"{r['scenarios']['compression_heldout_divcap']['aegis']['target_at_l0_any_rank'] * 100:.0f}% |"]
    nz = r["scenarios"].get("neural", {})
    if nz.get("available"):
        out += ["", f"### 1d · Neural Hybrid tier — {nz['model']}", "",
                "Same pipeline (BM25 + concept-BM25 + BQ + MRL-64 + RRF + ColBERT MaxSim + knapsack) with dense vectors from the model "
                "and its full cosine weighted 0.75 in the final score (frozen on dev MRR before the held-out run).", "",
                "| Engine | Dev L0 Recall@3 | Blind held-out v5 L0 Recall@3 | Held-out v5 target at L0 (any rank) | Held-out v1 (seen) L0 Recall@3 | Token reduction |",
                "|---|---|---|---|---|---|",
                f"| FastLocal (0-dep) | {r['scenarios']['compression']['aegis']['l0_recall_at_3'] * 100:.0f}% | {ha['l0_recall_at_3'] * 100:.0f}% | "
                f"{ha['target_at_l0_any_rank'] * 100:.0f}% | {v1['l0_recall_at_3'] * 100:.0f}% | {ha['token_reduction_pct']:.1f}% |",
                f"| Neural Hybrid (nomic-embed-text) | {nz['dev']['aegis']['l0_recall_at_3'] * 100:.0f}% | {nz['heldout_v2']['aegis']['l0_recall_at_3'] * 100:.0f}% | "
                f"{nz['heldout_v2']['aegis']['target_at_l0_any_rank'] * 100:.0f}% | {nz['heldout']['aegis']['l0_recall_at_3'] * 100:.0f}% | "
                f"{nz['heldout_v2']['aegis']['token_reduction_pct']:.1f}% |",
                f"| Neural Hybrid + composite indexing document (ablation) | {nz['dev_composite']['aegis']['l0_recall_at_3'] * 100:.0f}% | "
                f"{nz['heldout_composite']['aegis']['l0_recall_at_3'] * 100:.0f}% | {nz['heldout_composite']['aegis']['target_at_l0_any_rank'] * 100:.0f}% | — | "
                f"{nz['heldout_composite']['aegis']['token_reduction_pct']:.1f}% |", "",
                "| Held-out query | Target | FastLocal rank | Neural rank |", "|---|---|---|---|"]
        for a_, b_ in zip(ch["per_query"], nz["heldout_v2"]["per_query"]):
            out.append(f"| {a_['id']} | `{a_['target']}` | {a_['target_rank'] or '>48'} | {b_['target_rank'] or '>48'} |")
        out += ["", f"Neural model-call latency (informational; the < 10 ms rows are measured on the 0-dep engine): retrieval with a fresh, "
                f"uncached query p50 {nz['retrieval_fresh_query']['p50_ms']:.1f} ms / p95 {nz['retrieval_fresh_query']['p95_ms']:.1f} ms; "
                f"uncached cold file ingest p50 {nz['file_ingest_uncached']['p50_ms']:.0f} ms / p95 {nz['file_ingest_uncached']['p95_ms']:.0f} ms. "
                "Repeated texts hit the blake2b content-hash embedding cache."]
    out += ["", "### 1c · Per-level mipmap size", "",
            "| Codebase | Tokens | Functions (median tokens) | L1 / L0 | L2 / file | L3 / file |", "|---|---|---|---|---|---|"]
    for name, v in lv.items():
        out.append(f"| `{name}` | {v['tokens']:,} | {v['functions']} ({v['function_tokens_median']:.0f}) | {v['l1_over_l0'] * 100:.1f}% | "
                   f"{v['l2_over_file'] * 100:.1f}% | {v['l3_over_file'] * 100:.2f}% |")
    c, b = s2["conversation"], s2["context_block"]
    out += ["", f"## 2 · KV-cache prefix hit rate ({s2['turns']}-turn agent session)", "",
            "`KV_Hit_Rate = LCP_tokens(Prompt_{t-1}, Prompt_t) / Tokens(Prompt_t)`, averaged over turns 2–5.", "",
            "| Setting | Standard RAG (re-ranked) | aegis-mmu (append-only paging) |", "|---|---|---|",
            f"| Context block only (spec metric; stable-prefix policy) | {b['standard_rag']['kv_hit_rate_avg_t2_5'] * 100:.1f}% | **{b['aegis_pager']['kv_hit_rate_avg_t2_5'] * 100:.1f}%** |",
            f"| Context block, adaptive policy (ablation) | — | {b['aegis_pager_adaptive']['kv_hit_rate_avg_t2_5'] * 100:.1f}% |",
            f"| Full conversation prompt through the proxy | {c['standard_rag']['kv_hit_rate_avg_t2_5'] * 100:.1f}% | **{c['aegis']['kv_hit_rate_avg_t2_5'] * 100:.1f}%** |",
            f"| Previous prompt reused as prefix (retention) | {c['standard_rag']['prefix_retention_avg_t2_5'] * 100:.1f}% | **{c['aegis']['prefix_retention_avg_t2_5'] * 100:.1f}%** |", "",
            "Context block per turn — stable policy: hit " + " → ".join(f"{x * 100:.1f}%" for x in b["aegis_pager"]["kv_hit_rate_per_turn"])
            + " · tokens " + " → ".join(f"{x:,}" for x in b["aegis_pager"]["context_tokens_per_turn"]),
            "", "Context block per turn — adaptive policy: hit " + " → ".join(f"{x * 100:.1f}%" for x in b["aegis_pager_adaptive"]["kv_hit_rate_per_turn"])
            + " · tokens " + " → ".join(f"{x:,}" for x in b["aegis_pager_adaptive"]["context_tokens_per_turn"]), "",
            "Per turn (full conversation, aegis): hit " + " → ".join(f"{x * 100:.1f}%" for x in c["aegis"]["kv_hit_rate_per_turn"]) +
            " · append-only optimum " + " → ".join(f"{x * 100:.1f}%" for x in c["aegis"]["append_only_optimum_per_turn"]),
            "", "Per turn (full conversation, standard RAG): hit " + " → ".join(f"{x * 100:.1f}%" for x in c["standard_rag"]["kv_hit_rate_per_turn"]), "",
            "## 3 · Negative RAG regression prevention", "",
            f"{s3['memories']} recorded anti-patterns (race condition, memory leak, deprecated API, SQL injection, N+1), {s3['probes']} probes (10 positive, 10 negative). "
            f"Threshold {s3['threshold']} — self-calibrated: {s3['calibration']['source'] if s3.get('calibration') else 'explicit'}"
            + (f", null mean {s3['calibration']['null_mean']} + {s3['calibration']['z']:.0f} × std {s3['calibration']['null_std']}." if s3.get('calibration') and s3['calibration'].get('null_mean') is not None else "."), "",
            "| Precision | Recall | Pattern attribution | ROC AUC | Check latency p50 |", "|---|---|---|---|---|",
            f"| {s3['precision'] * 100:.0f}% | {s3['recall'] * 100:.0f}% | {s3['pattern_attribution_accuracy'] * 100:.0f}% | {s3['roc_auc']:.3f} | {s3['check_latency']['p50_ms']:.3f} ms |", "",
            "## 4 · Web-to-MCP noise pruning & schema self-healing", "",
            "| Metric | Result |", "|---|---|",
            f"| HAR requests (static {s4['categories'].get('static', 0)}, trackers {s4['categories'].get('tracker', 0)}, preflight {s4['categories'].get('preflight', 0)}, api {s4['categories'].get('api', 0)}, doc {s4['categories'].get('document', 0)}) | {s4['har_requests']} |",
            f"| Minimal chain isolated | {s4['minimal_chain']} (expected {s4['expected_chain']}) → **{'exact' if s4['chain_exact'] else 'MISMATCH'}** |",
            f"| Noise pruned | **{s4['noise_pruned_pct']:.1f}%** in {s4['slice_latency']['p50_ms']:.2f} ms (p50) |",
            f"| Drifted fields recovered (fresh v2 values, {s4['drift']['renamed_fields']} renamed + relocated, {s4['drift']['moved_fields']} moved) | **{s4['drift']['fields_recovered']}/{s4['drift']['contract_fields']}** ({s4['drift']['recovery_rate'] * 100:.0f}%) |",
            f"| Hungarian healing latency | p50 {s4['drift']['hungarian_latency']['p50_ms']:.2f} ms · p95 {s4['drift']['hungarian_latency']['p95_ms']:.2f} ms |",
            f"| Learned-alias fast path | p50 {s4['drift']['alias_fastpath_latency']['p50_ms']:.3f} ms |",
            f"| Live replay vs server with fresh CSRF/JWT + v2 schema | statuses {s4['live_replay']['steps']}, healed={s4['live_replay']['healed']}, values correct={s4['live_replay']['values_correct']} |", "",
            "## 5 · Latency profile", "", "| Operation | p50 | p95 | p99 |", "|---|---|---|---|"]
    labels = {"bq_hamming_top50_10k": "1-bit Hamming top-50 over 10k vectors", "mrl64_cosine_top50_10k": "Matryoshka-64 cosine top-50 over 10k",
              "fts5_bm25_codebase": "SQLite FTS5 BM25 (codebase)", "colbert_maxsim_rerank_48": "ColBERT MaxSim rerank (48 candidates)",
              "hybrid_retrieval_end_to_end": "Hybrid retrieval end-to-end (RRF+graph+MaxSim)", "ast_mipmap_per_file_cold": "AST mipmapping, per file (cold)",
              "ast_mipmap_largest_file": "AST mipmapping, largest file", "fold_pipeline_warm_16k_tokens": f"Full fold pipeline, warm ({s1['codebase_tokens']:,}-token codebase)",
              "proxy_rewrite_turn_warm": "Proxy request rewrite per turn (warm)",
              "proxy_rewrite_first_sight_cold": f"Proxy rewrite, first sight of a never-seen file (cold, on-disk WAL, ~{s5['cold_file_tokens_mean']:,.0f}-token files)",
              "file_ingest_cold": "Cold file ingest (parse + mipmap + embed + persist)",
              "proxy_rewrite_first_sight_cold_cpu": "↳ same, process CPU time (inflates under contention too: E-core migration on Apple silicon)",
              "file_ingest_cold_cpu": "↳ cold file ingest, process CPU time",
              "hungarian_self_heal": "Hungarian self-healing (11×17 matrix)"}
    for k, label in labels.items():
        v = s5[k]
        out.append(f"| {label} | {v['p50_ms']:.3f} ms | {v['p95_ms']:.3f} ms | {v['p99_ms']:.3f} ms |")
    out.append(f"\nEmbedding throughput (FastLocalEngine, single core): **{s5['embedding_throughput_docs_per_s']:,.0f} code units/s** "
               f"(mean {s5['embedding_doc_tokens_mean']:.0f} tokens/unit); {s5['embedding_throughput_whole_files_per_s']:,.0f} whole files/s "
               f"(~1.2k tokens/file).\n")
    return "\n".join(out)


def run() -> dict:
    files = load_codebase()
    queries = json.loads((DS / "queries.json").read_text())
    session = json.loads((DS / "session.json").read_text())
    count_tokens("warm the BPE encoder before any timer starts")
    from aegis_mmu.tokens import line_prefix

    line_prefix("x = 1\n")
    t0 = time.perf_counter()
    report = {
        "aegis_mmu_version": __version__,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "machine": f"{platform.system()} {platform.machine()} · Python {platform.python_version()} · numpy {np.__version__}",
        "tokenizer": tokenizer_name(),
        "cpu_count": __import__("os").cpu_count(),
        "load_avg_start": [round(x, 2) for x in __import__("os").getloadavg()],
        "engine": FastLocalEngine().name,
        "scenarios": {
            "compression": scenario_compression(files, queries),
            "compression_heldout": scenario_compression(files, json.loads((DS / "queries_heldout_v5.json").read_text())),
            "compression_heldout_v4_seen": scenario_compression(files, json.loads((DS / "queries_heldout_v4.json").read_text())),
            "compression_heldout_v3_seen": scenario_compression(files, json.loads((DS / "queries_heldout_v3.json").read_text())),
            "compression_heldout_v2_seen": scenario_compression(files, json.loads((DS / "queries_heldout_v2.json").read_text())),
            "compression_heldout_v1_seen": scenario_compression(files, json.loads((DS / "queries_heldout.json").read_text())),
            "compression_heldout_requested_cfg": scenario_compression(
                files, json.loads((DS / "queries_heldout_v5.json").read_text()), salience_weight=0.08),
            "compression_dev_requested_cfg": scenario_compression(files, queries, salience_weight=0.08),
            "compression_heldout_no_priors": scenario_compression(
                files, json.loads((DS / "queries_heldout_v5.json").read_text()), file_prior_weight=0.0, sibling_caller_weight=0.0),
            "compression_heldout_divcap": scenario_compression(
                files, json.loads((DS / "queries_heldout_v5.json").read_text()), prior_full_per_file=2, prior_decay=0.5),
            **neural_scenarios(files, queries),
            "levels": {"codebase": scenario_levels(files), "codebase_v2": scenario_levels(load_codebase("codebase_v2")),
                       "codebase_v1": scenario_levels(load_codebase("codebase_v1"))},
            "kv_cache": scenario_kv(files, session),
            "negative_rag": scenario_negative(),
            "harvest": scenario_harvest(),
            "latency": scenario_latency(files, queries),
        },
    }
    report["load_avg_end"] = [round(x, 2) for x in __import__("os").getloadavg()]
    report["compliance"] = compliance(report)
    report["suite_seconds"] = round(time.perf_counter() - t0, 2)
    return report


def main(out: str | None = None) -> dict:
    import os

    max_load = os.environ.get("AEGIS_BENCH_MAX_LOAD")
    if max_load:
        print(f"waiting for load < {max_load} ... now {wait_for_quiet(float(max_load))}")
    report = run()
    path = Path(out) if out else ROOT / "evals" / "latest_report.json"
    path.write_text(json.dumps(report, indent=2))
    md = to_markdown(report)
    path.with_suffix(".md").write_text(md + "\n")
    print(md)
    print(f"\nwrote {path} and {path.with_suffix('.md')} in {report['suite_seconds']} s")
    if out is None:
        from evals.update_readme import main as update_readme

        update_readme()
    return report


if __name__ == "__main__":
    if "--neural-only" in sys.argv:
        files_ = load_codebase()
        res = neural_scenarios(files_, json.loads((DS / "queries.json").read_text()))
        print(json.dumps({k: v for k, v in res["neural"].items() if k in ("available", "model")}),
              f"-> cached in {NEURAL_RESULTS}")
    else:
        main()
