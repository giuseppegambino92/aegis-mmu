"""Model-backed tier: content-hash embedding cache, Ollama adapter (skipped if not running),
composite indexing document and the dense-cosine term of the final score."""

import numpy as np
import pytest

from aegis_mmu.config import AegisSettings
from aegis_mmu.db.store import AegisStore
from aegis_mmu.embeddings.providers import EmbeddingCache, OllamaEngine
from aegis_mmu.mmu.pipeline import SemanticMMU

GATE = '''
class Gatekeeper:
    def admit(self, tenant: str, now: float) -> bool:
        slot = self._slot(tenant, now)
        return self.counters.incr(slot) <= self.allowance

    def _slot(self, tenant: str, now: float) -> str:
        return f"{tenant}:{int(now // 60)}"
'''


def test_embedding_cache_roundtrip(tmp_path):
    path = str(tmp_path / "cache.db")
    c = EmbeddingCache(path)
    k1, k2 = EmbeddingCache.key("m", "document", 4, "a"), EmbeddingCache.key("m", "query", 4, "a")
    assert k1 != k2  # task mode is part of the key
    c.put_many({k1: np.arange(4, dtype=np.float32)})
    fresh = EmbeddingCache(path)  # new process view: served from SQLite
    got = fresh.get_many([k1, k2])
    np.testing.assert_array_equal(got[k1], np.arange(4, dtype=np.float32)) and None
    assert k2 not in got


_ollama = OllamaEngine(cache_path=None)


@pytest.mark.skipif(not _ollama.available(), reason="Ollama with nomic-embed-text not running")
def test_ollama_engine_live_cache_and_task_prefixes(tmp_path):
    e = OllamaEngine(cache_path=str(tmp_path / "c.db"))
    v = e.embed(["def refund(order): ...", "class Ledger: ..."])
    assert v.shape == (2, 512) and np.allclose(np.linalg.norm(v, axis=1), 1, atol=1e-4)
    calls = e.calls
    e.embed(["def refund(order): ...", "class Ledger: ..."])
    assert e.calls == calls  # second call fully served by the content-hash cache
    q = e.embed_query("def refund(order): ...")
    assert float(q @ v[0]) < 0.9999  # search_query: vs search_document: prefixes give different vectors


def test_composite_index_carries_one_hop_names_into_fts(settings):
    s = settings.model_copy(update={"composite_index": True})
    m = SemanticMMU(AegisStore(":memory:", settings=s), settings=s)
    m.ingest_many([("gate.py", GATE)])
    slot = m.store.find_units("gate.py", "Gatekeeper._slot")[0].vec_id
    hits = [v for v, _ in m.store.bm25("admit", 10)]
    assert slot in hits  # `_slot` never says "admit"; its composite document lists called_by: Gatekeeper.admit
    plain = SemanticMMU(AegisStore(":memory:", settings=settings), settings=settings)
    plain.ingest_many([("gate.py", GATE)])
    assert plain.store.find_units("gate.py", "Gatekeeper._slot")[0].vec_id not in [v for v, _ in plain.store.bm25("admit", 10)]


def test_dense_weight_one_ranks_by_dense_cosine(settings):
    s = settings.model_copy(update={"dense_weight": 1.0, "graph_alpha": 0.0, "callee_activation": False,
                                    "file_prior_weight": 0.0, "sibling_caller_weight": 0.0, "salience_weight": 0.0})
    m = SemanticMMU(AegisStore(":memory:", settings=s), settings=s)
    m.ingest_many([("gate.py", GATE), ("ui.py", "def banner(title):\n    return title.upper()\n")])
    q = "tenant admission counter"
    hits = m.retriever.search(q).hits
    qv = m.engine.embed_query(q)
    dense = {h.vec_id: float(m.store.dense([h.vec_id])[0] @ qv) for h in hits}
    assert [h.vec_id for h in hits] == sorted(dense, key=lambda v: (-dense[v], v))
