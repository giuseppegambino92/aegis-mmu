import time

import numpy as np
import pytest

from aegis_mmu.embeddings.colbert import batch_maxsim, decode_token_matrix, encode_token_matrix, maxsim
from aegis_mmu.embeddings.matryoshka import (binary_quantize, cosine_topk, hamming_distances, hamming_topk,
                                             matryoshka_pyramid, mrl_slice, popcount, rrf_fuse)
from aegis_mmu.embeddings.providers import FastLocalEngine


@pytest.fixture(scope="module")
def engine():
    return FastLocalEngine()


def test_mrl_slice_renormalises_every_prefix():
    v = np.random.default_rng(0).standard_normal((3, 512)).astype(np.float32)
    for k, s in matryoshka_pyramid(v).items():
        assert s.shape == (3, k)
        np.testing.assert_allclose(np.linalg.norm(s, axis=1), 1.0, atol=1e-5)
        expected = v[:, :k] / (np.linalg.norm(v[:, :k], axis=1, keepdims=True) + 1e-9)
        np.testing.assert_allclose(s, expected, atol=1e-6)
    with pytest.raises(ValueError):
        mrl_slice(v, 1024)


def test_binary_quantization_packs_sign_bits():
    v = np.array([[0.5, -1, 2, 0, -0.1, 3, 4, -5, 1, 1, 1, 1, -1, -1, -1, -1]], dtype=np.float32)
    b = binary_quantize(v)
    assert b.dtype == np.uint8 and b.shape == (1, 2)
    assert b[0, 0] == 0b10100110 and b[0, 1] == 0b11110000


def test_hamming_matches_bruteforce_and_lut():
    rng = np.random.default_rng(1)
    db = binary_quantize(rng.standard_normal((257, 512)))
    q = binary_quantize(rng.standard_normal(512))
    brute = np.array([np.unpackbits(np.bitwise_xor(row, q)).sum() for row in db])
    np.testing.assert_array_equal(hamming_distances(q, db), brute)
    x = np.bitwise_xor(db[:4], q)
    assert popcount(x).sum() == brute[:4].sum()
    idx, dist = hamming_topk(q, db, 10)
    assert list(dist) == sorted(brute)[:10]
    assert list(idx) == list(np.lexsort((np.arange(257), brute))[:10])


def test_hamming_10k_under_half_millisecond():
    rng = np.random.default_rng(2)
    db = binary_quantize(rng.standard_normal((10_000, 512)).astype(np.float32))
    q = binary_quantize(rng.standard_normal(512))
    for _ in range(5):
        hamming_topk(q, db, 50)
    meds = []
    for _ in range(3):  # best of 3 medians
        lat = []
        for _ in range(50):
            t = time.perf_counter()
            hamming_topk(q, db, 50)
            lat.append((time.perf_counter() - t) * 1e3)
        meds.append(float(np.median(lat)))
    assert min(meds) < 0.5, f"median {min(meds):.3f} ms"


def test_rrf_weighted_fusion():
    fused = rrf_fuse({"a": ["x", "y", "z"], "b": ["y", "x"], "c": ["z"]}, {"a": 1.0, "b": 1.0, "c": 0.0}, k=60)
    scores = dict(fused)
    assert fused[0][0] in ("x", "y")
    assert scores["x"] == pytest.approx(1 / 61 + 1 / 62)
    assert scores["y"] == pytest.approx(1 / 62 + 1 / 61)
    assert scores["z"] == pytest.approx(1 / 63)


def test_colbert_maxsim_batch_equals_reference():
    rng = np.random.default_rng(3)

    def unit(n, d=16):
        m = rng.standard_normal((n, d)).astype(np.float32)
        return m / np.linalg.norm(m, axis=1, keepdims=True)

    q = unit(5)
    docs = [unit(7), unit(1), np.zeros((0, 16), np.float32), unit(12)]
    ref = [maxsim(q, d) / 5 for d in docs]
    np.testing.assert_allclose(batch_maxsim(q, docs), ref, atol=1e-5)
    blob = encode_token_matrix(docs[0])
    np.testing.assert_allclose(decode_token_matrix(blob, 16), docs[0], atol=2e-3)


def test_engine_is_deterministic_and_semantic(engine):
    a = engine.embed(["def compute_refund(order): return round(order.total * rate)"])[0]
    b = FastLocalEngine().embed(["def compute_refund(order): return round(order.total * rate)"])[0]
    np.testing.assert_array_equal(a, b)
    q = engine.embed_query("refund rounding for an order total")
    unrelated = engine.embed_query("render the dark mode toggle")
    assert float(a @ q) > float(unrelated @ q) + 0.2
    # concept lexicon: price ~ amount, origin ~ from
    assert float(engine.embed(["price_eur"])[0] @ engine.embed(["fare.amount_eur"])[0]) > 0.4
    assert float(engine.embed(["origin"], use_stopwords=False)[0] @ engine.embed(["from_airport"], use_stopwords=False)[0]) > 0.1


def test_matryoshka_prefix_preserves_ranking(engine):
    docs = ["refund calculator rounding cents", "jwt token verification expiry", "stock reservation warehouse",
            "email template notification", "rate limiter bucket key"]
    D = engine.embed(docs)
    q = engine.embed_query("token expiry verification")
    full = cosine_topk(q, D, 1)[0][0]
    pref = cosine_topk(mrl_slice(q, 64), mrl_slice(D, 64), 1)[0][0]
    assert full == pref == 1


def test_engine_throughput_over_2000_docs_per_second(engine):
    docs = [f"def handler_{i}(request, user_id: int) -> dict:\n    order = repo.load(user_id)\n    return {{'total': order.total_cents + {i}}}\n" * 4
            for i in range(3000)]
    engine.embed(docs[:50])
    rates = []
    for _ in range(3):  # best of 3: robust to OS scheduling spikes on a busy machine
        t = time.perf_counter()
        engine.embed(docs)
        rates.append(len(docs) / (time.perf_counter() - t))
    rate = max(rates)
    assert rate > 2000, f"{rate:.0f} docs/s"


def test_token_vectors_normalised(engine):
    t = engine.embed_tokens("refund the order total in cents", is_query=True)
    assert t.shape[1] == engine.token_dim
    np.testing.assert_allclose(np.linalg.norm(t, axis=1), 1.0, atol=1e-5)
