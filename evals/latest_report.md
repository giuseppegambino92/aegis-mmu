# aegis-mmu benchmark report

_2026-11-14 00:27 UTC · Darwin arm64 · Python 3.14.10 · numpy 2.5.4 · 14 CPUs · load avg start 7.91 / end 7.16 · tokenizer: tiktoken/cl100k_base · engine: fast-local-srp-512_

## Target compliance (14/24 pass)

| Target (acceptance rule) | Measured | Pass |
|---|---|---|
| L0 Recall@3 >= 90% — blind held-out v5, Neural Hybrid (nomic-embed-text) | 67% | ❌ |
| Call-cluster L0 Recall@3 >= 90% — blind held-out v5, Neural Hybrid | 80% | ❌ |
| L0 Recall@3 >= 90% — dev, Neural Hybrid (nomic-embed-text) | 70% | ❌ |
| Token reduction > 75% | 89.1% | ✅ |
| L0 Recall@3 >= 90% — blind held-out v5, FastLocal 0-dep | 60% | ❌ |
| Call-cluster L0 Recall@3 >= 90% — blind held-out v5, FastLocal 0-dep | 67% | ❌ |
| Target at full L0 anywhere in context >= 90% — blind held-out v5, FastLocal (not a top-3 metric) | 87% | ❌ |
| L0 Recall@3 >= 90% — dev, FastLocal 0-dep | 50% | ❌ |
| AST syntax preservation = 100% | 100% | ✅ |
| L1 ~25% of L0 (accept <= 30%) | 22.9% | ✅ |
| L2 ~6% of file (accept <= 8%) | 7.9% | ✅ |
| L3 ~1-2% of file (accept <= 2%) | 0.66% | ✅ |
| KV prefix hit > 85% (context block, stable-prefix policy) | 89.5% | ✅ |
| Standard RAG prefix hit < 25% (baseline sanity) | 0.1% | ✅ |
| Warm fold pipeline p95 < 10 ms | 9.07 ms | ✅ |
| Warm proxy rewrite p95 < 10 ms | 6.21 ms | ✅ |
| Cold first-sight proxy rewrite p95 < 10 ms | 11.62 ms | ❌ |
| Cold per-file ingest p95 < 10 ms | 16.61 ms | ❌ |
| Cold first-sight rewrite, CPU-time p95 < 10 ms | 11.62 ms | ❌ |
| 1-bit Hamming over 10k vectors < 0.5 ms (p95) | 0.149 ms | ✅ |
| Embedding > 2,000 docs/s | 14,603/s | ✅ |
| HAR minimal chain exact | 100% | ✅ |
| Drift recovery 100% | 100% | ✅ |
| Self-healing < 3 ms (p95) | 1.44 ms | ✅ |

## 1 · Token compression & fidelity

Codebase: 15 files / 363 units / **56,387 tokens** (Python, TypeScript, Kotlin, Swift, Rust); 10 bug-fix queries.

| Metric | Naive vector RAG (256-tok chunks, same budget) | aegis-mmu |
|---|---|---|
| Token reduction | 89.1% | **89.1%** |
| Target function intact in top-3 (L0 Recall@3 for aegis) | 10% | **50%** |
| Target function intact anywhere in context | 10% | **80%** |
| Target function lines present (mean, partial credit) | 80% | **80%** |
| Syntax preservation | 4% of chunks parse | **100%** of folded files parse |
| Fold latency incl. syntax re-verification of every output file (warm) | — | p50 9.64 ms · p95 19.95 ms |

| Query | Target | Tokens | Reduction | Target rank | Level |
|---|---|---|---|---|---|
| q01 | `refunds.py::RefundCalculator.compute_refund` | 56,387 → 6,193 | 89.0% | 1 | L0 |
| q02 | `ledger.py::Ledger.balance_as_of` | 56,387 → 6,166 | 89.1% | 4 | L0 |
| q03 | `service.py::OrderService.cancel_order` | 56,387 → 5,718 | 89.9% | 5 | L0 |
| q04 | `stock.py::StockLedger.reserve` | 56,387 → 6,047 | 89.3% | 8 | L0 |
| q05 | `tokens.py::TokenService.verify` | 56,387 → 5,725 | 89.8% | 2 | L0 |
| q06 | `rate_limit.py::RateLimiter._bucket_key` | 56,387 → 6,218 | 89.0% | >48 | L1 |
| q07 | `cache.py::make_key` | 56,387 → 6,343 | 88.8% | >48 | folded |
| q08 | `cartStore.ts::CartStore.applyCoupon` | 56,387 → 6,370 | 88.7% | 1 | L0 |
| q09 | `discount.rs::DiscountEngine.apply_tiered` | 56,387 → 6,571 | 88.3% | 1 | L0 |
| q10 | `SessionManager.swift::SessionManager.needsRefresh` | 56,387 → 5,875 | 89.6% | 1 | L0 |

### 1b · Blind held-out v5 (15 bug reports written after the final codebase growth, blind to the retrieval code and to earlier results; evaluated once, after defaults were frozen on dev)

Seen sets (evaluated in earlier rounds, their per-query results informed later design, so no longer blind), FastLocal: held-out v1 L0 Recall@3 53% (target at L0 73%); held-out v2 73% (target at L0 100%, call-cluster 73%); held-out v3 73% (target at L0 100%, call-cluster 87%); held-out v4 73% (target at L0 87%, call-cluster 73%).

| Metric | Naive vector RAG | aegis-mmu |
|---|---|---|
| Token reduction | 88.9% | **88.9%** |
| Target intact in top-3 (L0 Recall@3, strict single symbol) | 13% | **60%** |
| Call-cluster L0 Recall@3 (target at L0 and top-3 itself or a same-file 1/2-hop callee of a top-3 hit) | — | **67%** |
| Target intact anywhere in context | 27% | **87%** |
| Syntax preservation | 9% of chunks | **100%** |

| Query | Target | Reduction | Target rank | Level |
|---|---|---|---|---|
| h5_01 | `currency.py::sum_money` | 88.8% | 7 | L0 |
| h5_02 | `ledger.py::Ledger._entries_between` | 89.0% | 1 | L0 |
| h5_03 | `refunds.py::RefundPolicy.refunds_shipping` | 89.1% | 1 | L0 |
| h5_04 | `cache.py::CacheStats.hit_rate` | 89.0% | 1 | L0 |
| h5_05 | `tokens.py::_b64encode` | 89.3% | >48 | folded |
| h5_06 | `rate_limit.py::RateLimiter.peek` | 89.1% | 1 | L0 |
| h5_07 | `stock.py::StockLedger.release` | 89.1% | 1 | L0 |
| h5_08 | `repository.py::OrderRepository.update_status` | 89.0% | >48 | L1 |
| h5_09 | `service.py::_validate_items` | 88.9% | 1 | L0 |
| h5_10 | `email.py::RefundEmailComposer.render` | 88.7% | 4 | L0 |
| h5_11 | `client.ts::parseRetryAfter` | 88.6% | 1 | L0 |
| h5_12 | `cartStore.ts::describeError` | 88.8% | 4 | L0 |
| h5_13 | `SessionManager.swift::SessionManager.performRefresh` | 88.8% | 1 | L0 |
| h5_14 | `discount.rs::DiscountEngine.new` | 88.6% | 5 | L0 |
| h5_15 | `CheckoutViewModel.kt::CheckoutViewModel.retry` | 89.0% | 2 | L0 |

Ablation — hierarchical priors (FastLocal; frozen defaults = file prior 0.25 + sibling-caller 0.15, chosen on dev MRR):

| Config | Dev L0 Recall@3 | Blind held-out v5 L0 Recall@3 | Blind held-out v5 target at L0 |
|---|---|---|---|
| Frozen defaults | 50% | 60% | 87% |
| + structural salience 0.08 (requested; lowered dev MRR) | 50% | 60% | 87% |
| no hierarchical priors | — | 67% | 87% |
| + per-file diversity cap (2 full, 0.5x after; requested; lowered dev L0-anywhere 0.80 -> 0.70) | — | 67% | 87% |

### 1d · Neural Hybrid tier — nomic-embed-text (Ollama, 768-d truncated to 512)

Same pipeline (BM25 + concept-BM25 + BQ + MRL-64 + RRF + ColBERT MaxSim + knapsack) with dense vectors from the model and its full cosine weighted 0.75 in the final score (frozen on dev MRR before the held-out run).

| Engine | Dev L0 Recall@3 | Blind held-out v5 L0 Recall@3 | Held-out v5 target at L0 (any rank) | Held-out v1 (seen) L0 Recall@3 | Token reduction |
|---|---|---|---|---|---|
| FastLocal (0-dep) | 50% | 60% | 87% | 53% | 88.9% |
| Neural Hybrid (nomic-embed-text) | 70% | 67% | 93% | 40% | 89.0% |
| Neural Hybrid + composite indexing document (ablation) | 60% | 67% | 87% | — | 89.0% |

| Held-out query | Target | FastLocal rank | Neural rank |
|---|---|---|---|
| h5_01 | `currency.py::sum_money` | 7 | 8 |
| h5_02 | `ledger.py::Ledger._entries_between` | 1 | 2 |
| h5_03 | `refunds.py::RefundPolicy.refunds_shipping` | 1 | 1 |
| h5_04 | `cache.py::CacheStats.hit_rate` | 1 | 1 |
| h5_05 | `tokens.py::_b64encode` | >48 | >48 |
| h5_06 | `rate_limit.py::RateLimiter.peek` | 1 | 1 |
| h5_07 | `stock.py::StockLedger.release` | 1 | 2 |
| h5_08 | `repository.py::OrderRepository.update_status` | >48 | >48 |
| h5_09 | `service.py::_validate_items` | 1 | 3 |
| h5_10 | `email.py::RefundEmailComposer.render` | 4 | 3 |
| h5_11 | `client.ts::parseRetryAfter` | 1 | 2 |
| h5_12 | `cartStore.ts::describeError` | 4 | >48 |
| h5_13 | `SessionManager.swift::SessionManager.performRefresh` | 1 | 1 |
| h5_14 | `discount.rs::DiscountEngine.new` | 5 | 5 |
| h5_15 | `CheckoutViewModel.kt::CheckoutViewModel.retry` | 2 | 1 |

Neural model-call latency (informational; the < 10 ms rows are measured on the 0-dep engine): retrieval with a fresh, uncached query p50 39.1 ms / p95 45.6 ms; uncached cold file ingest p50 112 ms / p95 155 ms. Repeated texts hit the blake2b content-hash embedding cache.

### 1c · Per-level mipmap size

| Codebase | Tokens | Functions (median tokens) | L1 / L0 | L2 / file | L3 / file |
|---|---|---|---|---|---|
| `codebase` | 56,387 | 253 (181) | 22.9% | 7.9% | 0.66% |
| `codebase_v2` | 36,206 | 253 (79) | 32.8% | 12.3% | 1.01% |
| `codebase_v1` | 14,393 | 128 (56) | 45.6% | 18.3% | 2.40% |

## 2 · KV-cache prefix hit rate (5-turn agent session)

`KV_Hit_Rate = LCP_tokens(Prompt_{t-1}, Prompt_t) / Tokens(Prompt_t)`, averaged over turns 2–5.

| Setting | Standard RAG (re-ranked) | aegis-mmu (append-only paging) |
|---|---|---|
| Context block only (spec metric; stable-prefix policy) | 0.1% | **89.5%** |
| Context block, adaptive policy (ablation) | — | 83.8% |
| Full conversation prompt through the proxy | 0.1% | **69.0%** |
| Previous prompt reused as prefix (retention) | 0.2% | **100.0%** |

Context block per turn — stable policy: hit 0.0% → 91.5% → 87.0% → 87.1% → 92.6% · tokens 6,736 → 7,358 → 8,452 → 9,704 → 10,472

Context block per turn — adaptive policy: hit 0.0% → 87.1% → 75.6% → 86.2% → 86.4% · tokens 6,736 → 7,732 → 10,219 → 11,852 → 10,202

Per turn (full conversation, aegis): hit 0.0% → 36.6% → 74.3% → 82.8% → 82.4% · append-only optimum 0.0% → 36.6% → 74.3% → 82.8% → 82.4%

Per turn (full conversation, standard RAG): hit 0.0% → 0.2% → 0.1% → 0.1% → 0.1%

## 3 · Negative RAG regression prevention

5 recorded anti-patterns (race condition, memory leak, deprecated API, SQL injection, N+1), 20 probes (10 positive, 10 negative). Threshold 0.377 — self-calibrated: calibrated, null mean 0.0398 + 5 × std 0.0674.

| Precision | Recall | Pattern attribution | ROC AUC | Check latency p50 |
|---|---|---|---|---|
| 100% | 90% | 100% | 0.970 | 0.131 ms |

## 4 · Web-to-MCP noise pruning & schema self-healing

| Metric | Result |
|---|---|
| HAR requests (static 13, trackers 12, preflight 3, api 11, doc 1) | 40 |
| Minimal chain isolated | [10, 14, 20] (expected [10, 14, 20]) → **exact** |
| Noise pruned | **92.5%** in 0.90 ms (p50) |
| Drifted fields recovered (fresh v2 values, 8 renamed + relocated, 9 moved) | **11/11** (100%) |
| Hungarian healing latency | p50 1.07 ms · p95 1.44 ms |
| Learned-alias fast path | p50 0.118 ms |
| Live replay vs server with fresh CSRF/JWT + v2 schema | statuses [200, 201, 200], healed=True, values correct=True |

## 5 · Latency profile

| Operation | p50 | p95 | p99 |
|---|---|---|---|
| 1-bit Hamming top-50 over 10k vectors | 0.133 ms | 0.149 ms | 0.159 ms |
| Matryoshka-64 cosine top-50 over 10k | 0.060 ms | 0.068 ms | 0.106 ms |
| SQLite FTS5 BM25 (codebase) | 1.523 ms | 1.888 ms | 2.017 ms |
| ColBERT MaxSim rerank (48 candidates) | 0.080 ms | 0.106 ms | 0.165 ms |
| Hybrid retrieval end-to-end (RRF+graph+MaxSim) | 5.540 ms | 6.547 ms | 7.211 ms |
| AST mipmapping, per file (cold) | 4.119 ms | 5.349 ms | 5.743 ms |
| AST mipmapping, largest file | 5.508 ms | 6.325 ms | 6.437 ms |
| Full fold pipeline, warm (56,387-token codebase) | 7.858 ms | 9.069 ms | 9.782 ms |
| Proxy request rewrite per turn (warm) | 5.247 ms | 6.214 ms | 6.473 ms |
| Proxy rewrite, first sight of a never-seen file (cold, on-disk WAL, ~3,759-token files) | 9.739 ms | 11.623 ms | 12.211 ms |
| Cold file ingest (parse + mipmap + embed + persist) | 4.503 ms | 16.609 ms | 18.619 ms |
| ↳ same, process CPU time (inflates under contention too: E-core migration on Apple silicon) | 9.753 ms | 11.625 ms | 12.408 ms |
| ↳ cold file ingest, process CPU time | 4.675 ms | 17.602 ms | 21.816 ms |
| Hungarian self-healing (11×17 matrix) | 1.065 ms | 1.337 ms | 1.521 ms |

Embedding throughput (FastLocalEngine, single core): **14,603 code units/s** (mean 173 tokens/unit); 984 whole files/s (~1.2k tokens/file).

