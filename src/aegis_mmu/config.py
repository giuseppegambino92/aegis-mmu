"""Runtime configuration (environment variables prefixed with ``AEGIS_``)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ProxyMode = Literal["fold-local", "passthrough", "upstream-forward"]


def _default_home() -> Path:
    return Path.home() / ".aegis-mmu"


class RRFWeights(BaseSettings):
    """Per-ranker weights for weighted Reciprocal Rank Fusion."""

    bm25: float = 1.0
    bm25x: float = 0.6  # concept-expanded BM25 (SE thesaurus)
    bq: float = 0.8
    mrl64: float = 0.9
    graph: float = 0.5


class AegisSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="AEGIS_", env_nested_delimiter="__", extra="ignore")

    home: Path = Field(default_factory=_default_home)
    db_path: str = ""  # empty -> <home>/aegis.db ; ":memory:" for ephemeral

    # Embeddings
    embedding_provider: Literal["fast-local", "ollama", "openai", "voyage", "mlx"] = "fast-local"
    embedding_model: str = ""
    embedding_cache_path: str = ""  # model-backed engines: content-hash cache (default <home>/embeddings.db)
    embed_dim: int = 512
    colbert_dim: int = 128
    mrl_dims: tuple[int, ...] = (64, 128, 256, 512)
    hash_buckets: int = 1 << 14
    embed_seed: int = 0xAE615

    # Retrieval
    rrf_k: int = 60
    rrf_weights: RRFWeights = Field(default_factory=RRFWeights)
    stage1_candidates: int = 48
    colbert_max_doc_tokens: int = 160
    maxsim_blend: float = 0.75  # final = blend*maxsim + (1-blend)*rrf (both min-max normalised)
    dense_weight: float | None = None  # weight of the full dense cosine in the final score (None = engine default)
    dense_weight_neural: float = 0.75  # default for model-backed engines (frozen on dev MRR, see README)
    dense_weight_local: float = 0.25  # default for FastLocalEngine (frozen on dev MRR)
    composite_index: bool = False  # index from the composite document (literals, codes, 1-hop calls); off: lower dev MRR
    query_idf: bool = False  # weight query terms by corpus IDF (off: lower dev MRR, see README ablation)
    graph_alpha: float = 0.0  # 1-hop propagation s'(B) = s(B) + alpha*max_{A in N(B)} s(A); 0.35 tested, off by default (dev MRR)
    graph_seed_top: int = 12  # neighbours of the top-k stage-2 hits join the candidate set
    callee_activation: bool = False  # downward callee activation into helpers (off: lower dev MRR, see README)
    callee_gamma1: float = 0.95  # s(H) = max(s(H), gamma1 * s(P)) for direct callees of a top hit P
    callee_gamma2: float = 0.85  # ... and gamma2 for 2-hop callees
    callee_leaf_boost: float = 0.05  # extra for activated callees that call no further project function
    callee_max_in_degree: int = 0  # >0: only activate helpers with <= this many same-file callers (exclusive helpers)
    # hierarchical priors (None = engine default frozen on dev MRR: FastLocal 0.25 / 0.15 / 0, neural 0 / 0 / 0)
    file_prior_weight: float | None = None  # + w * best score in the symbol's file
    sibling_caller_weight: float | None = None  # + w * best same-file caller score (inside the top file only)
    salience_weight: float | None = None  # + w * normalised structural salience (state mutation / boundaries / locks)
    prior_full_per_file: int = 0  # >0: only the top-N units per file get the full prior (diversity cap)
    prior_decay: float = 0.5  # prior multiplier for the (N+1)th+ unit of the same file
    co_promote_callees: bool = True  # allocator: parent at L0 -> same-file callees to L0 if budget allows

    # Folding / allocation
    token_budget: int = 6000  # absolute cap for in-request code context
    budget_ratio: float = 0.22  # fraction of original code tokens allowed (min with token_budget)
    min_budget: int = 600
    l1_max_depth: int = 1  # control-flow nesting kept in L1 bodies (l1_mode="skeleton")
    l1_mode: str = "signature"
    cold_token_estimate: bool = True  # size units with the O(1) estimator on ingest; reported metrics stay exact  # "signature": declaration + summary docstring + one fold; "skeleton": + control flow
    relevance_temperature: float = 0.08

    # KV paging
    delta_budget: int = 2500  # tokens of newly paged-in/promoted pages per turn
    kv_policy: str = "stable"  # "stable": pinned repo L2 map + capped per-turn delta; "adaptive": delta_budget only
    kv_delta_frac: float = 0.15  # stable policy: per-turn delta <= this fraction of the resident block
    kv_max_new_l0: int = 2  # stable policy: new L0 functions per turn (further ones arrive at L1)
    page_in_top_k: int = 4
    repo_page_in: bool = False  # also page in relevant units from the indexed repository (aegis index)

    # Negative RAG (None -> self-calibrated to the active engine: mean + z*std of unrelated-task cosine)
    negative_threshold: float | None = None
    negative_threshold_z: float = 5.0

    # Proxy
    proxy_host: str = "127.0.0.1"
    proxy_port: int = 8787
    proxy_mode: ProxyMode = "upstream-forward"
    upstream_anthropic: str = "https://api.anthropic.com"
    upstream_openai: str = "https://api.openai.com"
    upstream_timeout_s: float = 600.0

    # Self-healing
    heal_min_confidence: float = 0.40

    def resolved_db_path(self) -> str:
        if self.db_path:
            return self.db_path
        self.home.mkdir(parents=True, exist_ok=True)
        return str(self.home / "aegis.db")


@lru_cache(maxsize=1)
def get_settings() -> AegisSettings:
    return AegisSettings()
