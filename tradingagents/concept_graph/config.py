"""Tunable parameters for concept-graph edge construction and fusion.

All knobs live in one dataclass so the build pipeline, tests, and the
eventual batch job share a single source of truth. Defaults reflect the
design doc (docs/concept-graph-design.md §4–§5).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class CommunityConfig:
    """Community detection (M2): hierarchical Leiden + anti-bloat + multi-membership."""

    seed: int = 42

    # --- Hierarchical Leiden (two resolutions) ---
    sector_resolution: float = 0.5     # coarse layer (L_sector), low resolution
    # L_theme resolution is binary-searched so the cluster count lands in range.
    theme_resolution_lo: float = 0.1
    theme_resolution_hi: float = 10.0
    theme_search_iters: int = 18
    theme_target_min: int = 40         # target #theme clusters (market-scale default)
    theme_target_max: int = 80

    # --- Anti-bloat ---
    min_cluster_size: int = 4          # smaller theme clusters get reassigned / UNCLUSTERED

    # --- Controlled multi-membership ---
    multi_membership_tau: float = 0.15  # min edge-weight share to add a secondary membership
    multi_membership_k: int = 3         # max memberships per ticker (incl. primary)

    representatives_top_n: int = 5      # top weighted-degree members kept as cluster reps


UNCLUSTERED = "UNCLUSTERED"


@dataclass(frozen=True)
class GraphConfig:
    # --- Co-movement (price residual correlation) ---
    market_ticker: str = "SPY"          # de-market basis; present in day_aggs_di
    comovement_window: int = 120        # trading-day lookback for returns
    comovement_min_periods: int = 40    # min overlapping days to form an edge
    comovement_method: str = "pearson"  # "pearson" or "spearman"
    winsorize_quantile: Optional[float] = 0.01  # clip residual tails; None disables
    keep_positive_only: bool = True     # only co-rising/co-falling edges

    # --- Co-mention (news article co-occurrence) ---
    comention_window_days: int = 90     # calendar lookback
    comention_decay_lambda: float = 0.02  # exp(-lambda * age_days) time decay
    comention_max_tickers_per_article: int = 15  # drop "gainers list" noise
    comention_method: str = "jaccard"   # "jaccard" or "pmi"
    comention_min_articles: int = 2     # min co-occurring articles to form an edge

    # --- Fusion weights: edge = a*comention + b*comovement + g*etf ---
    fuse_alpha: float = 0.4             # co-mention
    fuse_beta: float = 0.5             # co-movement
    fuse_gamma: float = 0.1             # etf co-membership (P2, currently unused)
    normalize_method: str = "minmax"    # "minmax" or "rank"

    # --- Pruning ---
    prune_min_weight: float = 0.1       # drop edges below this fused weight
    prune_top_k: Optional[int] = 20     # keep top-K neighbours per node; None disables
