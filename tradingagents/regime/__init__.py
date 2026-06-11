"""Regime gate: the strategic commander / circuit-breaker layer.

Three-layer, bottom-up aggregation (see docs/regime-gate-design.md):
  L1 per-stock signals -> L2 concept-cluster strength -> L3 market regime.
This package currently exposes the output contract (schemas); the L1/L2/L3
orchestration agents land next.
"""

from .commander import run_regime_gate
from .evaluate import HorizonScore, Scorecard, WhitelistScore, evaluate_report
from .l1_stock import analyze_stocks, select_news_tickers
from .l2_concept import (
    aggregate_concepts,
    judge_clusters,
    judge_sectors,
    propagate_catalysts,
)
from .l3_regime import analyze_regime
from .schemas import (
    ConceptSignal,
    Direction,
    MarketRegime,
    RegimeReport,
    StockSignal,
    Strength,
)

__all__ = [
    "MarketRegime",
    "Direction",
    "Strength",
    "StockSignal",
    "ConceptSignal",
    "RegimeReport",
    "analyze_regime",
    "aggregate_concepts",
    "judge_clusters",
    "judge_sectors",
    "propagate_catalysts",
    "select_news_tickers",
    "analyze_stocks",
    "run_regime_gate",
    "evaluate_report",
    "Scorecard",
    "HorizonScore",
    "WhitelistScore",
]
