"""Regime gate: the strategic commander / circuit-breaker layer.

Three-layer, bottom-up aggregation (see docs/regime-gate-design.md):
  L1 per-stock signals -> L2 concept-cluster strength -> L3 market regime.
This package currently exposes the output contract (schemas); the L1/L2/L3
orchestration agents land next.
"""

from .commander import run_regime_gate
from .evaluate import HorizonScore, Scorecard, WhitelistScore, evaluate_report
from .events import (
    Certainty,
    EventType,
    Horizon,
    Materiality,
    NewsEvent,
    Polarity,
    PriceInStatus,
    SourceReliability,
    extract_events,
)
from .l1_stock import analyze_stocks, select_news_tickers
from .l2_concept import (
    aggregate_concepts,
    judge_clusters,
    judge_sectors,
    propagate_catalysts,
)
from .l3_regime import analyze_regime
from .price_in import label_price_in, tag_price_in
from .schemas import (
    ConceptSignal,
    Direction,
    HorizonOutlook,
    MarketRegime,
    RegimeReport,
    StockSignal,
    Strength,
)
from .source_reliability import classify_source, tag_source_reliability

__all__ = [
    "MarketRegime",
    "Direction",
    "Strength",
    "StockSignal",
    "ConceptSignal",
    "HorizonOutlook",
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
    # event extraction + enrichment
    "EventType",
    "Certainty",
    "Polarity",
    "Materiality",
    "Horizon",
    "SourceReliability",
    "PriceInStatus",
    "NewsEvent",
    "extract_events",
    "classify_source",
    "tag_source_reliability",
    "label_price_in",
    "tag_price_in",
]
