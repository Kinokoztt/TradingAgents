"""Three-layer output contract for the regime gate (T3).

L1 (StockSignal) -> L2 (ConceptSignal) -> L3 (RegimeReport). The LLM commander
emits a typed RegimeReport via structured output; downstream quant models
consume the whitelists + catalyst confidences. See docs/regime-gate-design.md
§5.2.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class MarketRegime(str, Enum):
    BULLISH = "Bullish"
    RANGE = "Range"
    BEARISH = "Bearish"


class Direction(str, Enum):
    LONG = "Long"
    SHORT = "Short"
    BLOCK = "Block"   # circuit-breaker: not tradable


class Strength(str, Enum):
    STRONG = "Strong"
    NEUTRAL = "Neutral"
    WEAK = "Weak"


class StockSignal(BaseModel):
    """L1: per-stock direction permit + catalyst confidence."""

    ticker: str
    direction: Direction
    catalyst_confidence: float = Field(ge=0.0, le=1.0, description="0-1 catalyst confidence")
    reason: str


class ConceptSignal(BaseModel):
    """L2: verdict on a dynamic concept node (theme cluster or sector).

    Used at both hierarchy levels of the cascade (``level`` = "theme"/"sector").
    ``direction`` is the directional lean (Long=bullish, Short=bearish,
    Block=no clear lean); ``strength``/``confidence`` are conviction.
    """

    concept: str = Field(description="Concept/cluster label, e.g. 'Semiconductor', 'Solar'")
    cluster_id: str | None = Field(default=None, description="Source concept_graph cluster id, if any")
    level: str = Field(default="theme", description="'theme' | 'sector'")
    parent_sector: str | None = Field(default=None, description="Parent sector for a theme cluster")
    direction: Direction = Field(default=Direction.BLOCK, description="Bullish=Long / Bearish=Short / no-lean=Block")
    strength: Strength = Strength.NEUTRAL
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    member_tickers: list[str] = Field(default_factory=list)
    rationale: str = ""


class RegimeReport(BaseModel):
    """L3: market regime + concept and stock signals (the top-level output)."""

    as_of_date: str
    market_state: MarketRegime
    macro_summary: str = Field(description="Macro/fundamental synthesis, micro-noise stripped")
    concept_signals: list[ConceptSignal] = Field(default_factory=list)
    stock_signals: list[StockSignal] = Field(default_factory=list)

    @property
    def long_whitelist(self) -> list[str]:
        return [s.ticker for s in self.stock_signals if s.direction is Direction.LONG]

    @property
    def short_whitelist(self) -> list[str]:
        return [s.ticker for s in self.stock_signals if s.direction is Direction.SHORT]

    @property
    def block_list(self) -> list[str]:
        return [s.ticker for s in self.stock_signals if s.direction is Direction.BLOCK]
