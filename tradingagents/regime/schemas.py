"""Three-layer output contract for the regime gate (T3).

L1 (StockSignal) -> L2 (ConceptSignal) -> L3 (RegimeReport). The LLM commander
emits a typed RegimeReport via structured output; downstream quant models
consume the whitelists + catalyst confidences. See docs/regime-gate-design.md
§5.2.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field, model_validator


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


class HorizonOutlook(BaseModel):
    """B (multi-horizon): the commander's market call for one forward horizon.

    Forecast-only: the tradable whitelists stay anchored to the near-term
    ``market_state`` to avoid long-horizon overconfidence (design §3.4). The
    evaluator (module A) grades each horizon's ``direction`` against that
    horizon's realized path, so calibration can be tracked per holding period.
    """

    horizon: str = Field(description="'1d' | '3d' | '5d' (trading days, session counts as day 1)")
    direction: MarketRegime
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    rationale: str = ""

    @property
    def horizon_days(self) -> int:
        """Trading-day count parsed from ``horizon`` ('3d' -> 3)."""
        return int(self.horizon.strip().lower().rstrip("d"))


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

    @model_validator(mode="after")
    def _coerce_strength(self) -> "ConceptSignal":
        """Enforce direction/strength coherence (LLM sometimes emits e.g. Long/Neutral).

        - Block (no lean) ⇒ strength is Neutral by definition.
        - A directional lean (Long/Short) cannot be Neutral conviction: derive
          Strong/Weak from confidence (split at 0.5).
        """
        if self.direction is Direction.BLOCK:
            if self.strength is not Strength.NEUTRAL:
                object.__setattr__(self, "strength", Strength.NEUTRAL)
        elif self.strength is Strength.NEUTRAL:
            object.__setattr__(self, "strength", Strength.STRONG if self.confidence >= 0.5 else Strength.WEAK)
        return self


class RegimeReport(BaseModel):
    """L3: market regime + concept and stock signals (the top-level output).

    **Layer judgments are kept raw and are never overwritten by upper layers.**
    ``stock_signals``/``concept_signals`` hold the honest bottom-up direction
    each layer decided; the regime "circuit breaker" is *not* baked in by mutating
    them. Instead it's a **consumption-time rule** applied via the ``tradable_*``
    views: e.g. a Bearish ``market_state`` vetoes downstream Long permits without
    erasing the fact that L1 saw a bullish catalyst. This keeps every layer
    auditable (and usable by the evaluator / knowledge base) while still letting
    the commander gate what is actually tradable.
    """

    as_of_date: str
    market_state: MarketRegime
    macro_summary: str = Field(description="Macro/fundamental synthesis, micro-noise stripped")
    key_drivers: list[str] = Field(
        default_factory=list, description="LLM's top regime drivers behind market_state (audit trail)"
    )
    macro_snapshot: str = Field(
        default="",
        description="Raw structured macro_daily snapshot fed to L3 — the real numbers, "
        "for auditing market_state/circuit-breaker against the narrative macro_summary",
    )
    economic_calendar: str = Field(default="", description="Economic calendar input fed to L3 (audit trail)")
    range_min_confidence: float = Field(
        default=0.6, ge=0.0, le=1.0,
        description="Consumption rule param: in a Range regime, Long permits below this "
        "catalyst_confidence are vetoed (tradable_* views); recorded so the rule is reproducible.",
    )
    outlook: list[HorizonOutlook] = Field(
        default_factory=list,
        description="B: multi-horizon (1/3/5d) market outlook. Forecast only — the near-term "
        "market_state still drives the whitelists; the evaluator grades each horizon separately.",
    )
    concept_signals: list[ConceptSignal] = Field(default_factory=list)
    stock_signals: list[StockSignal] = Field(default_factory=list)

    def outlook_for(self, horizon_days: int) -> HorizonOutlook | None:
        """The multi-horizon call matching ``horizon_days`` trading days, if emitted."""
        for o in self.outlook:
            if o.horizon_days == horizon_days:
                return o
        return None

    # --- raw layer views (un-gated, as each layer decided) ---
    @property
    def long_whitelist(self) -> list[str]:
        """Raw L1 Long permits (NOT regime-gated; see ``tradable_long_whitelist``)."""
        return [s.ticker for s in self.stock_signals if s.direction is Direction.LONG]

    @property
    def short_whitelist(self) -> list[str]:
        return [s.ticker for s in self.stock_signals if s.direction is Direction.SHORT]

    @property
    def block_list(self) -> list[str]:
        """Tickers L1 itself blocked (per-stock bad news) — distinct from regime veto."""
        return [s.ticker for s in self.stock_signals if s.direction is Direction.BLOCK]

    # --- consumption rule: regime veto applied on top of raw signals ---
    def is_regime_vetoed_long(self, s: StockSignal) -> bool:
        """The veto rule: should this raw Long be skipped given ``market_state``?

        Bearish ⇒ veto every Long; Range ⇒ veto Longs below ``range_min_confidence``;
        Bullish ⇒ veto none. Shorts are never vetoed by this rule.
        """
        if s.direction is not Direction.LONG:
            return False
        if self.market_state is MarketRegime.BEARISH:
            return True
        if self.market_state is MarketRegime.RANGE:
            return s.catalyst_confidence < self.range_min_confidence
        return False

    @property
    def regime_blocked_longs(self) -> list[str]:
        """Raw Longs the regime rule vetoes (kept as Long in stock_signals)."""
        return [s.ticker for s in self.stock_signals if self.is_regime_vetoed_long(s)]

    @property
    def tradable_long_whitelist(self) -> list[str]:
        """Long permits that survive the regime veto — the actionable long list."""
        return [s.ticker for s in self.stock_signals
                if s.direction is Direction.LONG and not self.is_regime_vetoed_long(s)]

    @property
    def tradable_short_whitelist(self) -> list[str]:
        """Actionable short list (regime veto doesn't touch shorts today)."""
        return self.short_whitelist
