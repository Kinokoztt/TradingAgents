"""Tests for the regime gate output contract (T3)."""

import pytest
from pydantic import ValidationError

from tradingagents.regime import (
    ConceptSignal,
    Direction,
    HorizonOutlook,
    MarketRegime,
    RegimeReport,
    StockSignal,
    Strength,
)

pytestmark = pytest.mark.unit


def _report():
    return RegimeReport(
        as_of_date="2026-06-08",
        market_state=MarketRegime.BULLISH,
        macro_summary="Rates stable, VIX low.",
        concept_signals=[
            ConceptSignal(
                concept="Semiconductor", cluster_id="TH_0", strength=Strength.STRONG,
                member_tickers=["NVDA", "AMD"], rationale="AI capex tailwind",
            )
        ],
        stock_signals=[
            StockSignal(ticker="NVDA", direction=Direction.LONG, catalyst_confidence=0.8, reason="beat"),
            StockSignal(ticker="XOM", direction=Direction.SHORT, catalyst_confidence=0.4, reason="oil down"),
            StockSignal(ticker="ABC", direction=Direction.BLOCK, catalyst_confidence=0.1, reason="halt risk"),
        ],
    )


def test_whitelists_derive_from_direction():
    r = _report()
    assert r.long_whitelist == ["NVDA"]
    assert r.short_whitelist == ["XOM"]
    assert r.block_list == ["ABC"]


def test_catalyst_confidence_bounds():
    with pytest.raises(ValidationError):
        StockSignal(ticker="X", direction=Direction.LONG, catalyst_confidence=1.5, reason="bad")


def test_roundtrip_json():
    r = _report()
    restored = RegimeReport.model_validate_json(r.model_dump_json())
    assert restored.market_state is MarketRegime.BULLISH
    assert restored.long_whitelist == ["NVDA"]


def test_enum_values_are_stable():
    # downstream JSON consumers depend on these string values
    assert MarketRegime.BULLISH.value == "Bullish"
    assert Direction.BLOCK.value == "Block"
    assert Strength.WEAK.value == "Weak"


def test_concept_signal_block_forces_neutral_strength():
    cs = ConceptSignal(concept="X", direction=Direction.BLOCK, strength=Strength.STRONG, confidence=0.8)
    assert cs.strength is Strength.NEUTRAL


def test_concept_signal_directional_neutral_coerced_by_confidence():
    strong = ConceptSignal(concept="X", direction=Direction.LONG, strength=Strength.NEUTRAL, confidence=0.7)
    weak = ConceptSignal(concept="Y", direction=Direction.SHORT, strength=Strength.NEUTRAL, confidence=0.3)
    assert strong.strength is Strength.STRONG
    assert weak.strength is Strength.WEAK


def test_concept_signal_explicit_strength_preserved():
    cs = ConceptSignal(concept="X", direction=Direction.LONG, strength=Strength.WEAK, confidence=0.9)
    assert cs.strength is Strength.WEAK


def test_horizon_outlook_days_parse():
    assert HorizonOutlook(horizon="1d", direction=MarketRegime.BULLISH).horizon_days == 1
    assert HorizonOutlook(horizon="3D", direction=MarketRegime.RANGE).horizon_days == 3
    assert HorizonOutlook(horizon=" 5d ", direction=MarketRegime.BEARISH).horizon_days == 5


def test_outlook_for_looks_up_by_horizon_days():
    r = RegimeReport(
        as_of_date="2026-06-08", market_state=MarketRegime.RANGE, macro_summary="x",
        outlook=[
            HorizonOutlook(horizon="1d", direction=MarketRegime.RANGE, confidence=0.7),
            HorizonOutlook(horizon="3d", direction=MarketRegime.BULLISH, confidence=0.6),
            HorizonOutlook(horizon="5d", direction=MarketRegime.BULLISH, confidence=0.5),
        ],
    )
    assert r.outlook_for(1).direction is MarketRegime.RANGE
    assert r.outlook_for(3).direction is MarketRegime.BULLISH
    assert r.outlook_for(3).confidence == pytest.approx(0.6)
    assert r.outlook_for(2) is None  # no 2d call emitted


def test_outlook_roundtrip_json():
    r = RegimeReport(
        as_of_date="2026-06-08", market_state=MarketRegime.BULLISH, macro_summary="x",
        outlook=[HorizonOutlook(horizon="3d", direction=MarketRegime.BEARISH, confidence=0.4, rationale="CPI risk")],
    )
    restored = RegimeReport.model_validate_json(r.model_dump_json())
    assert restored.outlook_for(3).direction is MarketRegime.BEARISH
    assert restored.outlook_for(3).rationale == "CPI risk"
