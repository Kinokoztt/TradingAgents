"""Tests for the regime gate output contract (T3)."""

import pytest
from pydantic import ValidationError

from tradingagents.regime import (
    ConceptSignal,
    Direction,
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
