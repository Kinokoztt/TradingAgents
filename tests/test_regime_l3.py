"""Tests for L3 regime verdict + circuit breaker (no network)."""

import pytest

from tradingagents.regime import (
    ConceptSignal,
    Direction,
    MarketRegime,
    RegimeReport,
    StockSignal,
    Strength,
    analyze_regime,
    apply_circuit_breaker,
)
from tradingagents.regime.l3_regime import _L3Verdict

pytestmark = pytest.mark.unit


def _signals():
    return [
        StockSignal(ticker="HIGH", direction=Direction.LONG, catalyst_confidence=0.9, reason="strong"),
        StockSignal(ticker="LOW", direction=Direction.LONG, catalyst_confidence=0.3, reason="weak"),
        StockSignal(ticker="SHRT", direction=Direction.SHORT, catalyst_confidence=0.7, reason="bearish"),
    ]


def _report(state):
    return RegimeReport(
        as_of_date="2026-06-08", market_state=state, macro_summary="x", stock_signals=_signals()
    )


def test_bearish_blocks_all_longs():
    out = apply_circuit_breaker(_report(MarketRegime.BEARISH))
    by = {s.ticker: s.direction for s in out.stock_signals}
    assert by["HIGH"] is Direction.BLOCK
    assert by["LOW"] is Direction.BLOCK
    assert by["SHRT"] is Direction.SHORT  # shorts untouched


def test_range_blocks_only_low_confidence_longs():
    out = apply_circuit_breaker(_report(MarketRegime.RANGE), range_min_confidence=0.6)
    by = {s.ticker: s.direction for s in out.stock_signals}
    assert by["HIGH"] is Direction.LONG
    assert by["LOW"] is Direction.BLOCK
    assert by["SHRT"] is Direction.SHORT


def test_bullish_is_passthrough():
    out = apply_circuit_breaker(_report(MarketRegime.BULLISH))
    assert [s.direction for s in out.stock_signals] == [Direction.LONG, Direction.LONG, Direction.SHORT]


class _FakeTools:
    def get_macro_summary(self, curr_date, look_back_days=10):
        return "VIX low, rates stable."

    def get_market_news(self, curr_date, look_back_days=10, end_datetime=None):
        return "Risk-on tone."

    def get_economic_calendar(self, start_date, end_date):
        return f"No major prints {start_date}..{end_date}."


class _FakeStructured:
    def __init__(self, verdict):
        self._verdict = verdict

    def invoke(self, _prompt):
        return self._verdict


class _FakeLLM:
    def __init__(self, verdict):
        self._verdict = verdict

    def with_structured_output(self, _schema):
        return _FakeStructured(self._verdict)


def test_analyze_regime_assembles_and_breaks():
    verdict = _L3Verdict(market_state=MarketRegime.BEARISH, macro_summary="Systemic risk.", key_drivers=["VIX spike"])
    report = analyze_regime(
        "2026-06-08",
        concept_signals=[ConceptSignal(concept="Semis", strength=Strength.WEAK, member_tickers=["NVDA"], rationale="r")],
        stock_signals=_signals(),
        llm=_FakeLLM(verdict),
        tools=_FakeTools(),
    )
    assert report.market_state is MarketRegime.BEARISH
    assert report.macro_summary == "Systemic risk."
    # bearish breaker zeroed out the long whitelist
    assert report.long_whitelist == []
    assert "HIGH" in report.block_list and "LOW" in report.block_list
