"""Tests for L3 regime verdict + the non-destructive regime veto (no network)."""

import pytest

from tradingagents.regime import (
    ConceptSignal,
    Direction,
    HorizonOutlook,
    MarketRegime,
    RegimeReport,
    StockSignal,
    Strength,
    analyze_regime,
)
from tradingagents.regime.l3_regime import _L3Verdict

pytestmark = pytest.mark.unit


def _signals():
    return [
        StockSignal(ticker="HIGH", direction=Direction.LONG, catalyst_confidence=0.9, reason="strong"),
        StockSignal(ticker="LOW", direction=Direction.LONG, catalyst_confidence=0.3, reason="weak"),
        StockSignal(ticker="SHRT", direction=Direction.SHORT, catalyst_confidence=0.7, reason="bearish"),
    ]


def _report(state, range_min_confidence=0.6):
    return RegimeReport(
        as_of_date="2026-06-08", market_state=state, macro_summary="x",
        range_min_confidence=range_min_confidence, stock_signals=_signals()
    )


def test_bearish_vetoes_all_longs_without_overwriting():
    r = _report(MarketRegime.BEARISH)
    # raw L1 directions are preserved (not recoded to Block)...
    assert {s.ticker: s.direction for s in r.stock_signals} == {
        "HIGH": Direction.LONG, "LOW": Direction.LONG, "SHRT": Direction.SHORT
    }
    assert r.long_whitelist == ["HIGH", "LOW"]        # raw view unchanged
    # ...but the consumption rule vetoes every Long, shorts pass through.
    assert r.tradable_long_whitelist == []
    assert set(r.regime_blocked_longs) == {"HIGH", "LOW"}
    assert r.tradable_short_whitelist == ["SHRT"]


def test_range_vetoes_only_low_confidence_longs():
    r = _report(MarketRegime.RANGE, range_min_confidence=0.6)
    assert r.tradable_long_whitelist == ["HIGH"]       # 0.9 >= 0.6 survives
    assert r.regime_blocked_longs == ["LOW"]           # 0.3 < 0.6 vetoed
    assert {s.direction for s in r.stock_signals if s.ticker == "LOW"} == {Direction.LONG}  # still raw Long


def test_bullish_vetoes_nothing():
    r = _report(MarketRegime.BULLISH)
    assert r.tradable_long_whitelist == ["HIGH", "LOW"]
    assert r.regime_blocked_longs == []


class _FakeTools:
    def get_macro_summary(self, curr_date, look_back_days=10):
        return "VIX low, rates stable."

    def get_market_news(self, curr_date, look_back_days=10, end_datetime=None):
        return "Risk-on tone."

    def get_economic_calendar(self, start_date, end_date, cutoff=None):
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


def test_analyze_regime_preserves_raw_signals_and_records_rule():
    verdict = _L3Verdict(market_state=MarketRegime.BEARISH, macro_summary="Systemic risk.", key_drivers=["VIX spike"])
    report = analyze_regime(
        "2026-06-08",
        concept_signals=[ConceptSignal(concept="Semis", strength=Strength.WEAK, member_tickers=["NVDA"], rationale="r")],
        stock_signals=_signals(),
        range_min_confidence=0.6,
        llm=_FakeLLM(verdict),
        tools=_FakeTools(),
    )
    assert report.market_state is MarketRegime.BEARISH
    assert report.macro_summary == "Systemic risk."
    assert report.range_min_confidence == 0.6
    # raw L1 Longs are NOT overwritten; the veto is a consumption-time view.
    assert report.long_whitelist == ["HIGH", "LOW"]
    assert report.block_list == []                     # nothing recoded to Block
    assert report.tradable_long_whitelist == []        # bearish vetoes all longs
    assert set(report.regime_blocked_longs) == {"HIGH", "LOW"}


def test_analyze_regime_carries_multi_horizon_outlook():
    verdict = _L3Verdict(
        market_state=MarketRegime.RANGE, macro_summary="Choppy.", key_drivers=["CPI ahead"],
        outlook=[
            HorizonOutlook(horizon="1d", direction=MarketRegime.RANGE, confidence=0.7, rationale="event risk"),
            HorizonOutlook(horizon="3d", direction=MarketRegime.BULLISH, confidence=0.6, rationale="post-print relief"),
            HorizonOutlook(horizon="5d", direction=MarketRegime.BULLISH, confidence=0.55, rationale="trend resumes"),
        ],
    )
    report = analyze_regime("2026-06-08", llm=_FakeLLM(verdict), tools=_FakeTools())
    assert [o.horizon for o in report.outlook] == ["1d", "3d", "5d"]
    assert report.outlook_for(1).direction is MarketRegime.RANGE
    assert report.outlook_for(5).direction is MarketRegime.BULLISH
    # near-term anchor unchanged; outlook is forecast-only and doesn't touch whitelists.
    assert report.market_state is MarketRegime.RANGE
