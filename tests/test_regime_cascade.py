"""Tests for the hierarchical LLM cascade: judge_clusters, judge_sectors, commander."""

import re

import pandas as pd
import pytest

from tradingagents.concept_graph.schemas import Cluster, Membership
from tradingagents.regime import (
    ConceptSignal,
    Direction,
    Strength,
    StockSignal,
    judge_clusters,
    judge_sectors,
    run_regime_gate,
)
from tradingagents.regime.l1_stock import _StockSignalBatch
from tradingagents.regime.l2_concept import _ConceptVerdict
from tradingagents.regime.l3_regime import _L3Verdict
from tradingagents.regime.schemas import MarketRegime

pytestmark = pytest.mark.unit


def _cluster_map():
    return {
        "NVDA": [Membership(cluster_id="TH_0", weight=1.0, is_primary=True)],
        "AMD": [Membership(cluster_id="TH_0", weight=1.0, is_primary=True)],
        "MU": [Membership(cluster_id="TH_0", weight=0.8, is_primary=True)],
        "XOM": [Membership(cluster_id="TH_1", weight=1.0, is_primary=True)],
        "CVX": [Membership(cluster_id="TH_1", weight=1.0, is_primary=True)],
    }


def _clusters():
    return {
        "TH_0": Cluster(cluster_id="TH_0", parent_sector="Technology",
                        members=["NVDA", "AMD", "MU"], representatives=["NVDA"], label="AI Chips"),
        "TH_1": Cluster(cluster_id="TH_1", parent_sector="Energy",
                        members=["XOM", "CVX"], representatives=["XOM"], label="Oil & Gas"),
    }


class _FakeTools:
    def load_candidate_universe(self, **_):
        return ["NVDA", "AMD", "MU", "XOM", "CVX"]

    def load_news_articles(self, start_date, end_date, max_articles=20000, ticker=None):
        return pd.DataFrame({"tickers": [["NVDA", "AMD"], ["MU"], ["XOM"], ["CVX"]]})

    def get_stock_news(self, ticker, start_date, end_date):
        return f"news for {ticker}"

    def get_fundamentals(self, ticker, curr_date):
        return f"fundamentals for {ticker}"

    def get_macro_summary(self, curr_date, look_back_days=10):
        return "VIX low."

    def get_market_news(self, curr_date, look_back_days=10, end_datetime=None):
        return "Risk-on."

    def get_economic_calendar(self, start_date, end_date, cutoff=None):
        return "No prints."


class _FakeStructured:
    def __init__(self, schema):
        self._schema = schema

    def invoke(self, prompt):
        name = self._schema.__name__
        if name == "_StockSignalBatch":
            tickers = re.findall(r"^### (\S+)$", prompt, flags=re.MULTILINE)
            return _StockSignalBatch(
                signals=[
                    StockSignal(ticker=t, direction=Direction.LONG, catalyst_confidence=0.7, reason=f"c {t}")
                    for t in tickers
                ]
            )
        if name == "_ConceptVerdict":
            return _ConceptVerdict(
                direction=Direction.LONG, strength=Strength.STRONG, confidence=0.8, rationale="bullish theme"
            )
        if name == "_L3Verdict":
            return _L3Verdict(market_state=MarketRegime.BULLISH, macro_summary="ok", key_drivers=[])
        raise AssertionError(f"unexpected schema {name}")


class _FakeLLM:
    def with_structured_output(self, schema):
        return _FakeStructured(schema)


def _signals():
    return [
        StockSignal(ticker="NVDA", direction=Direction.LONG, catalyst_confidence=0.9, reason="beat"),
        StockSignal(ticker="AMD", direction=Direction.LONG, catalyst_confidence=0.8, reason="up"),
        StockSignal(ticker="MU", direction=Direction.LONG, catalyst_confidence=0.7, reason="hbm"),
    ]


def test_judge_clusters_only_active_clusters():
    out = judge_clusters(
        "2026-06-08", _signals(), tools=_FakeTools(), llm=_FakeLLM(),
        cluster_map=_cluster_map(), clusters=_clusters(), min_members=2,
    )
    # only TH_0 has >=2 actionable members; TH_1 has no signalled members
    assert len(out) == 1
    cs = out[0]
    assert cs.level == "theme"
    assert cs.cluster_id == "TH_0"
    assert cs.parent_sector == "Technology"
    assert cs.direction is Direction.LONG
    assert cs.strength is Strength.STRONG


def test_judge_sectors_groups_by_parent_sector():
    themes = [
        ConceptSignal(concept="AI Chips", cluster_id="TH_0", level="theme", parent_sector="Technology",
                      direction=Direction.LONG, strength=Strength.STRONG, confidence=0.8,
                      member_tickers=["NVDA", "AMD"], rationale="x"),
        ConceptSignal(concept="Software", cluster_id="TH_2", level="theme", parent_sector="Technology",
                      direction=Direction.LONG, strength=Strength.NEUTRAL, confidence=0.5,
                      member_tickers=["MSFT"], rationale="y"),
        ConceptSignal(concept="Oil & Gas", cluster_id="TH_1", level="theme", parent_sector="Energy",
                      direction=Direction.SHORT, strength=Strength.WEAK, confidence=0.3,
                      member_tickers=["XOM"], rationale="z"),
    ]
    out = judge_sectors(themes, "2026-06-08", llm=_FakeLLM())
    assert {cs.concept for cs in out} == {"Technology", "Energy"}
    assert all(cs.level == "sector" for cs in out)
    tech = next(cs for cs in out if cs.concept == "Technology")
    assert set(tech.member_tickers) == {"NVDA", "AMD", "MSFT"}


def test_premarket_cutoffs_are_session_open():
    from tradingagents.regime.commander import premarket_cutoffs

    utc, fmp = premarket_cutoffs("2026-06-09")  # June -> EDT (UTC-4)
    assert utc == "2026-06-09T13:00:00Z"  # 09:00 ET == 13:00 UTC in summer
    assert fmp == "2026-06-09 09:00:00"


def test_select_news_tickers_passes_cutoff_end():
    captured = {}

    class _CapTools(_FakeTools):
        def load_news_articles(self, start_date, end_date, max_articles=20000, ticker=None):
            captured["end"] = end_date
            return pd.DataFrame({"tickers": [["NVDA"], ["AMD"]]})

    from tradingagents.regime import select_news_tickers

    select_news_tickers("2026-06-09", tools=_CapTools(), news_end="2026-06-09T13:30:00Z")
    assert captured["end"] == "2026-06-09T13:30:00Z"


def test_run_regime_gate_end_to_end(monkeypatch):
    monkeypatch.setattr("tradingagents.concept_graph.store.load_memberships", lambda d, o: _cluster_map())
    monkeypatch.setattr("tradingagents.concept_graph.store.load_clusters", lambda d, o: _clusters())

    report = run_regime_gate(
        "2026-06-08", tools=_FakeTools(), llm=_FakeLLM(), propagate=False, max_workers=2,
    )
    assert report.market_state is MarketRegime.BULLISH
    # stock signals produced for news-active tickers, longs survive bullish breaker
    assert "NVDA" in report.long_whitelist
    # concept_signals carry both sector and theme levels
    levels = {cs.level for cs in report.concept_signals}
    assert "theme" in levels and "sector" in levels
