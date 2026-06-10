"""Tests for L1 stock signals: news-ticker selection + batched concurrent analysis."""

import re

import pandas as pd
import pytest

from tradingagents.regime import Direction, analyze_stocks, select_news_tickers
from tradingagents.regime.l1_stock import _StockSignalBatch
from tradingagents.regime.schemas import StockSignal

pytestmark = pytest.mark.unit


class _FakeTools:
    def __init__(self, articles, universe):
        self._articles = articles
        self._universe = universe

    def load_candidate_universe(self, **_):
        return self._universe

    def load_news_articles(self, start_date, end_date, max_articles=20000, ticker=None):
        return self._articles

    def get_stock_news(self, ticker, start_date, end_date):
        return f"news for {ticker}"

    def get_fundamentals(self, ticker, curr_date):
        return f"fundamentals for {ticker}"


def test_select_news_tickers_intersects_universe_and_ranks():
    articles = pd.DataFrame(
        {"tickers": [["NVDA", "AMD"], ["NVDA"], ["TSLA"], ["PRIVATE"]]}
    )
    tools = _FakeTools(articles, universe=["NVDA", "AMD", "TSLA"])  # PRIVATE excluded
    out = select_news_tickers("2026-06-08", tools=tools)
    assert out[0] == "NVDA"  # mentioned twice -> first
    assert set(out) == {"NVDA", "AMD", "TSLA"}
    assert "PRIVATE" not in out


def test_select_news_tickers_max_cap():
    articles = pd.DataFrame({"tickers": [["A"], ["A"], ["B"], ["C"]]})
    tools = _FakeTools(articles, universe=["A", "B", "C"])
    out = select_news_tickers("2026-06-08", tools=tools, max_tickers=2)
    assert out == ["A", out[1]] and len(out) == 2


class _FakeStructured:
    """Parses '### TICKER' headers from the prompt; returns a Long signal each."""

    def invoke(self, prompt):
        tickers = re.findall(r"^### (\S+)$", prompt, flags=re.MULTILINE)
        return _StockSignalBatch(
            signals=[
                StockSignal(ticker=t, direction=Direction.LONG, catalyst_confidence=0.7, reason=f"catalyst {t}")
                for t in tickers
            ]
        )


class _FakeLLM:
    def with_structured_output(self, _schema):
        return _FakeStructured()


def test_analyze_stocks_batches_and_preserves_order():
    tickers = ["NVDA", "AMD", "MU", "XOM", "CVX"]
    tools = _FakeTools(pd.DataFrame({"tickers": []}), universe=tickers)
    out = analyze_stocks(
        tickers, "2026-06-08", tools=tools, llm=_FakeLLM(), batch_size=2, max_workers=3
    )
    assert [s.ticker for s in out] == tickers  # input order preserved across batches
    assert all(s.direction is Direction.LONG for s in out)


def test_analyze_stocks_empty_is_noop():
    assert analyze_stocks([], "2026-06-08", llm=_FakeLLM(), tools=_FakeTools(pd.DataFrame(), [])) == []


def test_analyze_stocks_dedupes_by_ticker():
    # a ticker repeated across the input collapses to one signal
    tickers = ["NVDA", "NVDA"]
    tools = _FakeTools(pd.DataFrame({"tickers": []}), universe=tickers)
    out = analyze_stocks(tickers, "2026-06-08", tools=tools, llm=_FakeLLM(), batch_size=1)
    assert [s.ticker for s in out] == ["NVDA"]
