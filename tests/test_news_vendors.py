"""Unit tests for the Massive + FMP news vendors and the co-mention adapter.

HTTP is mocked, so these run offline without real API keys.
"""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from tradingagents.dataflows import fmp, massive
from tradingagents.market_tools.us import news as news_tools
from tradingagents.market_tools.us import splits as splits_tool

pytestmark = pytest.mark.unit


def _resp(payload):
    r = MagicMock()
    r.json.return_value = payload
    r.raise_for_status.return_value = None
    return r


@pytest.fixture(autouse=True)
def _news_keys(monkeypatch):
    monkeypatch.setenv("MASSIVE_API_KEY", "test-massive")
    monkeypatch.setenv("FMP_API_KEY", "test-fmp")


def test_massive_fetch_paginates_and_normalizes():
    page1 = {
        "results": [
            {
                "id": "1",
                "published_utc": "2024-06-02T13:00:00Z",
                "tickers": ["AAPL", "MSFT"],
                "title": "Big tech rallies",
                "publisher": {"name": "Massive"},
            }
        ],
        "next_url": "https://api.massive.com/v2/reference/news?cursor=abc",
    }
    page2 = {
        "results": [
            {
                "id": "2",
                "published_utc": "2024-06-01T13:00:00Z",
                "tickers": ["NVDA"],
                "title": "Chips up",
                "publisher": {"name": "Massive"},
            }
        ]
    }
    with patch.object(massive.requests, "get", side_effect=[_resp(page1), _resp(page2)]) as g:
        articles = massive.fetch_news_articles("2024-06-01", "2024-06-02")

    assert len(articles) == 2
    assert articles[0]["tickers"] == ["AAPL", "MSFT"]
    assert articles[0]["date"] == "2024-06-02"
    assert g.call_count == 2  # followed next_url


def test_massive_fetch_respects_max_articles():
    page = {
        "results": [
            {"id": str(i), "published_utc": "2024-06-02T13:00:00Z", "tickers": ["AAPL"], "title": "t"}
            for i in range(10)
        ],
        "next_url": "https://api.massive.com/next",
    }
    with patch.object(massive.requests, "get", return_value=_resp(page)):
        articles = massive.fetch_news_articles("2024-06-01", "2024-06-02", max_articles=5)
    assert len(articles) == 5


def test_massive_fetch_splits_paginates():
    page1 = {
        "results": [
            {"ticker": "NVDA", "execution_date": "2024-06-10", "split_from": 1, "split_to": 10},
        ],
        "next_url": "https://api.massive.com/v3/reference/splits?cursor=abc",
    }
    page2 = {
        "results": [
            {"ticker": "AAPL", "execution_date": "2024-08-31", "split_from": 1, "split_to": 4},
        ]
    }
    with patch.object(massive.requests, "get", side_effect=[_resp(page1), _resp(page2)]) as g:
        splits = massive.fetch_splits("2024-01-01", "2024-12-31")
    assert g.call_count == 2
    assert [s["ticker"] for s in splits] == ["NVDA", "AAPL"]
    assert splits[0] == {
        "ticker": "NVDA",
        "execution_date": "2024-06-10",
        "split_from": 1,
        "split_to": 10,
    }


def test_load_splits_filters_to_universe():
    raw = [
        {"ticker": "NVDA", "execution_date": "2024-06-10", "split_from": 1, "split_to": 10},
        {"ticker": "PENNY", "execution_date": "2024-06-11", "split_from": 20, "split_to": 1},
    ]
    with patch.object(massive, "fetch_splits", return_value=raw):
        df = splits_tool.load_splits(["NVDA", "AAPL"], "2024-01-01", "2024-12-31")
    assert df["ticker"].tolist() == ["NVDA"]
    assert pd.api.types.is_datetime64_any_dtype(df["execution_date"])


def test_massive_get_news_formats_sentiment():
    page = {
        "results": [
            {
                "published_utc": "2024-06-02T13:00:00Z",
                "tickers": ["AAPL"],
                "title": "Apple beats",
                "publisher": {"name": "Massive"},
                "description": "Strong quarter.",
                "insights": [
                    {
                        "ticker": "AAPL",
                        "sentiment": "positive",
                        "sentiment_reasoning": "earnings beat",
                    }
                ],
            }
        ]
    }
    with patch.object(massive.requests, "get", return_value=_resp(page)):
        text = massive.get_news("AAPL", "2024-06-01", "2024-06-02")
    assert "Apple beats" in text
    assert "positive" in text
    assert "earnings beat" in text


def test_massive_missing_key_raises(monkeypatch):
    monkeypatch.delenv("MASSIVE_API_KEY", raising=False)
    with pytest.raises(ValueError):
        massive.get_api_key()


def test_fmp_global_news_formats():
    payload = [
        {
            "publishedDate": "2024-06-02",
            "title": "Fed holds rates",
            "site": "Reuters",
            "text": "The Federal Reserve kept rates unchanged.",
        }
    ]
    with patch.object(fmp.requests, "get", return_value=_resp(payload)):
        text = fmp.get_global_news("2024-06-03", look_back_days=7, limit=20)
    assert "Fed holds rates" in text
    assert "Reuters" in text


def test_fmp_economic_calendar_table():
    payload = [
        {
            "date": "2024-06-07 08:30:00",
            "event": "Nonfarm Payrolls",
            "actual": "272K",
            "estimate": "185K",
            "previous": "165K",
            "impact": "High",
        }
    ]
    with patch.object(fmp.requests, "get", return_value=_resp(payload)):
        text = fmp.get_economic_calendar("2024-06-01", "2024-06-08")
    assert "Nonfarm Payrolls" in text
    assert "272K" in text
    assert "| Date | Event |" in text


def test_comention_adapter_drops_empty_tickers():
    raw = [
        {"published_utc": "2024-06-02T13:00:00Z", "tickers": ["AAPL", "MSFT"]},
        {"published_utc": "2024-06-02T14:00:00Z", "tickers": []},  # dropped
    ]
    with patch.object(massive, "fetch_news_articles", return_value=raw):
        df = news_tools.load_news_articles("2024-06-01", "2024-06-02")
    assert list(df.columns) == ["date", "tickers"]
    assert len(df) == 1
    assert pd.api.types.is_datetime64_any_dtype(df["date"])
