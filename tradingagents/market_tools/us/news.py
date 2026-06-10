"""US news facade.

Delegates to the vendor adapters in ``dataflows`` (Massive for stock-level
news, FMP for market news + economic calendar) so this market layer stays a
thin, swappable surface and the HTTP/parsing code is reused, not duplicated.
"""

from __future__ import annotations

import pandas as pd

from tradingagents.dataflows import fmp, massive


def load_news_articles(
    start_date: str,
    end_date: str,
    ticker: str | None = None,
    max_articles: int = 20000,
) -> pd.DataFrame:
    """Market-wide (or single-ticker) news as a co-mention frame.

    Returns columns ``date`` (datetime) and ``tickers`` (list[str]). Articles
    with no tickers are dropped — they cannot form co-mention edges.
    """
    raw = massive.fetch_news_articles(start_date, end_date, ticker=ticker, max_articles=max_articles)
    rows = [
        # published_utc is tz-aware (…Z); drop tz to a naive UTC wall-clock so
        # the co-mention builder can compare against naive as_of dates.
        {"date": pd.to_datetime(a["published_utc"], utc=True).tz_localize(None), "tickers": a["tickers"]}
        for a in raw
        if a["tickers"]
    ]
    return pd.DataFrame(rows, columns=["date", "tickers"])


def get_stock_news(ticker: str, start_date: str, end_date: str) -> str:
    """LLM-ready stock news block (Massive)."""
    return massive.get_news(ticker, start_date, end_date)


def get_market_news(curr_date: str, look_back_days: int = 3, end_datetime: str | None = None) -> str:
    """LLM-ready market/macro news block (FMP general news).

    ``end_datetime`` ("YYYY-MM-DD HH:MM:SS") drops items published after it, for
    pre-market cutoff (best-effort; see fmp.get_global_news).
    """
    return fmp.get_global_news(curr_date, look_back_days=look_back_days, end_datetime=end_datetime)


def get_economic_calendar(start_date: str, end_date: str) -> str:
    """LLM-ready economic calendar (FMP)."""
    return fmp.get_economic_calendar(start_date, end_date)
