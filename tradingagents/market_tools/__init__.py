"""Market-scoped data tools: the swappable external-data layer.

The regime gate and concept graph never talk to a specific vendor or BigQuery
table directly. They go through a market's tools module, which implements the
``MarketDataTools`` contract below. Swapping ``get_market_tools("US")`` for
another market is how the whole pipeline is re-pointed at a different data
source / exchange — "update the tools = switch the data source".

US is the only market implemented today; add ``market_tools/<mkt>/`` with the
same surface to support another.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import pandas as pd


@runtime_checkable
class MarketDataTools(Protocol):
    """The data surface every market implementation must provide.

    News / macro getters return LLM-ready text blocks; the loaders return
    DataFrames / lists for the quantitative layers (concept graph, filters).
    """

    # --- news (non-structured) ---
    def load_news_articles(
        self, start_date: str, end_date: str, ticker: str | None = ..., max_articles: int = ...
    ) -> pd.DataFrame: ...
    def get_stock_news(self, ticker: str, start_date: str, end_date: str) -> str: ...
    def get_market_news(self, curr_date: str, look_back_days: int = ..., end_datetime: str | None = ...) -> str: ...
    def get_economic_calendar(self, start_date: str, end_date: str) -> str: ...

    # --- fundamentals (vendor router) ---
    def get_fundamentals(self, ticker: str, curr_date: str) -> str: ...

    # --- prices (BigQuery) ---
    def load_daily_close(self, tickers: list[str], start_date: str, end_date: str) -> pd.DataFrame: ...
    def load_minute_close(self, tickers: list[str], trade_date: str) -> pd.DataFrame: ...
    def load_splits(self, tickers: list[str] | None, start_date: str, end_date: str) -> pd.DataFrame: ...
    def latest_trading_day(self, as_of: str | None = ...) -> str: ...
    def previous_trading_day(self, session_date: str) -> str: ...

    # --- macro (BigQuery) ---
    def load_macro_daily(self, start_date: str, end_date: str) -> pd.DataFrame: ...
    def get_macro_summary(self, curr_date: str, look_back_days: int = ...) -> str: ...

    # --- universe (BigQuery) ---
    def load_candidate_universe(
        self, min_avg_volume: float = ..., min_avg_price: float = ..., ticker_type: str | None = ...
    ) -> list[str]: ...


def get_market_tools(market: str = "US") -> MarketDataTools:
    """Return the tools module for ``market`` (only "US" implemented)."""
    if market.upper() == "US":
        from . import us

        return us
    raise ValueError(f"No market tools implemented for market={market!r}")
