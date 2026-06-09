"""US market tools — the data surface for the regime gate & concept graph.

News goes through Massive/FMP vendor adapters; prices, macro, and the
candidate universe come from BigQuery (project mystockproject-431701). API
keys are loaded from Secret Manager (see dataflows/secrets.py); BQ auth is ADC.
"""

from __future__ import annotations

from .macro import get_macro_summary, load_macro_daily
from .news import (
    get_economic_calendar,
    get_market_news,
    get_stock_news,
    load_news_articles,
)
from .prices import load_daily_close, load_minute_close
from .splits import load_splits
from .universe import load_candidate_universe

__all__ = [
    "load_news_articles",
    "get_stock_news",
    "get_market_news",
    "get_economic_calendar",
    "load_daily_close",
    "load_minute_close",
    "load_splits",
    "load_macro_daily",
    "get_macro_summary",
    "load_candidate_universe",
]
