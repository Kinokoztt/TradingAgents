"""US fundamentals facade.

Delegates to the configured fundamental_data vendor (yfinance/alpha_vantage)
via the dataflows router, so the regime gate sees one swappable surface and the
vendor can be switched in config without touching L1.
"""

from __future__ import annotations

from tradingagents.dataflows.interface import route_to_vendor


def get_fundamentals(ticker: str, curr_date: str) -> str:
    """LLM-ready fundamentals report for ``ticker`` as of ``curr_date``."""
    return route_to_vendor("get_fundamentals", ticker, curr_date)
