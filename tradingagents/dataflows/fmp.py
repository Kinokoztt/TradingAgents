"""Financial Modeling Prep (FMP) vendor — macro/market side of the regime gate.

Scope (Starter plan is sufficient): broad/macro market news via
``get_global_news`` and the economic release calendar via
``get_economic_calendar``. Per-ticker news and corporate catalysts come from
Massive instead. Auth is the ``apikey`` query parameter from FMP_API_KEY.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta

import requests

API_BASE_URL = "https://financialmodelingprep.com/stable"


def get_api_key() -> str:
    api_key = os.getenv("FMP_API_KEY")
    if not api_key:
        raise ValueError("FMP_API_KEY environment variable is not set.")
    return api_key


def _get(endpoint: str, params: dict):
    params = {**params, "apikey": get_api_key()}
    response = requests.get(f"{API_BASE_URL}/{endpoint}", params=params, timeout=30)
    response.raise_for_status()
    return response.json()


def _visible_premarket(published: str, end_datetime: str) -> bool:
    """Whether a news item published at ``published`` is visible at ``end_datetime``.

    Pre-market cutoff with backfill safety:
    - timestamped (has time-of-day): keep iff ``published <= end_datetime``.
    - date-only (can't tell pre/post open): keep only if it's a *strictly prior*
      day; a same-day date-only item is ambiguous on a backfill, so it's dropped.
    """
    pub = str(published or "")
    if not pub:
        return False
    if len(pub) <= 10:  # date only
        return pub[:10] < end_datetime[:10]
    return pub[:19] <= end_datetime


def get_global_news(
    curr_date: str, look_back_days: int = 7, limit: int = 20, end_datetime: str | None = None
) -> str:
    """Formatted macro/market news over the lookback window (vendor interface).

    ``end_datetime`` ("YYYY-MM-DD HH:MM:SS") enforces a pre-market cutoff (see
    ``_visible_premarket``). Live pre-market runs are unaffected (future items
    don't exist yet); on a backfill it prevents post-open leakage.
    """
    end = datetime.strptime(curr_date, "%Y-%m-%d")
    start = end - timedelta(days=look_back_days)
    data = _get(
        "news/general-latest",
        {"from": start.strftime("%Y-%m-%d"), "to": curr_date, "limit": limit},
    )
    if not isinstance(data, list) or not data:
        return f"No FMP general news found around {curr_date}."

    if end_datetime:
        data = [it for it in data if _visible_premarket(it.get("publishedDate", it.get("date", "")), end_datetime)]

    lines = [f"## FMP market news ({start:%Y-%m-%d} to {curr_date})", ""]
    for item in data[:limit]:
        title = item.get("title", "")
        date = item.get("publishedDate", item.get("date", ""))
        site = item.get("site", item.get("publisher", ""))
        text = item.get("text", item.get("snippet", ""))
        lines.append(f"### {date} — {title}")
        if site:
            lines.append(f"Source: {site}")
        if text:
            lines.append(text)
        lines.append("")
    return "\n".join(lines)


def get_economic_calendar(from_date: str, to_date: str, country: str = "US") -> str:
    """Formatted economic release calendar (CPI, NFP, FOMC, etc.). Max 90-day range."""
    params = {"from": from_date, "to": to_date}
    if country:
        params["country"] = country
    data = _get("economic-calendar", params)
    if not isinstance(data, list) or not data:
        return f"No FMP economic calendar events between {from_date} and {to_date}."

    lines = [f"## Economic calendar ({from_date} to {to_date}, {country or 'all'})", ""]
    lines.append("| Date | Event | Actual | Estimate | Previous | Impact |")
    lines.append("|---|---|---|---|---|---|")
    for ev in data:
        lines.append(
            "| {date} | {event} | {actual} | {estimate} | {previous} | {impact} |".format(
                date=ev.get("date", ""),
                event=ev.get("event", ""),
                actual=ev.get("actual", ""),
                estimate=ev.get("estimate", ""),
                previous=ev.get("previous", ""),
                impact=ev.get("impact", ""),
            )
        )
    return "\n".join(lines)
