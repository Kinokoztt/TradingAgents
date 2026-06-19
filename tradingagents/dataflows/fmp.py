"""Financial Modeling Prep (FMP) vendor — macro/market side of the regime gate.

Scope (Starter plan is sufficient): broad/macro market news via
``get_global_news`` and the economic release calendar via
``get_economic_calendar``. Per-ticker news and corporate catalysts come from
Massive instead. Auth is the ``apikey`` query parameter from FMP_API_KEY.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import requests

from . import _http

API_BASE_URL = "https://financialmodelingprep.com/stable"

_ET = ZoneInfo("America/New_York")
_UTC = ZoneInfo("UTC")
_NEWS_PAGE_LIMIT = 250


def get_api_key() -> str:
    api_key = os.getenv("FMP_API_KEY")
    if not api_key:
        raise ValueError("FMP_API_KEY environment variable is not set.")
    return api_key


def _get(endpoint: str, params: dict):
    params = {**params, "apikey": get_api_key()}
    response = _http.get_with_retry(requests.get, f"{API_BASE_URL}/{endpoint}", params=params, timeout=30)
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


def _news_dt_to_utc(published: str) -> str:
    """Normalise an FMP ``news/stock`` ``publishedDate`` to an RFC3339 UTC instant.

    FMP returns ``YYYY-MM-DD HH:MM:SS`` *without* a timezone for the stock-news
    endpoint; these timestamps are US Eastern. The rest of the pipeline (price-in
    after-hours boundary, pre-market cutoff) assumes UTC, so convert ET->UTC here.
    Values that already carry a 'T'/'Z' (other FMP endpoints) pass through.
    """
    if not published:
        return ""
    if "T" in published:
        return published
    dt = datetime.strptime(published, "%Y-%m-%d %H:%M:%S").replace(tzinfo=_ET)
    return dt.astimezone(_UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def fetch_stock_news(
    symbol: str,
    start_date: str,
    end_date: str,
    max_articles: int = 200,
) -> list[dict]:
    """Per-ticker news as normalized article dicts (shape matches massive's).

    Returns dicts with ``date, title, publisher, description, insights,
    published_utc, article_url, tickers`` so the event extractor can consume FMP
    and Massive interchangeably. ``start_date``/``end_date`` may be dates or
    RFC3339 instants — FMP filters by date, so the caller should additionally
    clip to a precise pre-market cutoff. Paginates via ``page`` until exhausted
    or ``max_articles`` is reached. FMP uses hyphen share-class tickers (BRK-B).
    """
    sym = symbol.replace(".", "-")
    articles: list[dict] = []
    page = 0
    while len(articles) < max_articles:
        data = _get(
            "news/stock",
            {"symbols": sym, "from": start_date[:10], "to": end_date[:10],
             "page": page, "limit": _NEWS_PAGE_LIMIT},
        )
        if not isinstance(data, list) or not data:
            break
        for it in data:
            pub = _news_dt_to_utc(it.get("publishedDate", ""))
            articles.append({
                "date": (pub or it.get("publishedDate", ""))[:10],
                "title": it.get("title", ""),
                "publisher": it.get("publisher", ""),
                "description": it.get("text", ""),
                "insights": [],
                "published_utc": pub,
                "article_url": it.get("url", ""),
                "tickers": [symbol],
            })
            if len(articles) >= max_articles:
                return articles
        if len(data) < _NEWS_PAGE_LIMIT:
            break
        page += 1
    return articles


# --- structured catalyst feeds (line-1: no LLM, deterministic) ---------------
# earnings/grades/dividends return deep history in one call (caller filters by
# date); price-target-news and mergers are "latest" feeds -> paginate, stopping
# once results predate ``stop_before`` (results are newest-first).

def fetch_earnings(symbol: str, limit: int = 120) -> list[dict]:
    """Earnings calendar rows (epsActual/epsEstimated/revenue..., date)."""
    return _get("earnings", {"symbol": symbol.replace(".", "-"), "limit": limit}) or []


def fetch_grades(symbol: str, limit: int = 1000) -> list[dict]:
    """Individual analyst grade actions (gradingCompany, prev->new, action)."""
    return _get("grades", {"symbol": symbol.replace(".", "-"), "limit": limit}) or []


def fetch_dividends(symbol: str, limit: int = 60) -> list[dict]:
    """Dividend history (declarationDate, dividend, yield, frequency, ...)."""
    return _get("dividends", {"symbol": symbol.replace(".", "-"), "limit": limit}) or []


def fetch_price_target_news(symbol: str, stop_before: str | None = None, max_items: int = 500) -> list[dict]:
    """Analyst price-target change items (priceTarget, priceWhenPosted, title)."""
    sym = symbol.replace(".", "-")
    items: list[dict] = []
    page = 0
    while len(items) < max_items:
        data = _get("price-target-news", {"symbol": sym, "page": page, "limit": 100})
        if not isinstance(data, list) or not data:
            break
        items.extend(data)
        if stop_before and data[-1].get("publishedDate", "")[:10] < stop_before:
            break
        if len(data) < 100:
            break
        page += 1
    return items


def fetch_mergers(stop_before: str | None = None, max_items: int = 3000) -> list[dict]:
    """Market-wide M&A deals (acquirer symbol, targetedSymbol, transactionDate)."""
    items: list[dict] = []
    page = 0
    while len(items) < max_items:
        data = _get("mergers-acquisitions-latest", {"page": page, "limit": 100})
        if not isinstance(data, list) or not data:
            break
        items.extend(data)
        if stop_before and data[-1].get("transactionDate", "")[:10] < stop_before:
            break
        if len(data) < 100:
            break
        page += 1
    return items


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


def get_economic_calendar(
    from_date: str, to_date: str, country: str = "US", cutoff: str | None = None
) -> str:
    """Formatted economic release calendar (CPI, NFP, FOMC, etc.). Max 90-day range.

    ``cutoff`` ("YYYY-MM-DD HH:MM:SS", pre-market) makes the calendar point-in-time:
    FMP's calendar is a *live* table, so on a rollback it back-fills the ``Actual``
    column for events that have since occurred — which would leak future macro
    prints (the very thing being forecast) into the model. With ``cutoff`` set, the
    ``Actual`` of any event timed after the cutoff is blanked; ``Estimate``/
    ``Previous`` (known ex-ante) are always kept. Live pre-market runs are
    unaffected (those actuals are null anyway).
    """
    params = {"from": from_date, "to": to_date}
    if country:
        params["country"] = country
    data = _get("economic-calendar", params)
    if not isinstance(data, list) or not data:
        return f"No FMP economic calendar events between {from_date} and {to_date}."

    note = " (Actual hidden for events after the pre-market cutoff)" if cutoff else ""
    lines = [f"## Economic calendar ({from_date} to {to_date}, {country or 'all'}){note}", ""]
    lines.append("| Date | Event | Actual | Estimate | Previous | Impact |")
    lines.append("|---|---|---|---|---|---|")
    for ev in data:
        # Keep the realized Actual only if the release is visible at the cutoff;
        # otherwise it hasn't happened yet (point-in-time) -> blank it.
        visible = cutoff is None or _visible_premarket(ev.get("date", ""), cutoff)
        lines.append(
            "| {date} | {event} | {actual} | {estimate} | {previous} | {impact} |".format(
                date=ev.get("date", ""),
                event=ev.get("event", ""),
                actual=(ev.get("actual", "") if visible else ""),
                estimate=ev.get("estimate", ""),
                previous=ev.get("previous", ""),
                impact=ev.get("impact", ""),
            )
        )
    return "\n".join(lines)


def get_financial_statements(symbol: str, period: str = "quarter", limit: int = 40) -> dict:
    """Raw FMP income/balance/cash-flow statements, most recent period first.

    Returns ``{"income": [...], "balance": [...], "cashflow": [...]}`` where each
    list element is a period dict carrying ``date``/``filingDate``/``acceptedDate``
    plus line items (revenue, netIncome, eps, totalStockholdersEquity, ...).

    Point-in-time is the caller's job: keep only periods whose ``acceptedDate``
    (SEC acceptance datetime) is on/before the session's pre-market cutoff. We
    fetch a deep window (``limit`` quarters ~= 10y) so a rolled-back session still
    finds the quarters that were actually public back then, not just the latest.
    """
    # FMP uses a hyphen for share-class tickers (BRK-B), while our universe/BQ/
    # Massive use a dot (BRK.B). Passing the dotted form returns a misleading
    # 402 "Special Endpoint ... not available under your subscription". Normalize
    # only here; the dotted ticker stays canonical everywhere else (e.g. BQ).
    sym = symbol.replace(".", "-")
    return {
        "income": _get("income-statement", {"symbol": sym, "period": period, "limit": limit}) or [],
        "balance": _get("balance-sheet-statement", {"symbol": sym, "period": period, "limit": limit}) or [],
        "cashflow": _get("cash-flow-statement", {"symbol": sym, "period": period, "limit": limit}) or [],
    }
