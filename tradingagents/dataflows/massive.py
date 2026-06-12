"""Massive (formerly Polygon.io) news vendor.

Two consumers share this module:

1. The regime gate / analysts, via ``get_news`` — returns a formatted text
   block matching the other news vendors (see news_data_tools.get_news).
2. The concept graph's co-mention builder, via ``fetch_news_articles`` —
   returns structured article dicts (date + tickers[]) so co-occurrence can
   be counted.

Auth is a Bearer token from MASSIVE_API_KEY. The news endpoint returns at
most 1000 results per page and paginates via ``next_url``.
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Optional

import requests

from . import _http

API_BASE_URL = "https://api.massive.com"
NEWS_ENDPOINT = "/v2/reference/news"
SPLITS_ENDPOINT = "/v3/reference/splits"
_PAGE_LIMIT = 1000

# Long backfills paginate market-wide news for hundreds of dates, so a single
# multi-minute TLS/connection blip would otherwise abort the whole run. Give
# Massive GETs a bigger retry budget than the shared default: 8 retries with
# backoff capped at 60s ~ a few minutes of outage tolerance before failing loudly.
_RETRIES = 8
_MAX_BACKOFF = 60.0


def get_api_key() -> str:
    api_key = os.getenv("MASSIVE_API_KEY")
    if not api_key:
        raise ValueError("MASSIVE_API_KEY environment variable is not set.")
    return api_key


def _to_rfc3339(date_str: str, end_of_day: bool = False) -> str:
    """Normalise a YYYY-MM-DD date to an RFC3339 instant; pass through if already one."""
    if "T" in date_str:
        return date_str
    suffix = "T23:59:59Z" if end_of_day else "T00:00:00Z"
    return f"{date_str}{suffix}"


def _get(url: str, params: Optional[dict] = None) -> dict:
    headers = {"Authorization": f"Bearer {get_api_key()}"}
    response = _http.get_with_retry(
        requests.get, url, params=params, headers=headers, timeout=30,
        retries=_RETRIES, max_backoff=_MAX_BACKOFF,
    )
    return response.json()


def _normalize_article(raw: dict) -> dict:
    published = raw.get("published_utc", "")
    return {
        "id": raw.get("id"),
        "published_utc": published,
        "date": published[:10] if published else None,
        "tickers": raw.get("tickers", []) or [],
        "title": raw.get("title", ""),
        "publisher": (raw.get("publisher") or {}).get("name", ""),
        "keywords": raw.get("keywords", []) or [],
        "description": raw.get("description", ""),
        "insights": raw.get("insights", []) or [],
        "article_url": raw.get("article_url", ""),
    }


def fetch_news_articles(
    start_date: str,
    end_date: str,
    ticker: Optional[str] = None,
    max_articles: int = 2000,
) -> list[dict]:
    """Fetch normalized articles in [start_date, end_date] (inclusive).

    When ``ticker`` is None, returns market-wide news so the co-mention
    builder can observe which tickers co-occur. Paginates via ``next_url``
    until exhausted or ``max_articles`` is reached.
    """
    params = {
        "order": "desc",
        "sort": "published_utc",
        "limit": _PAGE_LIMIT,
        "published_utc.gte": _to_rfc3339(start_date),
        "published_utc.lte": _to_rfc3339(end_date, end_of_day=True),
    }
    if ticker:
        params["ticker"] = ticker

    articles: list[dict] = []
    payload = _get(f"{API_BASE_URL}{NEWS_ENDPOINT}", params)

    while True:
        for raw in payload.get("results", []):
            articles.append(_normalize_article(raw))
            if len(articles) >= max_articles:
                return articles
        next_url = payload.get("next_url")
        if not next_url:
            return articles
        payload = _get(next_url)


def fetch_splits(
    start_date: str,
    end_date: str,
    ticker: Optional[str] = None,
    max_records: int = 50000,
) -> list[dict]:
    """Fetch stock splits with execution_date in [start_date, end_date].

    Returns dicts with ``ticker, execution_date, split_from, split_to`` — the
    shape compute_returns expects for split correction. ``execution_date`` is a
    plain YYYY-MM-DD date (Massive/Polygon takes it directly). Market-wide when
    ``ticker`` is None; paginates via ``next_url``.
    """
    params = {
        "order": "asc",
        "sort": "execution_date",
        "limit": _PAGE_LIMIT,
        "execution_date.gte": start_date,
        "execution_date.lte": end_date,
    }
    if ticker:
        params["ticker"] = ticker

    records: list[dict] = []
    payload = _get(f"{API_BASE_URL}{SPLITS_ENDPOINT}", params)
    while True:
        for raw in payload.get("results", []):
            records.append(
                {
                    "ticker": raw["ticker"],
                    "execution_date": raw["execution_date"],
                    "split_from": raw["split_from"],
                    "split_to": raw["split_to"],
                }
            )
            if len(records) >= max_records:
                return records
        next_url = payload.get("next_url")
        if not next_url:
            return records
        payload = _get(next_url)


def get_news(ticker: str, start_date: str, end_date: str) -> str:
    """Formatted ticker news block for analyst prompts (vendor interface)."""
    articles = fetch_news_articles(start_date, end_date, ticker=ticker, max_articles=50)
    if not articles:
        return f"No Massive news found for {ticker} between {start_date} and {end_date}."

    lines = [f"## Massive news for {ticker} ({start_date} to {end_date})", ""]
    for a in articles:
        lines.append(f"### {a['date']} — {a['title']}")
        if a["publisher"]:
            lines.append(f"Source: {a['publisher']}")
        sentiment = _summarize_sentiment(a["insights"], ticker)
        if sentiment:
            lines.append(f"Sentiment: {sentiment}")
        if a["description"]:
            lines.append(a["description"])
        lines.append("")
    return "\n".join(lines)


def _summarize_sentiment(insights: list[dict], ticker: str) -> str:
    """Pull this ticker's sentiment + reasoning out of the insights array."""
    for insight in insights:
        if insight.get("ticker", "").upper() == ticker.upper():
            sentiment = insight.get("sentiment", "")
            reasoning = insight.get("sentiment_reasoning", "")
            return f"{sentiment} — {reasoning}".strip(" —")
    return ""
