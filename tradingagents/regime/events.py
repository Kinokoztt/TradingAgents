"""Standardized news-event extraction (the LLM's new job).

Instead of asking the LLM to predict a trading *direction* (which no model has
been shown to do reliably), this layer asks it to do what LLMs are good at:
turn unstructured news into a *standardized, typed event record* — what kind of
event, how certain, how material — with NO price prediction. The structured
output is the standardized corpus that downstream encoding + NN training will
consume (see docs/nn-pipeline-roadmap.md).

Each ``NewsEvent`` keeps its provenance (source publisher, url, timestamp) so
two non-LLM enrichers can run on top:
  - source_reliability.py: tag each event with the publisher's reliability tier.
  - price_in.py: use the timestamp + real prices to label whether the
    information was already priced in (or is just a post-hoc recap).

The extractor mirrors l1_stock.analyze_stocks: per-ticker batched LLM calls run
on a thread pool, structured output via with_structured_output.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from enum import Enum

from pydantic import BaseModel, Field

from tradingagents.dataflows import massive

DEFAULT_EVENT_MODEL = "qwen3-32b"


class EventType(str, Enum):
    """Closed taxonomy of corporate/market events. Extend deliberately —
    the encoder one-hots this, so new members change the feature space."""

    EARNINGS_BEAT = "EarningsBeat"
    EARNINGS_MISS = "EarningsMiss"
    EARNINGS_INLINE = "EarningsInline"
    GUIDANCE_RAISE = "GuidanceRaise"
    GUIDANCE_CUT = "GuidanceCut"
    ANALYST_UPGRADE = "AnalystUpgrade"
    ANALYST_DOWNGRADE = "AnalystDowngrade"
    PRICE_TARGET_CHANGE = "PriceTargetChange"
    MNA = "MnA"
    PARTNERSHIP = "Partnership"
    PRODUCT_LAUNCH = "ProductLaunch"
    REGULATORY_APPROVAL = "RegulatoryApproval"
    REGULATORY_PROBE = "RegulatoryProbe"
    LITIGATION = "Litigation"
    SETTLEMENT = "Settlement"
    MANAGEMENT_CHANGE = "ManagementChange"
    INSIDER_TRANSACTION = "InsiderTransaction"
    DIVIDEND = "Dividend"
    BUYBACK = "Buyback"
    OFFERING = "Offering"
    MACRO = "Macro"
    OTHER = "Other"


class Certainty(str, Enum):
    """How firm the information is — distinct from how material it is."""

    RUMORED = "Rumored"      # speculation / unconfirmed report
    REPORTED = "Reported"    # media reporting, not company-confirmed
    CONFIRMED = "Confirmed"  # company/counterparty confirmed
    OFFICIAL = "Official"    # official filing / press release / regulator


class Polarity(str, Enum):
    """Sentiment toward the stock — NOT a trade direction or price call."""

    POSITIVE = "Positive"
    NEGATIVE = "Negative"
    NEUTRAL = "Neutral"
    MIXED = "Mixed"


class Materiality(str, Enum):
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"


class Horizon(str, Enum):
    IMMEDIATE = "Immediate"    # impact plays out intraday / next session
    SHORT_TERM = "ShortTerm"   # days to weeks
    LONG_TERM = "LongTerm"     # quarters+


class SourceReliability(str, Enum):
    """Publisher reliability tier (set by source_reliability.py, not the LLM)."""

    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"
    UNKNOWN = "Unknown"


class PriceInStatus(str, Enum):
    """Whether the market had already absorbed the info (set by price_in.py)."""

    NOT_PRICED_IN = "NotPricedIn"  # little pre-move, clear post-publication move
    PARTIAL = "Partial"            # some pre-move, some residual reaction
    PRICED_IN = "PricedIn"         # move happened before/at publication
    POST_HOC = "PostHoc"           # news merely recaps an already-completed move
    UNKNOWN = "Unknown"            # no/insufficient price data to judge


class ExtractedEvent(BaseModel):
    """One event as classified by the LLM from a single article.

    ``article_index`` ties the event back to the source article so the
    non-LLM enrichers can recover its timestamp/publisher/url. The LLM only
    fills the classification fields; provenance is attached by code.
    """

    article_index: int = Field(description="Index of the source article in the provided list")
    event_type: EventType
    certainty: Certainty
    polarity: Polarity
    materiality: Materiality
    horizon: Horizon
    summary: str = Field(description="One-line standardized summary of the event (no price prediction)")
    evidence: str = Field(default="", description="Short verbatim snippet from the article supporting the classification")


class StockEventExtraction(BaseModel):
    """LLM structured-output container: all events found across the batch."""

    events: list[ExtractedEvent] = Field(default_factory=list)


class NewsEvent(BaseModel):
    """A standardized event record: LLM classification + provenance + enrichment.

    This is the row written to events.jsonl and later encoded for NN training.
    ``source_reliability`` and the ``price_in``/return fields default to the
    Unknown/None state and are populated by the enrichers.
    """

    ticker: str
    as_of_date: str
    event_type: EventType
    certainty: Certainty
    polarity: Polarity
    materiality: Materiality
    horizon: Horizon
    summary: str
    evidence: str = ""

    # provenance (from the source article)
    source: str = ""
    article_url: str = ""
    published_utc: str = ""
    event_date: str = ""

    # enrichment (non-LLM)
    source_reliability: SourceReliability = SourceReliability.UNKNOWN
    price_in: PriceInStatus = PriceInStatus.UNKNOWN
    pre_return: float | None = None
    post_return: float | None = None
    pre_volume_ratio: float | None = None
    post_volume_ratio: float | None = None


def _days_before(date_str: str, days: int) -> str:
    return (datetime.fromisoformat(date_str[:10]) - timedelta(days=days)).strftime("%Y-%m-%d")


def _format_article(index: int, article: dict, ticker: str) -> str:
    """Render one article for the prompt. Uses only vendor-provided summary
    fields (title + description + insights) — we do NOT fetch full bodies."""
    lines = [f"[{index}] {article.get('date', '')} — {article.get('title', '')}"]
    publisher = article.get("publisher")
    if publisher:
        lines.append(f"Source: {publisher}")
    description = article.get("description")
    if description:
        lines.append(description)
    sentiment = _vendor_sentiment(article.get("insights", []), ticker)
    if sentiment:
        lines.append(f"Vendor sentiment: {sentiment}")
    return "\n".join(lines)


def _vendor_sentiment(insights: list[dict], ticker: str) -> str:
    for insight in insights:
        if insight.get("ticker", "").upper() == ticker.upper():
            return f"{insight.get('sentiment', '')} — {insight.get('sentiment_reasoning', '')}".strip(" —")
    return ""


def _build_prompt(ticker: str, articles: list[dict]) -> str:
    article_block = "\n\n".join(_format_article(i, a, ticker) for i, a in enumerate(articles))
    return f"""You are a financial news classifier. Extract the discrete, material EVENTS for {ticker}
from the articles below. This is a CLASSIFICATION task — do NOT predict price moves, do NOT
recommend buy/sell, do NOT guess direction of trading.

For each distinct event you find, return one record with:
- article_index: the [N] index of the source article
- event_type: one of the closed taxonomy values
- certainty: Rumored / Reported / Confirmed / Official
- polarity: sentiment toward the company (Positive/Negative/Neutral/Mixed) — this is sentiment, NOT a price call
- materiality: High / Medium / Low (expected importance to the company)
- horizon: Immediate / ShortTerm / LongTerm
- summary: one neutral line describing what happened (no forecasting)
- evidence: a short verbatim snippet supporting the classification

Rules:
- Multiple articles about the SAME event -> still emit per-article records (dedup happens later), but keep them faithful.
- Pure market-recap articles ("stock rose 5% today on heavy volume") with no underlying catalyst: classify as Macro/Other with Low materiality.
- If an article has no material event, skip it.

Articles for {ticker}:
{article_block}
"""


def _extract_for_ticker(ticker: str, articles: list[dict], as_of_date: str, structured_llm) -> list[NewsEvent]:
    if not articles:
        return []
    extraction: StockEventExtraction = structured_llm.invoke(_build_prompt(ticker, articles))
    events: list[NewsEvent] = []
    for ev in extraction.events:
        if ev.article_index < 0 or ev.article_index >= len(articles):
            continue  # LLM referenced a non-existent article; drop rather than guess
        src = articles[ev.article_index]
        events.append(
            NewsEvent(
                ticker=ticker,
                as_of_date=as_of_date,
                event_type=ev.event_type,
                certainty=ev.certainty,
                polarity=ev.polarity,
                materiality=ev.materiality,
                horizon=ev.horizon,
                summary=ev.summary,
                evidence=ev.evidence,
                source=src.get("publisher", ""),
                article_url=src.get("article_url", ""),
                published_utc=src.get("published_utc", ""),
                event_date=src.get("date", "") or "",
            )
        )
    return events


def extract_events(
    tickers: list[str],
    as_of_date: str,
    *,
    llm=None,
    provider: str = "vllm",
    model: str = DEFAULT_EVENT_MODEL,
    base_url: str | None = None,
    look_back_days: int = 7,
    news_end: str | None = None,
    max_articles_per_ticker: int = 50,
    max_workers: int = 4,
) -> list[NewsEvent]:
    """Extract standardized ``NewsEvent``s for ``tickers`` as of ``as_of_date``.

    Per-ticker articles are fetched from Massive (structured: timestamp +
    publisher + url preserved), classified by the LLM, and enriched later by
    source_reliability/price_in. ``llm`` is injectable for tests; otherwise a
    client is built for ``provider``/``model``/``base_url`` (defaults to the
    self-hosted vLLM Qwen).
    """
    if not tickers:
        return []
    news_start = _days_before(as_of_date, look_back_days)
    news_end = news_end or as_of_date

    if llm is None:
        from tradingagents.llm_clients import create_llm_client

        llm = create_llm_client(provider, model, base_url=base_url).get_llm()
    structured_llm = llm.with_structured_output(StockEventExtraction)

    def run(ticker: str) -> list[NewsEvent]:
        articles = massive.fetch_news_articles(
            news_start, news_end, ticker=ticker, max_articles=max_articles_per_ticker
        )
        return _extract_for_ticker(ticker, articles, as_of_date, structured_llm)

    results: list[NewsEvent] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for events in pool.map(run, tickers):
            results.extend(events)
    return results
