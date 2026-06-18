"""Standardized news-event extraction (the LLM's new job).

Instead of asking the LLM to predict a trading *direction* (which no model has
been shown to do reliably), this layer asks it to do what LLMs are good at:
turn unstructured news into a *standardized, typed event record* — what kind of
event, is it about this company, is it confirmed — with NO price prediction. The
structured output is the corpus that downstream encoding + NN training consumes
(see docs/nn-pipeline-roadmap.md).

Two-stage extraction (decomposed for reliability — one big "judge everything at
once" call degenerates into default labels):
  - Stage 1 (reading): per-ticker, read the raw articles and emit discrete
    events with ``is_primary`` (is the ticker the subject), ``certainty`` and a
    clean one-line ``summary``. Non-events are skipped here.
  - Stage 2 (classification): classify each clean summary into ``event_type``
    (reduced taxonomy) + ``polarity``. Classifying a one-liner is far more
    consistent than classifying a noisy article.

Each ``NewsEvent`` keeps its provenance (publisher, url, timestamp) so two
non-LLM enrichers run on top: source_reliability.py (publisher tier) and
price_in.py (was it already priced in, via real prices).
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from enum import Enum

from pydantic import BaseModel, Field

from tradingagents.dataflows import massive

DEFAULT_EVENT_MODEL = "qwen3-32b"


class EventType(str, Enum):
    """Reduced, high-signal taxonomy. The encoder one-hots this, so keep it
    small and well-populated; direction (beat/miss, up/down, approve/probe) is
    carried by ``polarity`` rather than separate members."""

    EARNINGS = "Earnings"            # results: beat/miss/inline (polarity = sign)
    GUIDANCE = "Guidance"            # outlook raise/cut (polarity = sign)
    ANALYST_ACTION = "AnalystAction"  # upgrade/downgrade/price-target change
    MNA = "MnA"                      # mergers & acquisitions, takeovers
    PARTNERSHIP = "Partnership"      # deals, JVs, strategic investments, supply
    PRODUCT = "Product"             # product / technology / platform launches
    REGULATORY = "Regulatory"        # approvals / probes / policy actions
    LEGAL = "Legal"                 # litigation / settlements
    CAPITAL = "Capital"             # dividend / buyback / offering / raise
    GOVERNANCE = "Governance"        # management change / insider transactions
    MACRO = "Macro"                 # sector / market backdrop
    OTHER = "Other"                 # no discrete company event (opinion/recap)


class Certainty(str, Enum):
    """How firm the information is. Binary on purpose: a finer scale is not
    reliably separable by the model and adds noise rather than NN signal."""

    CONFIRMED = "Confirmed"      # company/counterparty/regulator confirmed, official
                                 # filing/release, OR a concretely announced deal
    UNCONFIRMED = "Unconfirmed"  # media report w/o confirmation, opinion, forecast, rumor


class Polarity(str, Enum):
    """Sentiment toward the stock — NOT a trade direction or price call."""

    POSITIVE = "Positive"
    NEGATIVE = "Negative"
    NEUTRAL = "Neutral"
    MIXED = "Mixed"


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


class Stage1Event(BaseModel):
    """Stage-1 (reading) output: one discrete event found in one article."""

    article_index: int = Field(description="Index [N] of the source article")
    is_primary: bool = Field(description="True if the ticker is the MAIN subject/driver of this event (not merely mentioned, compared, or held)")
    certainty: Certainty
    summary: str = Field(description="One neutral line describing what happened (no forecast, no price call)")


class Stage1Extraction(BaseModel):
    events: list[Stage1Event] = Field(default_factory=list)


class Stage2Label(BaseModel):
    """Stage-2 (classification) output: type + polarity for one summary."""

    index: int = Field(description="Index of the summary in the provided list")
    event_type: EventType
    polarity: Polarity


class Stage2Labels(BaseModel):
    labels: list[Stage2Label] = Field(default_factory=list)


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
    is_primary: bool = True
    summary: str = ""

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


def _vendor_sentiment(insights: list[dict], ticker: str) -> str:
    for insight in insights:
        if insight.get("ticker", "").upper() == ticker.upper():
            return f"{insight.get('sentiment', '')} — {insight.get('sentiment_reasoning', '')}".strip(" —")
    return ""


def _format_article(index: int, article: dict, ticker: str) -> str:
    """Render one article for the stage-1 prompt. Vendor summary fields only
    (title + description + insights) — we do NOT fetch full bodies."""
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


def _build_stage1_prompt(ticker: str, articles: list[dict]) -> str:
    article_block = "\n\n".join(_format_article(i, a, ticker) for i, a in enumerate(articles))
    return f"""You are a financial news reader. From the articles below, extract the discrete,
material EVENTS that concern {ticker}. This is reading + extraction — do NOT predict prices,
do NOT recommend buy/sell, do NOT guess trading direction.

For each distinct event emit one record:
- article_index: the [N] index of the source article
- is_primary: true ONLY if {ticker} is the MAIN subject/driver of the event. Set false if
  {ticker} is merely mentioned, compared to a peer, listed as an ETF/fund holding, or is a
  secondary beneficiary. (Still emit it, just mark is_primary=false.)
- certainty: "Confirmed" if the company/counterparty/regulator confirmed it, it's an official
  filing/press release, OR it's a concretely announced transaction (named parties + terms/amount).
  Otherwise "Unconfirmed" (media report without confirmation, analyst opinion/estimate, forecast,
  prediction, rumor, speculation).
- summary: ONE neutral sentence stating what happened — factual, no forecasting, no price call.

Rules:
- SKIP articles with no discrete event: pure opinion/valuation pieces ("is X a buy?", "X looks
  cheap"), generic stock-price recaps ("X rose 5% on volume"), and ETF/fund composition notes.
  Do not invent events for them.
- Multiple articles about the SAME event -> emit one record per article (dedup happens later).

Articles for {ticker}:
{article_block}
"""


_EVENT_TYPE_GUIDE = """- Earnings: quarterly/annual results (beat/miss/inline)
- Guidance: forward outlook raised or cut
- AnalystAction: analyst upgrade/downgrade or price-target change
- MnA: merger, acquisition, takeover, divestiture
- Partnership: commercial deal, JV, strategic investment, supply/customer agreement
- Product: product / technology / platform launch or major capability announcement
- Regulatory: regulatory approval, probe/investigation, policy/government action
- Legal: lawsuit, litigation, settlement
- Capital: dividend, buyback, share offering, capital raise
- Governance: executive/board change, insider buying/selling
- Macro: sector- or market-level backdrop affecting the name
- Other: anything without a clear discrete company event"""


def _build_stage2_prompt(ticker: str, summaries: list[str]) -> str:
    listing = "\n".join(f"[{i}] {s}" for i, s in enumerate(summaries))
    return f"""Classify each one-line event about {ticker}. For each, return its index plus:
- event_type: exactly one of the taxonomy below
- polarity: sentiment toward {ticker} (Positive/Negative/Neutral/Mixed) — sentiment, NOT a price call

event_type taxonomy:
{_EVENT_TYPE_GUIDE}

Notes:
- The taxonomy is direction-agnostic; an earnings miss is still "Earnings" (polarity Negative),
  an analyst downgrade is "AnalystAction" (polarity Negative).
- If no taxonomy member fits, use "Other".

Events:
{listing}
"""


def build_event_llms(
    *,
    llm=None,
    provider: str = "vllm",
    model: str = DEFAULT_EVENT_MODEL,
    base_url: str | None = None,
    timeout: float = 180.0,
    max_tokens: int = 4096,
    max_retries: int = 1,
):
    """Build the two structured LLMs (stage-1 reader, stage-2 classifier).

    ``llm`` is injectable for tests; otherwise a client is built for
    ``provider``/``model``/``base_url`` (defaults to the self-hosted vLLM Qwen).
    Both stages share the same underlying model.

    ``timeout``/``max_tokens`` are set on purpose: without them a single request
    can hang forever (no timeout) or generate up to the context limit (the event
    list is unbounded), which stalls the whole pool. Qwen thinking is disabled so
    no reasoning tokens are spent before the structured JSON.
    """
    if llm is None:
        from tradingagents.llm_clients import create_llm_client

        client_kwargs: dict = {"timeout": timeout, "max_tokens": max_tokens, "max_retries": max_retries}
        if "qwen" in model.lower():
            client_kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
        llm = create_llm_client(provider, model, base_url=base_url, **client_kwargs).get_llm()
    return llm.with_structured_output(Stage1Extraction), llm.with_structured_output(Stage2Labels)


def extract_ticker_events(
    ticker: str,
    as_of_date: str,
    articles: list[dict],
    stage1_llm,
    stage2_llm,
) -> list[NewsEvent]:
    """Two-stage extraction for ONE ticker's articles. Pure given the LLMs."""
    if not articles:
        return []

    stage1: Stage1Extraction = stage1_llm.invoke(_build_stage1_prompt(ticker, articles))
    rows: list[tuple[Stage1Event, dict]] = []
    for ev in stage1.events:
        if 0 <= ev.article_index < len(articles):
            rows.append((ev, articles[ev.article_index]))
    if not rows:
        return []

    summaries = [ev.summary for ev, _ in rows]
    stage2: Stage2Labels = stage2_llm.invoke(_build_stage2_prompt(ticker, summaries))
    labels = {lab.index: lab for lab in stage2.labels}

    events: list[NewsEvent] = []
    for i, (s1, art) in enumerate(rows):
        lab = labels.get(i)
        if lab is None:
            continue  # stage-2 dropped this summary; skip rather than guess a type
        events.append(
            NewsEvent(
                ticker=ticker,
                as_of_date=as_of_date,
                event_type=lab.event_type,
                certainty=s1.certainty,
                polarity=lab.polarity,
                is_primary=s1.is_primary,
                summary=s1.summary,
                source=art.get("publisher", ""),
                article_url=art.get("article_url", ""),
                published_utc=art.get("published_utc", ""),
                event_date=art.get("date", "") or "",
            )
        )
    return events


def fetch_ticker_articles(
    ticker: str,
    as_of_date: str,
    *,
    look_back_days: int = 7,
    news_start: str | None = None,
    news_end: str | None = None,
    max_articles_per_ticker: int = 50,
) -> list[dict]:
    """Fetch one ticker's articles for the extraction window (structured).

    ``news_start`` (RFC3339 instant/date) overrides the window start; pass the
    previous session's cutoff for a gapless, non-overlapping incremental window.
    Defaults to ``as_of - look_back_days``.
    """
    start = news_start or _days_before(as_of_date, look_back_days)
    return massive.fetch_news_articles(
        start, news_end or as_of_date, ticker=ticker, max_articles=max_articles_per_ticker
    )


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
    """Extract standardized ``NewsEvent``s for ``tickers`` (batch convenience).

    Per-ticker two-stage extraction runs on a thread pool. For incremental
    progress / resume, callers should instead drive ``extract_ticker_events``
    per ticker (see scripts/extract_events.py).
    """
    if not tickers:
        return []
    stage1_llm, stage2_llm = build_event_llms(llm=llm, provider=provider, model=model, base_url=base_url)

    def run(ticker: str) -> list[NewsEvent]:
        articles = fetch_ticker_articles(
            ticker, as_of_date, look_back_days=look_back_days, news_end=news_end,
            max_articles_per_ticker=max_articles_per_ticker,
        )
        return extract_ticker_events(ticker, as_of_date, articles, stage1_llm, stage2_llm)

    results: list[NewsEvent] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for events in pool.map(run, tickers):
            results.extend(events)
    return results
