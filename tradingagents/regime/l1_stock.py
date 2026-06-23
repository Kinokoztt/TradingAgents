"""L1: per-stock direction permit + catalyst confidence.

Scope is news-driven: only names with recent news/catalysts (intersected with
the candidate universe) are analysed — the quiet majority needs no per-name LLM
call. Each name's context is stock news + fundamentals. Names are packed into
multi-ticker batches and the batches run concurrently (thread pool) to keep the
pre-market wall-clock down. See docs/regime-gate-design.md §5.3.
"""

from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta

from pydantic import BaseModel

from tradingagents.market_tools import MarketDataTools, get_market_tools

from .schemas import StockSignal
from .tickers import canonical_ticker, canonicalize_tickers

DEFAULT_STOCK_MODEL = "gemini-3.1-pro-preview"


class _StockSignalBatch(BaseModel):
    signals: list[StockSignal]


def _days_before(date_str: str, days: int) -> str:
    return (datetime.fromisoformat(date_str[:10]) - timedelta(days=days)).strftime("%Y-%m-%d")


def select_news_tickers(
    as_of_date: str,
    *,
    look_back_days: int = 3,
    universe: list[str] | None = None,
    max_tickers: int | None = None,
    market: str = "US",
    tools: MarketDataTools | None = None,
    max_articles: int = 20000,
    news_start: str | None = None,
    news_end: str | None = None,
) -> list[str]:
    """Tickers with recent news, restricted to the candidate universe.

    Ranked by mention frequency (most-covered first). ``max_tickers`` caps the
    list. ``news_start`` (RFC3339 instant or date) overrides the look-back window
    start — pass the previous session's cutoff for gapless incremental scans;
    defaults to ``as_of - look_back_days``. ``news_end`` (RFC3339 instant) caps
    news at a pre-market cutoff; defaults to ``as_of_date`` (whole day).
    ``universe``/``tools`` are injectable for tests.
    """
    tools = tools or get_market_tools(market)
    if universe is None:
        universe = tools.load_candidate_universe()
    universe_set = set(universe)

    start = news_start or _days_before(as_of_date, look_back_days)
    articles = tools.load_news_articles(start, news_end or as_of_date, max_articles=max_articles)

    counts: Counter[str] = Counter()
    for tickers in articles["tickers"]:
        for t in tickers:
            if t in universe_set:
                # merge dual-class siblings (GOOGL→GOOG) so they rank/whitelist once
                counts[canonical_ticker(t)] += 1

    ranked = [t for t, _ in counts.most_common()]
    return ranked[:max_tickers] if max_tickers else ranked


def _render_events(events: list) -> str:
    """Render pre-extracted ``NewsEvent``s for one ticker into a compact typed
    block. Primary events first; ``certainty`` / source tier / ``price_in`` are
    kept so the model can discount already-absorbed or low-reliability items
    instead of re-reading noisy article bodies."""
    if not events:
        return "(no standardized events)"
    primary = [e for e in events if e.is_primary]
    secondary = [e for e in events if not e.is_primary]

    def line(e) -> str:
        d = e.event_date or (e.published_utc[:10] if e.published_utc else "")
        meta = f"certainty={e.certainty.value}, src={e.source_reliability.value}, priced_in={e.price_in.value}"
        return f"- [{d}] {e.event_type.value}/{e.polarity.value} ({meta}): {e.summary}"

    lines = [line(e) for e in primary[:20]]
    if secondary:
        lines.append("(secondary mentions, ticker not the main subject:)")
        lines += [line(e) for e in secondary[:8]]
    return "\n".join(lines)


def _render_catalysts(catalysts: list) -> str:
    """Render structured (deterministic, numeric) catalysts for one ticker.

    Each carries ``age_sessions`` (trading days since its effective date): a
    same-/prior-day catalyst is fresh, an older one is largely digested. We sort
    freshest-first and surface the age so the model weights recency rather than
    treating a week-old earnings the same as today's — the lasting re-rating is
    already reflected in the Fundamentals block."""
    if not catalysts:
        return ""
    rows = sorted(catalysts, key=lambda c: c.get("age_sessions", 0))
    lines = []
    for c in rows[:20]:
        age = c.get("age_sessions")
        age_tag = f"{age}d ago" if age else "today"
        lines.append(f"- [{c.get('effective_date', '')}, {age_tag}] "
                     f"{c.get('catalyst_type', '')}/{c.get('polarity', '')}: {c.get('summary', '')}")
    return "\n".join(lines)


def _gather_context(
    ticker: str,
    as_of_date: str,
    tools: MarketDataTools,
    look_back_days: int,
    with_fundamentals: bool,
    news_end: str,
    events: list | None = None,
    catalysts: list | None = None,
) -> str:
    block = [f"### {ticker}", ""]
    if events is not None:
        # Clean-input mode: feed the standardized event corpus + structured
        # catalysts instead of raw vendor news.
        block += ["#### Standardized news events", _render_events(events)]
        cat = _render_catalysts(catalysts or [])
        if cat:
            block += ["", "#### Structured catalysts", cat]
    else:
        news_start = _days_before(as_of_date, look_back_days)
        news = tools.get_stock_news(ticker, news_start, news_end)
        block += ["#### Recent news", news]
    if with_fundamentals:
        block += ["", "#### Fundamentals", tools.get_fundamentals(ticker, as_of_date)]
    return "\n".join(block)


def _build_batch_prompt(contexts: dict[str, str], as_of_date: str) -> str:
    header = f"""You are an equity catalyst analyst. For each ticker below, as of {as_of_date},
decide a trading `direction` and a `catalyst_confidence` in [0,1]:
- Long: a credible bullish catalyst (beat/raise, upgrade, positive 8-K, M&A).
- Short: a credible bearish catalyst (miss/cut, downgrade, negative 8-K, probe).
- Block: no real catalyst, or untradable risk (halt/bankruptcy/extreme uncertainty).

Confidence anchors: 0.0-0.2 none, 0.2-0.5 weak/ambiguous, 0.5-0.8 clear, 0.8-1.0 strong confirmed.
Context may include standardized typed events (with certainty / source reliability / priced_in) and
structured catalysts (with an age in trading days): weigh Confirmed, high-reliability, not-yet-priced-in,
RECENT catalysts most; discount already-PricedIn or low-reliability items, secondary mentions where the
ticker isn't the subject, and older catalysts (a 5-day-old earnings is largely digested — its lasting
effect already shows in Fundamentals, so don't treat it as a fresh catalyst).
Return exactly one signal per ticker, with the ticker symbol verbatim and a one-line reason.

Tickers and context:
"""
    return header + "\n\n".join(contexts[t] for t in contexts)


def _analyze_batch(
    tickers: list[str],
    as_of_date: str,
    tools: MarketDataTools,
    structured_llm,
    look_back_days: int,
    with_fundamentals: bool,
    news_end: str,
    events_by_ticker: dict[str, list] | None = None,
    catalysts_by_ticker: dict[str, list] | None = None,
) -> list[StockSignal]:
    contexts = {
        t: _gather_context(
            t, as_of_date, tools, look_back_days, with_fundamentals, news_end,
            events=(events_by_ticker.get(t, []) if events_by_ticker is not None else None),
            catalysts=(catalysts_by_ticker.get(t, []) if catalysts_by_ticker is not None else None),
        )
        for t in tickers
    }
    batch: _StockSignalBatch = structured_llm.invoke(_build_batch_prompt(contexts, as_of_date))
    requested = set(tickers)
    return [s for s in batch.signals if s.ticker in requested]


def analyze_stocks(
    tickers: list[str],
    as_of_date: str,
    *,
    market: str = "US",
    tools: MarketDataTools | None = None,
    llm=None,
    provider: str = "google",
    model: str = DEFAULT_STOCK_MODEL,
    base_url: str | None = None,
    batch_size: int = 20,
    max_workers: int = 4,
    look_back_days: int = 7,
    with_fundamentals: bool = True,
    news_end: str | None = None,
    events_by_ticker: dict[str, list] | None = None,
    catalysts_by_ticker: dict[str, list] | None = None,
) -> list[StockSignal]:
    """Analyse ``tickers`` into ``StockSignal``s via concurrent batched LLM calls.

    Tickers are packed into ``batch_size`` groups; batches run on a thread pool
    of ``max_workers``. ``news_end`` (RFC3339 instant) caps per-stock news at a
    pre-market cutoff (defaults to ``as_of_date``). Output preserves input order
    and is deduped by ticker (first wins). ``tools``/``llm`` injectable for tests.

    Clean-input mode: when ``events_by_ticker`` is given, each ticker's context
    is built from its standardized news events (+ ``catalysts_by_ticker``) rather
    than raw vendor news — the events are already pre-market-cut, so ``news_end``
    is unused for those names. Fundamentals are still attached if requested.
    """
    if not tickers:
        return []
    tickers = canonicalize_tickers(tickers)  # collapse dual-class siblings, dedupe
    news_end = news_end or as_of_date

    tools = tools or get_market_tools(market)

    if llm is None:
        from ._llm import build_cascade_llm

        llm = build_cascade_llm(provider, model, base_url)
    structured_llm = llm.with_structured_output(_StockSignalBatch)

    batches = [tickers[i : i + batch_size] for i in range(0, len(tickers), batch_size)]

    def run(batch: list[str]) -> list[StockSignal]:
        return _analyze_batch(batch, as_of_date, tools, structured_llm, look_back_days,
                              with_fundamentals, news_end, events_by_ticker, catalysts_by_ticker)

    by_ticker: dict[str, StockSignal] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for batch_result in pool.map(run, batches):
            for s in batch_result:
                by_ticker.setdefault(s.ticker, s)

    return [by_ticker[t] for t in dict.fromkeys(tickers) if t in by_ticker]
