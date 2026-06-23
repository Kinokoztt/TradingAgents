"""Commander: the bottom-up hierarchical cascade (S0→S4).

Linear pipeline (no branches/loops), so it's plain sequential orchestration
rather than a LangGraph: per-stock signals roll up to theme clusters, then
sectors, then the market verdict + circuit breaker. Each stage only fires the
LLM where the layer below produced activity. See docs/regime-gate-design.md §5.3.
"""

from __future__ import annotations

from datetime import date, datetime, time
from zoneinfo import ZoneInfo

from tradingagents.market_tools import MarketDataTools, get_market_tools

from .l1_stock import analyze_stocks, select_news_tickers
from .l2_concept import aggregate_concepts, judge_clusters, judge_sectors, propagate_catalysts
from .l3_regime import analyze_regime
from .schemas import RegimeReport

_ET = ZoneInfo("America/New_York")
_UTC = ZoneInfo("UTC")


def premarket_cutoffs(session_date: str, hour: int = 9, minute: int = 0) -> tuple[str, str]:
    """Pre-market news cutoff for ``session_date`` (default 09:00 ET).

    09:00 (not the 09:30 bell) so the morning macro prints (08:30 ET CPI/NFP) are
    captured without crowding the open. Returns ``(utc_instant, et_wallclock)``:
    an RFC3339 ``...Z`` instant for the Massive paths (precise lte) and a
    ``YYYY-MM-DD HH:MM:SS`` wall-clock for the FMP filter. News after the cutoff
    is excluded.
    """
    et_dt = datetime.combine(date.fromisoformat(session_date[:10]), time(hour, minute), tzinfo=_ET)
    utc_instant = et_dt.astimezone(_UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    return utc_instant, et_dt.strftime("%Y-%m-%d %H:%M:%S")


def run_regime_gate(
    as_of_date: str,
    *,
    market: str = "US",
    tools: MarketDataTools | None = None,
    llm=None,
    provider: str = "google",
    base_url: str | None = None,
    model: str | None = None,
    l1_model: str = "gemini-3-flash-preview",
    concept_model: str = "gemini-3.1-pro-preview",
    regime_model: str = "gemini-3.1-pro-preview",
    universe: list[str] | None = None,
    news_tickers: list[str] | None = None,
    out_dir: str | None = None,
    news_look_back_days: int = 3,
    max_news_tickers: int | None = None,
    batch_size: int = 20,
    max_workers: int = 6,
    with_fundamentals: bool = True,
    propagate: bool = True,
    use_llm_concepts: bool = True,
    events_source: str | None = None,
    gcs_bucket: str | None = None,
    events_prefix: str = "event_corpus",
    events_out_dir: str | None = None,
) -> RegimeReport:
    """Run the full cascade for trading session ``as_of_date`` and return a
    circuit-broken ``RegimeReport``.

    ``as_of_date`` is the **trading session** (the day being traded pre-open), not
    the data date: news is capped at that session's open (pre-market cutoff) and
    the concept graph snapshot is read under that session.     ``use_llm_concepts``
    toggles the LLM cluster/sector judges (S2/S3); when False the numeric
    ``aggregate_concepts`` gate feeds L3 directly. ``tools``/``llm`` injectable.

    ``events_source`` ("gcs" | "local" | None) switches L1's input from raw
    vendor news to the standardized event corpus: "gcs" pulls
    ``events.jsonl``/``catalysts.jsonl`` for the session (needs ``gcs_bucket``)
    into ``events_out_dir``; "local" reads them from there directly. The
    news-active ticker set is then derived from the events themselves (ranked by
    primary-event count), so the market-wide S0 BQ scan is skipped.
    """
    # Per-layer models: deep Pro reserved for the high-leverage L3 market verdict;
    # the high-volume L1 (per-stock) and L2 (cluster/sector) use a faster flash
    # tier. Legacy ``model`` (if set) pins all three to one model.
    if model:
        l1_model = concept_model = regime_model = model

    tools = tools or get_market_tools(market)
    from tradingagents.concept_graph import store

    snapshot_dir = out_dir or store.DEFAULT_OUT_DIR
    cutoff_utc, cutoff_fmp = premarket_cutoffs(as_of_date)

    # S0 + S1: news-active stocks (news capped at pre-market) -> per-stock signals.
    # In events mode L1 reads the standardized corpus and the active set is the
    # tickers carrying events; otherwise S0 scans market-wide co-occurrence (or
    # the caller pins names explicitly for small test runs).
    events_by_ticker: dict[str, list] | None = None
    catalysts_by_ticker: dict[str, list] | None = None
    if events_source:
        from .store import DEFAULT_OUT_DIR as REGIME_OUT_DIR
        from .store import load_catalysts, load_events
        from .tickers import canonical_ticker

        ev_dir = events_out_dir or REGIME_OUT_DIR
        if events_source == "gcs":
            if not gcs_bucket:
                raise ValueError("events_source='gcs' requires gcs_bucket")
            from .gcs import download_catalysts, download_events

            download_events(as_of_date, gcs_bucket, events_prefix, out_dir=ev_dir)
            download_catalysts(as_of_date, gcs_bucket, events_prefix, out_dir=ev_dir)
        elif events_source != "local":
            raise ValueError(f"events_source must be 'gcs', 'local', or None (got {events_source!r})")

        events_by_ticker = {}
        for e in load_events(as_of_date, out_dir=ev_dir):
            events_by_ticker.setdefault(canonical_ticker(e.ticker), []).append(e)
        catalysts_by_ticker = {}
        for c in load_catalysts(as_of_date, out_dir=ev_dir):
            catalysts_by_ticker.setdefault(canonical_ticker(c.get("ticker", "")), []).append(c)

        def _rank(t: str):
            evs = events_by_ticker[t]
            return (-sum(1 for e in evs if e.is_primary), -len(evs), t)

        derived = sorted(events_by_ticker.keys(), key=_rank)
        if news_tickers is not None:
            pinned = {canonical_ticker(t) for t in news_tickers}
            tickers = [t for t in derived if t in pinned]
        else:
            tickers = derived[:max_news_tickers] if max_news_tickers else derived
    elif news_tickers is not None:
        tickers = news_tickers
    else:
        tickers = select_news_tickers(
            as_of_date, look_back_days=news_look_back_days, universe=universe,
            max_tickers=max_news_tickers, market=market, tools=tools, news_end=cutoff_utc,
        )
    stock_signals = analyze_stocks(
        tickers, as_of_date, market=market, tools=tools, llm=llm, provider=provider, model=l1_model,
        base_url=base_url, batch_size=batch_size, max_workers=max_workers, with_fundamentals=with_fundamentals,
        news_end=cutoff_utc, events_by_ticker=events_by_ticker, catalysts_by_ticker=catalysts_by_ticker,
    )

    cluster_map = store.load_memberships(as_of_date, snapshot_dir)
    clusters = store.load_clusters(as_of_date, snapshot_dir)

    if propagate:
        stock_signals = propagate_catalysts(as_of_date, stock_signals, out_dir=snapshot_dir)

    # S2 + S3: theme-cluster then sector verdicts (LLM cascade) or numeric gate.
    if use_llm_concepts:
        cluster_verdicts = judge_clusters(
            as_of_date, stock_signals, market=market, tools=tools, llm=llm, provider=provider, model=concept_model,
            base_url=base_url, out_dir=snapshot_dir, cluster_map=cluster_map, clusters=clusters,
            max_workers=max_workers, news_end=cutoff_utc,
        )
    else:
        cluster_verdicts = aggregate_concepts(
            as_of_date, stock_signals, out_dir=snapshot_dir, cluster_map=cluster_map, clusters=clusters,
        )
    sector_verdicts = judge_sectors(
        cluster_verdicts, as_of_date, llm=llm, provider=provider, model=concept_model,
        base_url=base_url, max_workers=max_workers,
    ) if use_llm_concepts else []

    # S4: market regime fed by the most-aggregated layer, + circuit breaker.
    top_concepts = sector_verdicts or cluster_verdicts
    report = analyze_regime(
        as_of_date, market=market, concept_signals=top_concepts, stock_signals=stock_signals,
        llm=llm, provider=provider, model=regime_model, base_url=base_url, tools=tools, news_end=cutoff_fmp,
    )
    return report.model_copy(update={"concept_signals": sector_verdicts + cluster_verdicts})
