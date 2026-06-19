"""Extract standardized news events for a session and persist events.jsonl.

Pipeline: select news-active tickers (within the same candidate universe the
concept graph uses) -> two-stage LLM extraction (stage 1 read, stage 2 classify;
NOT direction) -> tag source reliability -> price-in labeling against real prices
-> append {out_dir}/{as_of}/events.jsonl.

Resumable: each ticker is flushed and recorded as it finishes, so a re-run skips
completed tickers (use --no-resume to start over). Progress prints per ticker.

Defaults to the self-hosted vLLM Qwen and auto-starts/reuses its service; pass
--provider/--backend-url to use a hosted API instead.

Examples:
    python scripts/extract_events.py --as-of 2026-05-11
    python scripts/extract_events.py --as-of 2026-05-11 --news-tickers AAPL,NVDA
    python scripts/extract_events.py --as-of 2026-05-11 --model gemma-4-31b --stop-after-task
"""

from __future__ import annotations

import os
import sys
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

os.environ.setdefault("GRPC_VERBOSITY", "ERROR")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse

from tradingagents.llm_clients import config
from tradingagents.dataflows.secrets import load_secrets_to_env
from tradingagents.market_tools import get_market_tools
from tradingagents.regime import (
    build_event_llms,
    extract_ticker_events,
    fetch_ticker_articles,
    tag_price_in,
    tag_source_reliability,
)
from tradingagents.regime.commander import premarket_cutoffs
from tradingagents.regime.l1_stock import select_news_tickers
from tradingagents.regime.store import (
    DEFAULT_OUT_DIR,
    append_events,
    load_event_progress,
    load_events,
    mark_event_progress,
)


def _previous_session(as_of: str, tools, proxy: str) -> str:
    """Last trading session strictly before ``as_of`` (from the proxy's bars)."""
    from datetime import datetime, timedelta

    lo = (datetime.fromisoformat(as_of[:10]) - timedelta(days=12)).strftime("%Y-%m-%d")
    df = tools.load_daily_ohlc([proxy], lo, as_of)
    days = [d.strftime("%Y-%m-%d") for d in sorted(df["trade_date"].unique()) if d.strftime("%Y-%m-%d") < as_of]
    if not days:
        raise SystemExit(f"no {proxy} trading session found before {as_of} (needed for the incremental window)")
    return days[-1]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--as-of", required=True, help="Trading session YYYY-MM-DD (names the output dir)")
    p.add_argument("--market", default="US")
    p.add_argument("--tickers", default=None, help="Comma-separated universe subset for the news scan")
    p.add_argument("--news-tickers", default=None, help="Comma-separated tickers to extract directly (skip the scan)")
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR)

    p.add_argument("--provider", default="vllm", help="LLM provider (default vllm)")
    p.add_argument("--model", default="qwen3-32b", help="Model id / served-model-name")
    p.add_argument("--backend-url", default=None, help="Override provider base URL (e.g. http://localhost:8000/v1)")
    p.add_argument("--port", type=int, default=config.default_port(),
                   help="vLLM port for auto-serve (default from config / 8000)")
    p.add_argument("--auto-serve", action=argparse.BooleanOptionalAction, default=True,
                   help="For provider=vllm with no --backend-url: ensure the model's vLLM service is running. Default on.")
    p.add_argument("--stop-after-task", action="store_true",
                   help="Shut the auto-started vLLM service down when the task finishes")

    p.add_argument("--window", choices=["incremental", "lookback"], default="incremental",
                   help="incremental (default): news in (prev session cutoff, this session cutoff] — gapless, "
                        "no cross-day duplication. lookback: a fixed --news-look-back day window (overlaps daily).")
    p.add_argument("--proxy", default="SPY", help="Ticker whose bars define the trading calendar (for the prev session)")
    p.add_argument("--news-start", default=None,
                   help="Explicit window start (RFC3339 instant or YYYY-MM-DD); overrides --window")
    p.add_argument("--news-source", choices=["fmp", "massive"], default="fmp",
                   help="Per-ticker news vendor (default fmp: far more publishers than massive)")
    p.add_argument("--min-source-tier", choices=["high", "medium", "low", "all"], default="medium",
                   help="Drop publishers below this reliability tier before the LLM (default medium: "
                        "keeps wires + real journalism, drops opinion mills)")
    p.add_argument("--news-look-back", type=int, default=7, help="Window length in days when --window lookback")
    p.add_argument("--max-news-tickers", type=int, default=None)
    p.add_argument("--max-articles-per-ticker", type=int, default=50,
                   help="Cap articles per ticker (bounds prompt size / generation time)")
    p.add_argument("--max-workers", type=int, default=4)
    p.add_argument("--timeout", type=float, default=300.0,
                   help="Per-request LLM timeout in seconds (a stuck request fails instead of hanging)")
    p.add_argument("--max-tokens", type=int, default=8192,
                   help="Max generated tokens per LLM call (bounds the unbounded event list)")
    p.add_argument("--no-resume", action="store_true", help="Ignore prior progress and re-extract all tickers")
    p.add_argument("--no-price-in", action="store_true", help="Skip price-in labeling (no BigQuery price reads)")
    args = p.parse_args()

    load_secrets_to_env()
    as_of = args.as_of
    tools = get_market_tools(args.market)
    cutoff_utc, _ = premarket_cutoffs(as_of)

    # News window. Default "incremental": (prev session cutoff, this session
    # cutoff] — each article is processed exactly once across a daily backfill,
    # no cross-day duplication. "lookback" keeps a fixed N-day window (overlaps).
    if args.news_start:
        news_start = args.news_start
    elif args.window == "incremental":
        news_start = premarket_cutoffs(_previous_session(as_of, tools, args.proxy))[0]
    else:
        news_start = None  # select/fetch fall back to as_of - look_back_days
    window_desc = f"{news_start or f'{as_of}-{args.news_look_back}d'} .. {cutoff_utc}"
    print(f"news window [{args.window}]: {window_desc}")

    # Ticker universe: same candidate pool the concept graph uses
    # (tools.load_candidate_universe via select_news_tickers), restricted to
    # names with news in the window.
    if args.news_tickers:
        tickers = [t.strip().upper() for t in args.news_tickers.split(",") if t.strip()]
    else:
        universe = None
        if args.tickers:
            universe = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
        tickers = select_news_tickers(
            as_of, look_back_days=args.news_look_back, universe=universe,
            max_tickers=args.max_news_tickers, market=args.market, tools=tools,
            news_start=news_start, news_end=cutoff_utc,
        )

    done = set() if args.no_resume else load_event_progress(as_of, out_dir=args.out_dir)
    pending = [t for t in tickers if t not in done]
    total = len(tickers)
    print(f"as of {as_of}: {total} news-active ticker(s); {len(done)} already done, {len(pending)} to process "
          f"via {args.provider}/{args.model}")
    if not pending:
        print("nothing to do (all tickers already processed; use --no-resume to redo)")
        return 0

    base_url = args.backend_url
    auto_served = False
    if args.provider == "vllm" and args.auto_serve and args.backend_url is None:
        from tradingagents.llm_clients import vllm_service

        print(f"ensuring vLLM service for '{args.model}' on port {args.port} ...")
        state = vllm_service.ensure(args.model, port=args.port)
        base_url = state.base_url
        auto_served = True
        print(f"vLLM ready at {base_url} (pid {state.pid}, log {state.log_file})")

    stage1_llm, stage2_llm = build_event_llms(
        provider=args.provider, model=args.model, base_url=base_url,
        timeout=args.timeout, max_tokens=args.max_tokens,
    )

    from tradingagents.regime import SourceReliability

    min_tier = {
        "high": SourceReliability.HIGH, "medium": SourceReliability.MEDIUM,
        "low": SourceReliability.LOW, "all": None,
    }[args.min_source_tier]

    def work(ticker: str):
        articles = fetch_ticker_articles(
            ticker, as_of, look_back_days=args.news_look_back,
            news_start=news_start, news_end=cutoff_utc,
            max_articles_per_ticker=args.max_articles_per_ticker,
            source=args.news_source, min_source_tier=min_tier,
        )
        evs = extract_ticker_events(ticker, as_of, articles, stage1_llm, stage2_llm)
        tag_source_reliability(evs)
        if evs and not args.no_price_in:
            tag_price_in(evs, market=args.market, tools=tools)
        return ticker, evs

    lock = threading.Lock()
    processed = len(done)
    try:
        with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
            futures = {pool.submit(work, t): t for t in pending}
            for fut in as_completed(futures):
                ticker, evs = fut.result()
                with lock:
                    append_events(as_of, evs, out_dir=args.out_dir)
                    mark_event_progress(as_of, ticker, out_dir=args.out_dir)
                    processed += 1
                    print(f"[{processed}/{total}] {ticker:<8} -> {len(evs)} event(s)", flush=True)
    finally:
        if auto_served and args.stop_after_task:
            from tradingagents.llm_clients import vllm_service

            vllm_service.stop()
            print("stopped the auto-started vLLM service")

    events = load_events(as_of, out_dir=args.out_dir)
    by_type = Counter(e.event_type.value for e in events)
    by_polarity = Counter(e.polarity.value for e in events)
    by_certainty = Counter(e.certainty.value for e in events)
    by_pricein = Counter(e.price_in.value for e in events)
    primary = sum(1 for e in events if e.is_primary)
    print(f"\n{len(events)} total event(s) in {args.out_dir}/{as_of}/events.jsonl "
          f"({primary} primary, {len(events) - primary} secondary)")
    print(f"event_type: {dict(by_type)}")
    print(f"polarity:   {dict(by_polarity)}")
    print(f"certainty:  {dict(by_certainty)}")
    print(f"price_in:   {dict(by_pricein)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
