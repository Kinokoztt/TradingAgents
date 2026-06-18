"""Extract standardized news events for a session and persist events.jsonl.

Pipeline: select news-active tickers (or use --news-tickers) -> LLM event
extraction (classification, NOT direction) -> tag source reliability -> price-in
labeling against real prices -> write {out_dir}/{as_of}/events.jsonl.

This is the standardized corpus the NN pipeline will encode (see
docs/nn-pipeline-roadmap.md). Defaults to the self-hosted vLLM Qwen so repeated
re-runs are free; pass --provider/--backend-url to use a hosted API instead.

Examples:
    # local vLLM, explicit tickers (skip the market-wide news scan)
    python scripts/extract_events.py --as-of 2026-05-11 --news-tickers AAPL,NVDA

    # market-wide, hosted model
    python scripts/extract_events.py --as-of 2026-05-11 --provider google --model gemini-3.1-pro-preview
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("GRPC_VERBOSITY", "ERROR")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse

from tradingagents.llm_clients import config
from tradingagents.dataflows.secrets import load_secrets_to_env
from tradingagents.market_tools import get_market_tools
from tradingagents.regime import extract_events, tag_price_in, tag_source_reliability
from tradingagents.regime.commander import premarket_cutoffs
from tradingagents.regime.l1_stock import select_news_tickers
from tradingagents.regime.store import DEFAULT_OUT_DIR, save_events


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--as-of", required=True, help="Trading session YYYY-MM-DD (names the output dir)")
    p.add_argument("--market", default="US")
    p.add_argument("--tickers", default=None, help="Comma-separated universe subset for the news scan")
    p.add_argument("--news-tickers", default=None, help="Comma-separated tickers to extract directly (skip S0 scan)")
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR)

    p.add_argument("--provider", default="vllm", help="LLM provider (default vllm)")
    p.add_argument("--model", default="qwen3-32b", help="Model id / served-model-name")
    p.add_argument("--backend-url", default=None, help="Override provider base URL (e.g. http://localhost:8000/v1)")
    p.add_argument("--port", type=int, default=config.default_port(),
                   help="vLLM port for auto-serve (default from config / 8000)")
    p.add_argument("--auto-serve", action=argparse.BooleanOptionalAction, default=True,
                   help="For provider=vllm with no --backend-url: ensure the model's vLLM service is running "
                        "(start it, reuse if already up, or replace a different running model). Default on.")
    p.add_argument("--stop-after-task", action="store_true",
                   help="Shut the auto-started vLLM service down when the task finishes (default: leave it running)")

    p.add_argument("--news-look-back", type=int, default=7)
    p.add_argument("--max-news-tickers", type=int, default=None)
    p.add_argument("--max-articles-per-ticker", type=int, default=50)
    p.add_argument("--max-workers", type=int, default=4)
    p.add_argument("--no-price-in", action="store_true", help="Skip price-in labeling (no BigQuery price reads)")
    args = p.parse_args()

    load_secrets_to_env()
    as_of = args.as_of
    tools = get_market_tools(args.market)
    cutoff_utc, _ = premarket_cutoffs(as_of)

    base_url = args.backend_url
    auto_served = False
    if args.provider == "vllm" and args.auto_serve and args.backend_url is None:
        from tradingagents.llm_clients import vllm_service

        print(f"ensuring vLLM service for '{args.model}' on port {args.port} ...")
        state = vllm_service.ensure(args.model, port=args.port)
        base_url = state.base_url
        auto_served = True
        print(f"vLLM ready at {base_url} (pid {state.pid}, log {state.log_file})")

    try:
        if args.news_tickers:
            tickers = [t.strip().upper() for t in args.news_tickers.split(",") if t.strip()]
        else:
            universe = None
            if args.tickers:
                universe = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
            tickers = select_news_tickers(
                as_of, look_back_days=args.news_look_back, universe=universe,
                max_tickers=args.max_news_tickers, market=args.market, tools=tools, news_end=cutoff_utc,
            )
        print(f"extracting events for {len(tickers)} ticker(s) as of {as_of} via {args.provider}/{args.model}")

        events = extract_events(
            tickers, as_of, provider=args.provider, model=args.model, base_url=base_url,
            look_back_days=args.news_look_back, news_end=cutoff_utc,
            max_articles_per_ticker=args.max_articles_per_ticker, max_workers=args.max_workers,
        )
        print(f"extracted {len(events)} event(s)")

        tag_source_reliability(events)
        if not args.no_price_in:
            tag_price_in(events, market=args.market, tools=tools)

        path = save_events(as_of, events, out_dir=args.out_dir)

        from collections import Counter

        by_type = Counter(e.event_type.value for e in events)
        by_pricein = Counter(e.price_in.value for e in events)
        by_source = Counter(e.source_reliability.value for e in events)
        print(f"\nwritten to {path}")
        print(f"event types: {dict(by_type)}")
        print(f"price_in:    {dict(by_pricein)}")
        print(f"source tier: {dict(by_source)}")
    finally:
        if auto_served and args.stop_after_task:
            from tradingagents.llm_clients import vllm_service

            vllm_service.stop()
            print("stopped the auto-started vLLM service")
    return 0


if __name__ == "__main__":
    sys.exit(main())
