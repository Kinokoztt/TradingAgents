"""Run the regime gate for a date: cascade S0->S4 -> RegimeReport -> persist.

A schedulable CLI over tradingagents.regime.commander. API keys come from
Secret Manager (dataflows.secrets); BigQuery/GCS auth is ADC. Reads the concept
graph snapshot for the same date (build it first via rebuild_concept_graph.py).

Pre-market semantics: --as-of is the **trading session** (the day being traded),
and 'latest' resolves to today (ET). News is capped at that session's open
(09:30 ET pre-market cutoff); macro uses the session row (the shifted prior
close, visible pre-open); the concept graph snapshot is read under the session.
So nothing published/closed after the bell leaks in. Output is named by session.

Rollback safety: fundamentals are point-in-time (FMP statements filtered by SEC
acceptedDate <= session pre-open, ratios + price fields recomputed from BigQuery),
so historical replay is leak-free with fundamentals on. Use --no-fundamentals to
skip them entirely.

Examples:
    # simulate 2026-06-09 pre-market: news <= 2026-06-09 09:30 ET, output 2026-06-09/
    python scripts/run_regime_gate.py --as-of 2026-06-09 --gcs-bucket trading_agent

    # production pre-market: 'latest' = today ET (the session about to open)
    python scripts/run_regime_gate.py --as-of latest --gcs-bucket trading_agent
"""

from __future__ import annotations

import os

os.environ.setdefault("GRPC_VERBOSITY", "ERROR")
os.environ.setdefault("GLOG_minloglevel", "2")

import argparse
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tradingagents.concept_graph.store import DEFAULT_OUT_DIR as CG_OUT_DIR
from tradingagents.dataflows.secrets import load_secrets_to_env
from tradingagents.regime import run_regime_gate
from tradingagents.regime.store import DEFAULT_OUT_DIR, save_report


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--as-of", required=True, help="Trading session YYYY-MM-DD (named output), or 'latest' (= today ET)")
    p.add_argument("--market", default="US")
    p.add_argument("--tickers", default=None, help="Comma-separated universe subset (default: full candidate pool)")
    p.add_argument("--news-tickers", default=None,
                   help="Comma-separated tickers to analyze directly, SKIPPING the S0 market-wide news scan (small test runs)")
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="Where to write the regime report")
    p.add_argument("--cg-out-dir", default=CG_OUT_DIR, help="Concept graph snapshot dir to read")
    # Per-layer models: deep Pro only for the high-leverage L3 market verdict;
    # high-volume L1/L2 use a faster flash tier. --model pins all three.
    p.add_argument("--model", default=None, help="If set, use this model for ALL layers (overrides the per-layer flags)")
    p.add_argument("--l1-model", default="gemini-3-flash-preview", help="L1 per-stock analysis model")
    p.add_argument("--concept-model", default="gemini-3.1-pro-preview", help="L2 cluster/sector judge model")
    p.add_argument("--regime-model", default="gemini-3.1-pro-preview", help="L3 market-regime model (deep thinking)")

    # cascade knobs
    p.add_argument("--news-look-back", type=int, default=3)
    p.add_argument("--max-news-tickers", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=20)
    p.add_argument("--max-workers", type=int, default=6)
    p.add_argument("--no-fundamentals", action="store_true")
    p.add_argument("--no-propagate", action="store_true")
    p.add_argument("--no-llm-concepts", action="store_true", help="Use numeric gate instead of LLM cluster/sector judges")

    # durable storage (GCS)
    p.add_argument("--gcs-bucket", default=None, help="If set, upload the report to this bucket")
    p.add_argument("--gcs-prefix", default="regime_gate")

    args = p.parse_args()

    load_secrets_to_env()

    import datetime as _dt
    import zoneinfo

    today_et = _dt.datetime.now(zoneinfo.ZoneInfo("America/New_York")).date()

    as_of = args.as_of
    if as_of == "latest":
        as_of = today_et.strftime("%Y-%m-%d")
        print(f"resolved --as-of latest -> session {as_of} (today ET)")

    universe = None
    if args.tickers:
        universe = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]

    news_tickers = None
    if args.news_tickers:
        news_tickers = [t.strip().upper() for t in args.news_tickers.split(",") if t.strip()]

    report = run_regime_gate(
        as_of,
        market=args.market,
        universe=universe,
        news_tickers=news_tickers,
        out_dir=args.cg_out_dir,
        model=args.model,
        l1_model=args.l1_model,
        concept_model=args.concept_model,
        regime_model=args.regime_model,
        news_look_back_days=args.news_look_back,
        max_news_tickers=args.max_news_tickers,
        batch_size=args.batch_size,
        max_workers=args.max_workers,
        with_fundamentals=not args.no_fundamentals,
        propagate=not args.no_propagate,
        use_llm_concepts=not args.no_llm_concepts,
    )

    path = save_report(as_of, report, out_dir=args.out_dir)

    print(f"\nregime={report.market_state.value}  as_of={as_of}")
    print(f"stocks={len(report.stock_signals)}  raw_long={len(report.long_whitelist)}  "
          f"short={len(report.short_whitelist)}  l1_block={len(report.block_list)}")
    print(f"regime veto (rule, not overwrite): tradable_long={len(report.tradable_long_whitelist)}  "
          f"vetoed_long={len(report.regime_blocked_longs)}  (market_state={report.market_state.value})")
    sectors = [c for c in report.concept_signals if c.level == "sector"]
    themes = [c for c in report.concept_signals if c.level == "theme"]
    print(f"concepts: {len(sectors)} sector / {len(themes)} theme")
    for c in sectors:
        print(f"  [sector] {c.concept}: {c.direction.value}/{c.strength.value} conf={c.confidence:.2f}")
    print(f"\nmacro (LLM narrative): {report.macro_summary}")
    if report.key_drivers:
        print("key drivers:")
        for d in report.key_drivers:
            print(f"  - {d}")
    if report.macro_snapshot:
        print(f"\nmacro snapshot (raw macro_daily fed to L3):\n{report.macro_snapshot}")
    print(f"\nraw long whitelist (pre-veto): {report.long_whitelist}")
    print(f"tradable long whitelist (post-veto): {report.tradable_long_whitelist}")
    print(f"\nreport written to {path}")

    if args.gcs_bucket:
        from tradingagents.regime.gcs import upload_report

        uri = upload_report(as_of, args.gcs_bucket, args.gcs_prefix, out_dir=args.out_dir)
        print(f"uploaded to GCS: {uri}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
