"""Backfill the regime gate over a date range: run S0->S4 per trading session,
persist locally and (optionally) upload to GCS — one report per session.

The concept-graph snapshot lives on GCS (never local): each session is pulled
from gs://{cg_bucket}/{cg_prefix}/{session}/ before the gate runs. Pass
--rebuild-graph to build it per session instead. Trading days are derived from
the proxy's (SPY) bars in [--start, --end], so holidays/weekends are skipped.

Rollback safety: each session uses only data visible pre-open that day (prices/
graph = prior close; news <= 09:00 ET; fundamentals = FMP point-in-time via
acceptedDate; economic-calendar actuals after the cutoff are blanked). The only
non-engineerable caveat is the LLM's own training cutoff (sessions the model may
have "seen" in training are not true out-of-sample).

Examples:
    # dry-run: list the sessions that would be processed
    python scripts/backfill_regime.py --start 2026-05-01 --end 2026-05-31 --dry-run

    # backfill May, upload to GCS, skip sessions already done, keep going on errors
    python scripts/backfill_regime.py --start 2026-05-01 --end 2026-05-31 \
        --gcs-bucket trading_agent --skip-existing --continue-on-error
"""

from __future__ import annotations

import os

os.environ.setdefault("GRPC_VERBOSITY", "ERROR")
os.environ.setdefault("GLOG_minloglevel", "2")

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tradingagents.concept_graph.store import DEFAULT_OUT_DIR as CG_OUT_DIR
from tradingagents.dataflows.secrets import load_secrets_to_env
from tradingagents.market_tools import get_market_tools
from tradingagents.regime import run_regime_gate
from tradingagents.regime.store import DEFAULT_OUT_DIR, REPORT_FILE, save_report


def _trading_days(tools, proxy: str, start: str, end: str) -> list[str]:
    """Sessions in [start, end] where the proxy printed a bar (the trading calendar)."""
    df = tools.load_daily_ohlc([proxy], start, end)
    if df.empty:
        raise ValueError(f"no {proxy} bars in {start}..{end}; check the date range / proxy / BQ data")
    return [d.strftime("%Y-%m-%d") for d in sorted(df["trade_date"].unique())]


def _rebuild_graph(session: str, args) -> None:
    """Shell out to the tested concept-graph CLI for one session (optional path)."""
    cmd = [
        sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "rebuild_concept_graph.py"),
        "--as-of", session, "--all", "--name", "--out-dir", args.cg_out_dir,
    ]
    if args.gcs_bucket:
        cmd += ["--gcs-bucket", args.gcs_bucket, "--gcs-prefix", "concept_graph"]
    subprocess.run(cmd, check=True)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--start", required=True, help="First session YYYY-MM-DD (inclusive)")
    p.add_argument("--end", required=True, help="Last session YYYY-MM-DD (inclusive)")
    p.add_argument("--market", default="US")
    p.add_argument("--proxy", default="SPY", help="Ticker whose bars define the trading calendar")
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--cg-out-dir", default=CG_OUT_DIR, help="Concept graph snapshot dir to read (and write if --rebuild-graph)")

    # per-layer models (match run_regime_gate.py defaults: L1 flash, L2/L3 Pro)
    p.add_argument("--model", default=None, help="If set, use this model for ALL layers")
    p.add_argument("--l1-model", default="gemini-3-flash-preview")
    p.add_argument("--concept-model", default="gemini-3.1-pro-preview")
    p.add_argument("--regime-model", default="gemini-3.1-pro-preview")

    # cascade knobs
    p.add_argument("--news-look-back", type=int, default=3)
    p.add_argument("--max-news-tickers", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=20)
    p.add_argument("--max-workers", type=int, default=6)
    p.add_argument("--no-fundamentals", action="store_true")
    p.add_argument("--no-propagate", action="store_true")
    p.add_argument("--no-llm-concepts", action="store_true")

    # storage + backfill control
    p.add_argument("--gcs-bucket", default=None, help="If set, upload each report to this bucket")
    p.add_argument("--gcs-prefix", default="regime_gate")
    p.add_argument("--cg-gcs-bucket", default=None, help="Bucket holding concept-graph snapshots (defaults to --gcs-bucket)")
    p.add_argument("--cg-gcs-prefix", default="concept_graph")
    p.add_argument("--rebuild-graph", action="store_true", help="Rebuild the concept graph per session instead of pulling from GCS")
    p.add_argument("--skip-existing", action="store_true", help="Skip sessions whose local report already exists")
    p.add_argument("--continue-on-error", action="store_true", help="Log and continue instead of aborting on a failed session")
    p.add_argument("--dry-run", action="store_true", help="Only list the sessions that would be processed")

    args = p.parse_args()
    cg_bucket = args.cg_gcs_bucket or args.gcs_bucket
    if not args.rebuild_graph and not cg_bucket:
        p.error("concept graph lives on GCS: pass --gcs-bucket (or --cg-gcs-bucket), or use --rebuild-graph")
    load_secrets_to_env()

    tools = get_market_tools(args.market)
    days = _trading_days(tools, args.proxy, args.start, args.end)
    print(f"{len(days)} trading session(s) in {args.start}..{args.end}: {days}")
    if args.dry_run:
        return 0

    done: list[str] = []
    skipped: list[str] = []
    failed: list[tuple[str, str]] = []

    for session in days:
        if args.skip_existing and (Path(args.out_dir) / session / REPORT_FILE).exists():
            print(f"[{session}] skip (report exists)")
            skipped.append(session)
            continue

        print(f"\n========== {session} ==========")
        try:
            if args.rebuild_graph:
                _rebuild_graph(session, args)
            else:
                from tradingagents.concept_graph.gcs import download_snapshot

                download_snapshot(session, cg_bucket, args.cg_gcs_prefix, out_dir=args.cg_out_dir)
                print(f"[{session}] concept graph pulled from gs://{cg_bucket}/{args.cg_gcs_prefix}/{session}/")

            report = run_regime_gate(
                session, market=args.market, out_dir=args.cg_out_dir,
                model=args.model, l1_model=args.l1_model,
                concept_model=args.concept_model, regime_model=args.regime_model,
                news_look_back_days=args.news_look_back, max_news_tickers=args.max_news_tickers,
                batch_size=args.batch_size, max_workers=args.max_workers,
                with_fundamentals=not args.no_fundamentals, propagate=not args.no_propagate,
                use_llm_concepts=not args.no_llm_concepts,
            )
            save_report(session, report, out_dir=args.out_dir)
            if args.gcs_bucket:
                from tradingagents.regime.gcs import upload_report

                upload_report(session, args.gcs_bucket, args.gcs_prefix, out_dir=args.out_dir)

            print(f"[{session}] regime={report.market_state.value}  stocks={len(report.stock_signals)}  "
                  f"raw_long={len(report.long_whitelist)}  tradable_long={len(report.tradable_long_whitelist)}")
            done.append(session)
        except Exception as e:  # noqa: BLE001 — backfill loop: optionally survive one bad session
            if not args.continue_on_error:
                raise
            print(f"[{session}] FAILED: {type(e).__name__}: {e}")
            failed.append((session, f"{type(e).__name__}: {e}"))

    print(f"\n=== backfill summary === done={len(done)} skipped={len(skipped)} failed={len(failed)}")
    if failed:
        for s, msg in failed:
            print(f"  FAILED {s}: {msg}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
