"""Backfill Module A scorecards over a date range: score each session's persisted
RegimeReport against realized forward returns, then persist + upload — one
scorecard per session.

Reports live on GCS; each session is pulled to the local layout (unless it's
already there) before scoring. Trading days are derived from the proxy's (SPY)
bars in [--start, --end], so holidays/weekends are skipped. Sessions whose full
forward window hasn't elapsed yet are skipped by default (their scorecard would
be incomplete) — drop --skip-incomplete to write partial scorecards anyway.

Zero LLM cost, no look-ahead: only prices on/after each session's open are read,
and the ATR band uses strictly pre-session bars. See evaluate.py for the scoring
methodology (open-anchor returns, ATR-adaptive band, path-aware slope, regime
veto as a rule).

Examples:
    # dry-run: list sessions in range
    python scripts/backfill_evaluate.py --start 2026-02-01 --end 2026-05-31 --dry-run

    # backfill scorecards, pulling reports from GCS, skipping done sessions
    BQ_USE_STORAGE_API=0 python scripts/backfill_evaluate.py \
        --start 2026-02-01 --end 2026-05-31 \
        --gcs-bucket trading_agent --skip-existing --continue-on-error
"""

from __future__ import annotations

import os

os.environ.setdefault("GRPC_VERBOSITY", "ERROR")
os.environ.setdefault("GLOG_minloglevel", "2")

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tradingagents.dataflows.secrets import load_secrets_to_env
from tradingagents.market_tools import get_market_tools
from tradingagents.regime.evaluate import evaluate_report
from tradingagents.regime.store import DEFAULT_OUT_DIR, REPORT_FILE, SCORECARD_FILE, load_report, save_scorecard


def _trading_days(tools, proxy: str, start: str, end: str) -> list[str]:
    df = tools.load_daily_ohlc([proxy], start, end)
    if df.empty:
        raise ValueError(f"no {proxy} bars in {start}..{end}; check the date range / proxy / BQ data")
    return [d.strftime("%Y-%m-%d") for d in sorted(df["trade_date"].unique())]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--start", required=True, help="First session YYYY-MM-DD (inclusive)")
    p.add_argument("--end", required=True, help="Last session YYYY-MM-DD (inclusive)")
    p.add_argument("--market", default="US")
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="Where reports live / scorecards are written")
    p.add_argument("--proxy", default="SPY", help="Market proxy for truth + de-marketing and the trading calendar")
    p.add_argument("--horizons", default="1,3,5", help="Comma-separated trading-day horizons")
    p.add_argument("--band-mode", default="atr", choices=["atr", "fixed"])
    p.add_argument("--atr-window", type=int, default=14)
    p.add_argument("--atr-k", type=float, default=1.0)
    p.add_argument("--range-band", type=float, default=0.01)
    p.add_argument("--trend-metric", default="slope", choices=["slope", "endpoint"])

    # storage + backfill control
    p.add_argument("--gcs-bucket", default=None, help="Bucket to pull reports from and upload scorecards to")
    p.add_argument("--gcs-prefix", default="regime_gate")
    p.add_argument("--skip-existing", action="store_true", help="Skip sessions whose scorecard already exists locally")
    p.add_argument("--skip-incomplete", action="store_true",
                   help="Skip sessions whose longest horizon hasn't elapsed yet (avoid partial scorecards)")
    p.add_argument("--continue-on-error", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    horizons = tuple(int(h) for h in args.horizons.split(",") if h.strip())
    max_h = max(horizons)
    load_secrets_to_env()

    tools = get_market_tools(args.market)
    days = _trading_days(tools, args.proxy, args.start, args.end)
    latest = tools.latest_trading_day()
    print(f"{len(days)} trading session(s) in {args.start}..{args.end}; latest available={latest}")
    if args.dry_run:
        print(days)
        return 0

    done: list[str] = []
    skipped: list[str] = []
    failed: list[tuple[str, str]] = []

    for session in days:
        if args.skip_existing and (Path(args.out_dir) / session / SCORECARD_FILE).exists():
            print(f"[{session}] skip (scorecard exists)")
            skipped.append(session)
            continue

        print(f"\n========== {session} ==========")
        try:
            # ensure the report is local (reports live on GCS)
            if not (Path(args.out_dir) / session / REPORT_FILE).exists():
                if not args.gcs_bucket:
                    raise FileNotFoundError(
                        f"report for {session} not local and no --gcs-bucket given to pull it")
                from tradingagents.regime.gcs import download_report

                download_report(session, args.gcs_bucket, args.gcs_prefix, out_dir=args.out_dir)
                print(f"[{session}] report pulled from gs://{args.gcs_bucket}/{args.gcs_prefix}/{session}/")

            report = load_report(session, out_dir=args.out_dir)

            tickers = {s.ticker for s in report.stock_signals}
            for c in report.concept_signals:
                tickers.update(c.member_tickers)
            tickers.add(args.proxy)

            start = (datetime.fromisoformat(session) - timedelta(days=args.atr_window * 2 + 14)).strftime("%Y-%m-%d")
            end = (datetime.fromisoformat(session) + timedelta(days=max_h * 3 + 10)).strftime("%Y-%m-%d")
            price_df = tools.load_daily_ohlc(sorted(tickers), start, end)

            scorecard = evaluate_report(
                report, price_df, proxy=args.proxy, horizons=horizons,
                band_mode=args.band_mode, atr_window=args.atr_window, atr_k=args.atr_k,
                range_band=args.range_band, trend_metric=args.trend_metric,
            )

            if args.skip_incomplete and not scorecard.complete:
                print(f"[{session}] skip (incomplete: longest horizon not elapsed, latest={latest})")
                skipped.append(session)
                continue

            save_scorecard(session, scorecard, out_dir=args.out_dir)
            if args.gcs_bucket:
                from tradingagents.regime.gcs import upload_scorecard

                upload_scorecard(session, args.gcs_bucket, args.gcs_prefix, out_dir=args.out_dir)

            mkt_hits = [h.market_hit for h in scorecard.horizons if h.market_hit is not None]
            print(f"[{session}] state={scorecard.market_state} complete={scorecard.complete} "
                  f"market_hits={sum(1 for x in mkt_hits if x)}/{len(mkt_hits)}")
            done.append(session)
        except Exception as e:  # noqa: BLE001 — backfill loop may survive a bad session
            if not args.continue_on_error:
                raise
            print(f"[{session}] FAILED: {type(e).__name__}: {e}")
            failed.append((session, f"{type(e).__name__}: {e}"))

    print(f"\n=== evaluate backfill summary === done={len(done)} skipped={len(skipped)} failed={len(failed)}")
    if failed:
        for s, msg in failed:
            print(f"  FAILED {s}: {msg}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
