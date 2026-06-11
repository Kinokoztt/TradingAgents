"""Evaluate a past RegimeReport against realized forward returns (Module A).

Zero LLM cost: reads a persisted report + BQ day closes, scores the pre-market
judgment over 1/3/5 trading-day horizons, and writes scorecard.json next to the
report. No look-ahead — only prices on/after the session close are read.

Run it once the forward window has elapsed (5 trading days for the 5d horizon);
horizons that haven't elapsed yet score as null and ``complete`` stays false, so
you can rerun later. Backfill history by looping sessions in your shell.

Examples:
    # evaluate the 2026-06-09 judgment (needs prices through ~2026-06-16)
    python scripts/evaluate_regime.py --session 2026-06-09 --gcs-bucket trading_agent

    # custom proxy / horizons
    python scripts/evaluate_regime.py --session 2026-06-09 --proxy QQQ --horizons 1,3,5,10
"""

from __future__ import annotations

import os

os.environ.setdefault("GRPC_VERBOSITY", "ERROR")
os.environ.setdefault("GLOG_minloglevel", "2")

import argparse
import sys
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tradingagents.market_tools import get_market_tools
from tradingagents.regime.evaluate import evaluate_report
from tradingagents.regime.store import DEFAULT_OUT_DIR, load_report, save_scorecard


def _fmt(x: float | None, pct: bool = False) -> str:
    if x is None:
        return "n/a"
    return f"{x:+.2%}" if pct else f"{x:.3f}"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--session", required=True, help="Session date YYYY-MM-DD of the report to evaluate")
    p.add_argument("--market", default="US")
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="Where reports live / scorecard is written")
    p.add_argument("--proxy", default="SPY", help="Market proxy for truth + de-marketing (SPY/QQQ)")
    p.add_argument("--horizons", default="1,3,5", help="Comma-separated trading-day horizons")
    p.add_argument("--band-mode", default="atr", choices=["atr", "fixed"],
                   help="Market flat-zone band: 'atr' = atr_k*proxyATR%%*sqrt(N) (volatility-adaptive), 'fixed' = --range-band")
    p.add_argument("--atr-window", type=int, default=14, help="Wilder ATR lookback (pre-session bars)")
    p.add_argument("--atr-k", type=float, default=1.0, help="Band = atr_k * dayATR%% * sqrt(horizon)")
    p.add_argument("--range-band", type=float, default=0.01, help="Fixed fallback band (|proxy trend| <= band => Range)")
    p.add_argument("--trend-metric", default="slope", choices=["slope", "endpoint"],
                   help="Grade market_state on path-aware fitted drift ('slope') or single endpoint return ('endpoint')")
    p.add_argument("--gcs-bucket", default=None, help="If set, upload the scorecard to this bucket")
    p.add_argument("--gcs-prefix", default="regime_gate")
    args = p.parse_args()

    horizons = tuple(int(h) for h in args.horizons.split(",") if h.strip())
    report = load_report(args.session, out_dir=args.out_dir)
    tools = get_market_tools(args.market)

    # Early data-readiness check: forward returns need the session close + horizons
    # to be in BQ. The daily table can lag the session, so fail clearly here
    # instead of crashing deep inside on an empty calendar.
    latest = tools.latest_trading_day()
    if latest < args.session:
        print(f"ERROR: session {args.session} not yet in the daily table (latest available={latest}). "
              f"Nothing to evaluate until the session close is ingested.", file=sys.stderr)
        return 1
    if latest == args.session:
        print(f"NOTE: latest available trading day == session ({latest}); only h-horizons that have "
              f"elapsed will be scored, the rest stay null. Rerun later for the full window.")

    # tickers we need prices for: every signalled stock, every concept member, + proxy
    tickers = {s.ticker for s in report.stock_signals}
    for c in report.concept_signals:
        tickers.update(c.member_tickers)
    tickers.add(args.proxy)

    # Fetch from BEFORE the session (for the pre-session ATR) through enough
    # calendar days to span the longest horizon (trading days < calendar days;
    # pad generously for weekends/holidays).
    max_h = max(horizons)
    start = (datetime.fromisoformat(args.session) - timedelta(days=args.atr_window * 2 + 14)).strftime("%Y-%m-%d")
    end = (datetime.fromisoformat(args.session) + timedelta(days=max_h * 3 + 10)).strftime("%Y-%m-%d")
    price_df = tools.load_daily_ohlc(sorted(tickers), start, end)

    scorecard = evaluate_report(
        report, price_df, proxy=args.proxy, horizons=horizons,
        band_mode=args.band_mode, atr_window=args.atr_window, atr_k=args.atr_k,
        range_band=args.range_band, trend_metric=args.trend_metric,
    )

    path = save_scorecard(args.session, scorecard, out_dir=args.out_dir)

    band_desc = (f"atr (atr%={_fmt(scorecard.atr_pct, pct=True)}, k={scorecard.atr_k})"
                 if scorecard.band_mode == "atr" else f"fixed {scorecard.range_band:.2%}")
    print(f"session={scorecard.session}  market_state={scorecard.market_state}  "
          f"proxy={scorecard.proxy}  band={band_desc}  trend_metric={scorecard.trend_metric}  "
          f"complete={scorecard.complete}")
    for h in scorecard.horizons:
        tag = h.target_date or "not-elapsed"
        print(f"\n[h{h.horizon}d -> {tag}]  endpoint_ret={_fmt(h.market_return, pct=True)}  "
              f"trend={_fmt(h.market_trend, pct=True)} (R²={_fmt(h.market_trend_r2)})  "
              f"band=±{_fmt(h.range_band_used, pct=True)}  market_hit={h.market_hit}")
        print(f"  long : n={h.long.count} hit={_fmt(h.long.hit_rate)} "
              f"avg_alpha={_fmt(h.long.avg_alpha, pct=True)} win={_fmt(h.long.win_rate)} "
              f"precision={_fmt(h.long_precision)}")
        print(f"  short: n={h.short.count} hit={_fmt(h.short.hit_rate)} "
              f"avg_alpha={_fmt(h.short.avg_alpha, pct=True)} win={_fmt(h.short.win_rate)} "
              f"precision={_fmt(h.short_precision)}")
        print(f"  confidence_brier={_fmt(h.confidence_brier)}  "
              f"sector_hit={_fmt(h.sector_hit_rate)}  theme_hit={_fmt(h.theme_hit_rate)}")
        rv = h.regime_veto
        print(f"  regime veto: vetoed_longs={rv['vetoed_long_count']} eval={rv['evaluated']} "
              f"avg_ret={_fmt(rv['vetoed_avg_return'], pct=True)} rose={_fmt(rv['vetoed_rose_rate'])}")

    print(f"\nscorecard written to {path}")

    if args.gcs_bucket:
        from tradingagents.regime.gcs import upload_scorecard

        uri = upload_scorecard(args.session, args.gcs_bucket, args.gcs_prefix, out_dir=args.out_dir)
        print(f"uploaded to GCS: {uri}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
