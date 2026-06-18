"""Conviction-stratified evaluation + exclusion-rule mining.

Two questions, one pass over persisted RegimeReports in a date range:

1. Do catalyst_confidence (stocks) and confidence/strength (concepts) carry
   information? -> precision/return/alpha per conviction bucket.

2. (the operational goal) What rule reliably filters OUT names that won't rise,
   at any granularity (stock / concept / regime)? -> for each candidate veto
   rule we report, on the EXCLUDED set, the realized up-rate, avg return, and how
   far the up-rate sits BELOW the long-everything baseline (a good "don't long"
   filter excludes names whose up-rate is well under the base rate — crucial in a
   bull month where almost everything rises).

Reuses the evaluator's return math (open-anchor, session = day 1). Prices for the
whole range are pulled once and sliced per session. Reports read locally (pulled
from GCS if missing). No look-ahead.

Example:
    BQ_USE_STORAGE_API=0 python scripts/analyze_conviction.py \
        --start 2026-02-01 --end 2026-05-31 --gcs-bucket trading_agent
"""

from __future__ import annotations

import os

os.environ.setdefault("GRPC_VERBOSITY", "ERROR")
os.environ.setdefault("GLOG_minloglevel", "2")

import argparse
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tradingagents.dataflows.secrets import load_secrets_to_env
from tradingagents.market_tools import get_market_tools
from tradingagents.regime.evaluate import _forward_return, _member_move, _wide
from tradingagents.regime.schemas import Direction, MarketRegime
from tradingagents.regime.store import DEFAULT_OUT_DIR, REPORT_FILE, load_report


def _mean(xs):
    return sum(xs) / len(xs) if xs else None


def _fmt(x, pct=False):
    if x is None:
        return "  n/a "
    return f"{x:+.2%}" if pct else f"{x:.3f}"


def _ensure_local(session: str, out_dir: str, bucket: str | None, prefix: str) -> bool:
    if (Path(out_dir) / session / REPORT_FILE).exists():
        return True
    if not bucket:
        return False
    from tradingagents.regime.gcs import download_report

    try:
        download_report(session, bucket, prefix, out_dir=out_dir)
        return True
    except FileNotFoundError:
        return False


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--market", default="US")
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--proxy", default="SPY")
    p.add_argument("--horizons", default="1,3,5")
    p.add_argument("--conf-edges", default="0,0.3,0.5,0.6,0.7,0.8,0.9,1.01",
                   help="catalyst_confidence bucket edges (stocks)")
    p.add_argument("--cc-edges", default="0,0.6,0.7,0.8,0.9,1.01",
                   help="concept confidence bucket edges")
    p.add_argument("--trend-metric", default="slope", choices=["slope", "endpoint"],
                   help="metric for concept member moves (stock returns always use endpoint P&L)")
    p.add_argument("--veto-conf", type=float, default=0.6, help="concept confidence cutoff for bearish-concept veto")
    p.add_argument("--bull-conf", type=float, default=0.8, help="concept confidence cutoff for bullish-concept backing")
    p.add_argument("--gcs-bucket", default=None, help="pull missing reports from this bucket")
    p.add_argument("--gcs-prefix", default="regime_gate")
    args = p.parse_args()

    horizons = tuple(int(h) for h in args.horizons.split(",") if h.strip())
    edges = [float(x) for x in args.conf_edges.split(",")]
    cc_edges = [float(x) for x in args.cc_edges.split(",")]
    max_h = max(horizons)
    load_secrets_to_env()

    tools = get_market_tools(args.market)
    cal_df = tools.load_daily_ohlc([args.proxy], args.start, args.end)
    if cal_df.empty:
        raise ValueError(f"no {args.proxy} bars in {args.start}..{args.end}")
    sessions = [d.strftime("%Y-%m-%d") for d in sorted(cal_df["trade_date"].unique())]

    reports = {}
    tickers: set[str] = {args.proxy}
    for s in sessions:
        if not _ensure_local(s, args.out_dir, args.gcs_bucket, args.gcs_prefix):
            continue
        r = load_report(s, out_dir=args.out_dir)
        reports[s] = r
        tickers.update(sig.ticker for sig in r.stock_signals)
        for c in r.concept_signals:
            tickers.update(c.member_tickers)
    if not reports:
        raise ValueError("no reports found in range")
    print(f"reports: {len(reports)}  ({min(reports)} .. {max(reports)})  tickers={len(tickers)}")

    g_start = (datetime.fromisoformat(min(reports)) - timedelta(days=14)).strftime("%Y-%m-%d")
    g_end = (datetime.fromisoformat(max(reports)) + timedelta(days=max_h * 3 + 10)).strftime("%Y-%m-%d")
    price_df = tools.load_daily_ohlc(sorted(tickers), g_start, g_end)
    opens, closes = _wide(price_df, "open"), _wide(price_df, "close")
    calendar = list(closes.index)
    cal_index = {ts: i for i, ts in enumerate(calendar)}

    # accumulators per horizon -------------------------------------------------
    L = {n: [] for n in horizons}            # longs: (conf, ret, mkt)
    S = {n: [] for n in horizons}            # shorts: (conf, ret, mkt)
    CC = {n: defaultdict(list) for n in horizons}   # concept hits by strength/level/conf-bucket
    # exclusion rules: rule -> list of (ret, mkt). "up" = ret>0
    RULE = {n: defaultdict(list) for n in horizons}
    # per-rule per-day coverage counts
    COVER = defaultdict(list)

    def cret(i0, n, t):
        return _forward_return(opens, closes, calendar, i0, n, t)

    for s, r in reports.items():
        ts = pd.Timestamp(s)
        if ts not in cal_index:
            continue
        i0 = cal_index[ts]

        # concept members by lean at/above the relevant confidence cutoff
        short_members, bull_members = set(), set()
        for c in r.concept_signals:
            if c.direction is Direction.SHORT and c.confidence >= args.veto_conf:
                short_members.update(c.member_tickers)
            if c.direction is Direction.LONG and c.confidence >= args.bull_conf:
                bull_members.update(c.member_tickers)

        longs = [sig for sig in r.stock_signals if sig.direction is Direction.LONG]
        # daily coverage (independent of horizon/price availability)
        COVER["stock_BLOCK"].append(sum(1 for sig in r.stock_signals if sig.direction is Direction.BLOCK))
        COVER["stock_SHORT"].append(sum(1 for sig in r.stock_signals if sig.direction is Direction.SHORT))
        COVER["long_conf<0.3"].append(sum(1 for sig in longs if sig.catalyst_confidence < 0.3))
        COVER["long_in_short_concept"].append(sum(1 for sig in longs if sig.ticker in short_members))
        COVER["long_regime_vetoed"].append(sum(1 for sig in longs if r.is_regime_vetoed_long(sig)))
        COVER["long_NOT_bull_backed"].append(sum(1 for sig in longs if sig.ticker not in bull_members))
        COVER["long_bull_backed"].append(sum(1 for sig in longs if sig.ticker in bull_members))
        COVER["long_bull_backed+conf>=0.8"].append(
            sum(1 for sig in longs if sig.ticker in bull_members and sig.catalyst_confidence >= 0.8))
        COVER["all_longs"].append(len(longs))

        for n in horizons:
            mkt = cret(i0, n, args.proxy)

            for sig in r.stock_signals:
                ret = cret(i0, n, sig.ticker)
                if ret is None:
                    continue
                pair = (ret, mkt)
                # conviction buckets
                if sig.direction is Direction.LONG:
                    L[n].append((sig.catalyst_confidence, ret, mkt))
                elif sig.direction is Direction.SHORT:
                    S[n].append((sig.catalyst_confidence, ret, mkt))
                # exclusion-rule populations
                RULE[n]["baseline_all_signals"].append(pair)
                if sig.direction is Direction.BLOCK:
                    RULE[n]["stock_BLOCK"].append(pair)
                if sig.direction is Direction.SHORT:
                    RULE[n]["stock_SHORT"].append(pair)
                if sig.direction is Direction.LONG:
                    RULE[n]["all_longs"].append(pair)
                    if sig.catalyst_confidence < 0.3:
                        RULE[n]["long_conf<0.3"].append(pair)
                    if sig.ticker in short_members:
                        RULE[n]["long_in_short_concept"].append(pair)
                    if r.is_regime_vetoed_long(sig):
                        RULE[n]["long_regime_vetoed"].append(pair)
                    if sig.ticker not in bull_members:
                        RULE[n]["long_NOT_bull_backed"].append(pair)
                    else:
                        RULE[n]["long_bull_backed"].append(pair)
                        if sig.catalyst_confidence >= 0.8:
                            RULE[n]["long_bull_backed+conf>=0.8"].append(pair)

            # concept hits by strength / level / confidence bucket
            for c in r.concept_signals:
                if c.direction is Direction.BLOCK or not c.member_tickers:
                    continue
                moves = [m for t in c.member_tickers
                         if (m := _member_move(opens, closes, calendar, i0, n, t, args.trend_metric)) is not None]
                if not moves:
                    continue
                avg = sum(moves) / len(moves)
                hit = 1.0 if ((c.direction is Direction.LONG and avg > 0) or
                              (c.direction is Direction.SHORT and avg < 0)) else 0.0
                CC[n][("level", c.level)].append(hit)
                CC[n][("strength", c.strength.value)].append(hit)
                for i in range(len(cc_edges) - 1):
                    if cc_edges[i] <= c.confidence < cc_edges[i + 1]:
                        CC[n][("conf", f"[{cc_edges[i]:.1f},{cc_edges[i+1]:.1f})")].append(hit)
                        break

    # ---- output: conviction buckets ----
    def bstats(rows, short):
        rets = [r for _, r, _ in rows]
        if not rets:
            return dict(n=0, prec=None, ret=None, alpha=None)
        if short:
            prec = _mean([1.0 if r < 0 else 0.0 for r in rets])
            alpha = _mean([m - r for _, r, m in rows if m is not None])
        else:
            prec = _mean([1.0 if r > 0 else 0.0 for r in rets])
            alpha = _mean([r - m for _, r, m in rows if m is not None])
        return dict(n=len(rets), prec=prec, ret=_mean(rets), alpha=alpha)

    for n in horizons:
        print(f"\n############## horizon {n}d ##############")
        for name, rows, short in (("LONG", L[n], False), ("SHORT", S[n], True)):
            print(f"\n  {name}  precision = % moved in bet direction")
            print(f"  {'conf bucket':<14}{'n':>7}{'prec':>8}{'avg_ret':>10}{'alpha':>10}")
            for i in range(len(edges) - 1):
                sub = [t for t in rows if edges[i] <= t[0] < edges[i + 1]]
                a = bstats(sub, short)
                print(f"  [{edges[i]:.2f},{edges[i+1]:.2f}){'':<3}{a['n']:>7}{_fmt(a['prec']):>8}"
                      f"{_fmt(a['ret'], 1):>10}{_fmt(a['alpha'], 1):>10}")
        print(f"\n  CONCEPT hit-rate ({args.trend_metric}):")
        for key in sorted(CC[n]):
            kind, val = key
            print(f"    {kind:<9}{val:<12} n={len(CC[n][key]):>4}  hit={_fmt(_mean(CC[n][key]))}")

    # ---- output: EXCLUSION rules (the operational goal) ----
    print("\n\n========================= EXCLUSION / VETO RULES =========================")
    print("Goal: high-precision 'do NOT long' filter. A good rule's EXCLUDED set has")
    print("up_rate WELL BELOW the long-everything baseline. 'lift' = baseline_up - rule_up")
    print("(higher lift = the rule removes names far less likely to rise).\n")
    print("(also shown: POSITIVE gates — keep these — where up_rate is ABOVE baseline)\n")
    rule_order = ["all_longs", "long_conf<0.3", "long_in_short_concept", "long_regime_vetoed",
                  "long_NOT_bull_backed", "stock_SHORT", "stock_BLOCK",
                  "long_bull_backed", "long_bull_backed+conf>=0.8"]
    for n in horizons:
        base_long = bstats([(0, r, m) for r, m in RULE[n]["all_longs"]], short=False)
        base_up = _mean([1.0 if r > 0 else 0.0 for r, _ in RULE[n]["all_longs"]])
        print(f"\n  --- horizon {n}d ---  (baseline all_longs up_rate={_fmt(base_up)}, "
              f"avg_ret={_fmt(base_long['ret'], 1)})")
        print(f"  {'rule':<24}{'n/day':>7}{'up_rate':>9}{'avg_ret':>10}{'below_mkt':>11}{'lift':>8}")
        for rule in rule_order:
            rows = RULE[n][rule]
            if not rows:
                continue
            up = _mean([1.0 if r > 0 else 0.0 for r, _ in rows])
            ret = _mean([r for r, _ in rows])
            below = _mean([1.0 if (m is not None and r < m) else 0.0 for r, m in rows])
            cov = _mean(COVER[rule]) if rule in COVER else None
            lift = (base_up - up) if up is not None else None
            print(f"  {rule:<24}{_fmt(cov) if cov is not None else '   n/a':>7}{_fmt(up):>9}"
                  f"{_fmt(ret, 1):>10}{_fmt(below):>11}{_fmt(lift):>8}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
