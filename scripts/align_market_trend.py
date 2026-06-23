"""Re-score the regime gate against the stock_rank_v1 ground-truth market trend.

Ground truth (mirrors ``stockprophet/.../estimate/market_trend.py``): the market's
true move for a session D over horizon k is the **equal-weight candidate-universe
forward return** — NOT SPY/QQQ — because the downstream amplifier trades that
universe. Here the candidate universe = every ``stock_signal`` in the session's
RegimeReport (Long+Short+Block); the forward return is open(D)-anchored and holds
k trading days (session counts as day 1, exit at close of i0+(k-1)).

state3 uses the ground-truth **dead band**: theta_k = dead_band * sigma_k where
sigma_k is the cross-session std of mkt_ret at horizon k (computed over ALL scored
sessions, ddof=1 — exactly the ground-truth definition, which is why this is a
cross-session aggregate, not a per-session scorecard). mkt_ret>theta -> Up
(==Bullish truth), <-theta -> Down (==Bearish), else Flat (==Range).

The gate's predicted state per horizon is its per-horizon ``outlook`` if emitted,
else the near-term ``market_state`` (matching evaluate.py's ``graded_state``). We
join truth vs gate by (session, horizon) and report the confusion matrix,
accuracy, base rate, and Bearish precision/recall.

Examples:
    python scripts/align_market_trend.py --horizons 1,3,5 --dead-band 0.25
    python scripts/align_market_trend.py --start 2026-02-01 --end 2026-03-31 \
        --out market_trend_align.csv
"""

from __future__ import annotations

import os

os.environ.setdefault("GRPC_VERBOSITY", "ERROR")
os.environ.setdefault("GLOG_minloglevel", "2")

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tradingagents.dataflows.secrets import load_secrets_to_env
from tradingagents.market_tools import get_market_tools
from tradingagents.regime.schemas import MarketRegime
from tradingagents.regime.store import DEFAULT_OUT_DIR, REPORT_FILE, load_report

STATES = [MarketRegime.BULLISH.value, MarketRegime.RANGE.value, MarketRegime.BEARISH.value]
TRUTH_OF_STATE3 = {"Up": "Bullish", "Flat": "Range", "Down": "Bearish"}


def _sessions(out_dir: str, start: str | None, end: str | None) -> list[str]:
    days = sorted(p.name for p in Path(out_dir).iterdir()
                  if p.is_dir() and (p / REPORT_FILE).exists())
    if start:
        days = [d for d in days if d >= start]
    if end:
        days = [d for d in days if d <= end]
    return days


def _universe_forward_returns(price_df: pd.DataFrame, proxy: str, session: str,
                              universe: list[str], horizons: tuple[int, ...]) -> dict[int, float | None]:
    """Equal-weight mean of open(session)->close(i0+k-1) returns over ``universe``.

    Proxy supplies the trading calendar (it prints every session). Horizons whose
    exit hasn't elapsed -> None (excluded downstream).
    """
    opens = price_df.pivot_table(index="trade_date", columns="ticker", values="open").sort_index()
    closes = price_df.pivot_table(index="trade_date", columns="ticker", values="close").sort_index()
    calendar = [d.strftime("%Y-%m-%d") for d in closes.index]
    opens.index = calendar
    closes.index = calendar
    if session not in calendar:
        raise ValueError(f"session {session} not in price calendar {calendar[0]}..{calendar[-1]}")
    i0 = calendar.index(session)

    held = [t for t in universe if t in opens.columns and t in closes.columns]
    out: dict[int, float | None] = {}
    for k in horizons:
        exit_i = i0 + (k - 1)
        if exit_i >= len(calendar):
            out[k] = None
            continue
        rets = []
        for t in held:
            base = opens.at[session, t]
            tgt = closes.at[calendar[exit_i], t]
            if pd.notna(base) and pd.notna(tgt) and base > 0:
                rets.append(tgt / base - 1.0)
        out[k] = float(np.mean(rets)) if rets else None
    return out


def _gate_state(report, k: int) -> str:
    outlook = report.outlook_for(k)
    return (outlook.direction if outlook else report.market_state).value


def _confusion(rows: pd.DataFrame) -> pd.DataFrame:
    m = pd.DataFrame(0, index=[f"pred {s}" for s in STATES],
                     columns=[f"truth {s}" for s in STATES])
    for _, r in rows.iterrows():
        m.at[f"pred {r['gate_state']}", f"truth {r['truth_state']}"] += 1
    return m


def _report_block(rows: pd.DataFrame, title: str) -> None:
    n = len(rows)
    acc = float((rows["gate_state"] == rows["truth_state"]).mean())
    up = int((rows["sign"] > 0).sum())
    dn = int((rows["sign"] < 0).sum())
    base = up / max(up + dn, 1)
    print(f"\n===== {title} =====  n={n}  accuracy={acc:.3f}  base_rate(P_up)={base:.3f}  "
          f"(up={up} flat={int((rows['sign'] == 0).sum())} down={dn})")
    print(_confusion(rows).to_string())

    # Per-state precision / recall (Bearish is the actionable veto signal).
    for s in STATES:
        pred = rows[rows["gate_state"] == s]
        truth = rows[rows["truth_state"] == s]
        tp = int((pred["truth_state"] == s).sum())
        prec = tp / len(pred) if len(pred) else None
        rec = tp / len(truth) if len(truth) else None
        f1 = (2 * prec * rec / (prec + rec)) if (prec and rec) else None
        print(f"  {s:<8} precision={_f(prec)}  recall={_f(rec)}  f1={_f(f1)}  "
              f"(pred_n={len(pred)} truth_n={len(truth)})")


def _f(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.3f}"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="Where the per-session reports live")
    p.add_argument("--start", default=None, help="First session YYYY-MM-DD (inclusive)")
    p.add_argument("--end", default=None, help="Last session YYYY-MM-DD (inclusive)")
    p.add_argument("--market", default="US")
    p.add_argument("--proxy", default="SPY", help="Only used for the trading calendar")
    p.add_argument("--horizons", default="1,3,5", help="Comma-separated trading-day horizons")
    p.add_argument("--dead-band", type=float, default=0.25,
                   help="Flat dead band = dead_band * cross-session sigma of mkt_ret (0 = pure sign)")
    p.add_argument("--out", default="", help="Optional CSV path for the per-(session,horizon) table")
    args = p.parse_args()

    horizons = tuple(int(h) for h in args.horizons.split(",") if h.strip())
    max_h = max(horizons)
    load_secrets_to_env()
    tools = get_market_tools(args.market)

    sessions = _sessions(args.out_dir, args.start, args.end)
    if not sessions:
        print(f"no sessions with {REPORT_FILE} under {args.out_dir}", file=sys.stderr)
        return 1
    print(f"{len(sessions)} session(s) in {args.out_dir} "
          f"[{sessions[0]}..{sessions[-1]}]  horizons={horizons}  dead_band={args.dead_band}")

    # Pass 1: per (session, horizon) equal-weight universe forward return + gate call.
    records: list[dict] = []
    for session in sessions:
        report = load_report(session, out_dir=args.out_dir)
        universe = sorted({s.ticker for s in report.stock_signals})
        if not universe:
            print(f"[{session}] skip (empty universe)")
            continue
        start = session
        end = (datetime.fromisoformat(session) + timedelta(days=max_h * 3 + 10)).strftime("%Y-%m-%d")
        price_df = tools.load_daily_ohlc(sorted(set(universe) | {args.proxy}), start, end)
        if price_df.empty:
            print(f"[{session}] skip (no price rows)")
            continue
        mkt = _universe_forward_returns(price_df, args.proxy, session, universe, horizons)
        for k in horizons:
            if mkt[k] is None:
                continue
            records.append({
                "session": session, "horizon": k, "n_universe": len(universe),
                "mkt_ret": mkt[k], "mkt_ret_bps": mkt[k] * 1e4,
                "gate_state": _gate_state(report, k),
            })
        elapsed = sum(1 for k in horizons if mkt[k] is not None)
        print(f"[{session}] universe={len(universe)}  horizons_elapsed={elapsed}/{len(horizons)}")

    df = pd.DataFrame.from_records(records)
    if df.empty:
        print("no elapsed (session, horizon) pairs to score", file=sys.stderr)
        return 1

    # Pass 2: ground-truth state3 with cross-session sigma dead band, per horizon.
    df["sign"] = np.sign(df["mkt_ret"]).astype(int)
    df["truth_state"] = None
    sigmas = {}
    for k in horizons:
        sel = df["horizon"] == k
        sub = df.loc[sel, "mkt_ret"]
        sigma = float(sub.std())  # ddof=1, matches ground-truth pandas .std()
        sigmas[k] = sigma
        theta = args.dead_band * sigma
        state3 = np.where(sub > theta, "Up", np.where(sub < -theta, "Down", "Flat"))
        df.loc[sel, "truth_state"] = pd.Series(state3, index=sub.index).map(TRUTH_OF_STATE3)

    print("\nper-horizon mkt_ret cross-session sigma (bps) / dead band theta (bps):")
    for k in horizons:
        print(f"  k={k}: sigma={sigmas[k] * 1e4:+.1f}bps  theta=±{args.dead_band * sigmas[k] * 1e4:.1f}bps  "
              f"mkt_ret_mean={df.loc[df['horizon'] == k, 'mkt_ret_bps'].mean():+.1f}bps")

    for k in horizons:
        _report_block(df[df["horizon"] == k], f"horizon {k}d")
    _report_block(df, "POOLED (all horizons)")

    if args.out:
        df.to_csv(args.out, index=False)
        print(f"\nper-(session,horizon) table written to {args.out} ({len(df)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
