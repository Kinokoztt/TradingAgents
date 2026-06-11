"""US macro structured data (BigQuery: macro_daily).

The pipeline that loads macro_daily already applies shift(+1) so each row is
the previous close, visible pre-market for that trade_date — reading the
current row carries no look-ahead bias (see docs/macro_daily_deploy.py).
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd

from ._bigquery import MACRO_TABLE, fq, run_query

_FIELDS = [
    "trade_date",
    "us10y_yield",
    "us_yield_curve_spread",
    "vix_close",
    "nasdaq100_close",
    "usd_broad_index",
    "vix_pct_change",
    "nasdaq100_pct_change",
    "usd_broad_index_stale_days",
]


def load_macro_daily(start_date: str, end_date: str) -> pd.DataFrame:
    """Macro rows in [start_date, end_date], ordered by trade_date ascending."""
    from google.cloud import bigquery

    cols = ", ".join(_FIELDS)
    sql = f"""
        SELECT {cols}
        FROM {fq(MACRO_TABLE)}
        WHERE trade_date BETWEEN @start_date AND @end_date
        ORDER BY trade_date
    """
    params = [
        bigquery.ScalarQueryParameter("start_date", "DATETIME", start_date),
        bigquery.ScalarQueryParameter("end_date", "DATETIME", end_date),
    ]
    df = run_query(sql, params)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    return df


def get_macro_summary(curr_date: str, look_back_days: int = 10) -> str:
    """LLM-ready trend summary of rates/VIX/spread/DXY over the look-back window."""
    start = (datetime.fromisoformat(curr_date[:10]) - timedelta(days=look_back_days)).strftime("%Y-%m-%d")
    df = load_macro_daily(start, curr_date)
    if df.empty:
        return f"No macro_daily rows between {start} and {curr_date}."

    latest = df.iloc[-1]
    usd_stale = int(latest["usd_broad_index_stale_days"])
    usd_note = f" (stale {usd_stale}d — no fresh obs)" if usd_stale > 0 else ""
    lines = [
        f"## Macro snapshot as of {curr_date} (pre-market visible)",
        "",
        f"- US 10Y yield: {latest['us10y_yield']}",
        f"- 10Y-2Y spread: {latest['us_yield_curve_spread']}",
        f"- VIX close: {latest['vix_close']} ({latest['vix_pct_change']:+.2%} d/d)"
        if pd.notna(latest["vix_pct_change"])
        else f"- VIX close: {latest['vix_close']}",
        f"- Nasdaq-100 (spot): {latest['nasdaq100_close']} ({latest['nasdaq100_pct_change']:+.2%} d/d)"
        if pd.notna(latest["nasdaq100_pct_change"])
        else f"- Nasdaq-100 (spot): {latest['nasdaq100_close']}",
        f"- USD broad index (Fed DTWEXBGS, ~120; not ICE DXY): {latest['usd_broad_index']}{usd_note}",
        "",
        f"### {look_back_days}-day trend (oldest → newest)",
        "| Date | 10Y | Spread | VIX | NQ100 | USD idx | USD stale(d) |",
        "|---|---|---|---|---|---|---|",
    ]
    for _, row in df.iterrows():
        lines.append(
            f"| {row['trade_date'].date()} | {row['us10y_yield']} | "
            f"{row['us_yield_curve_spread']} | {row['vix_close']} | "
            f"{row['nasdaq100_close']} | {row['usd_broad_index']} | {int(row['usd_broad_index_stale_days'])} |"
        )
    return "\n".join(lines)
