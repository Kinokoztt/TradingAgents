"""US price loaders (BigQuery: day_aggs_di / minute_aggs_di).

Prices are unadjusted; apply split correction downstream (see
concept_graph/sources/comovement.py and concept-graph-design.md §4.2).
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

import pandas as pd

from ._bigquery import DAY_TABLE, MINUTE_TABLE, fq, run_query


def latest_trading_day(as_of: str | None = None, lookback_days: int = 14) -> str:
    """Most recent trade_date in day_aggs_di (<= as_of), as YYYY-MM-DD.

    Used by the pre-market batch: the graph is built as of the last completed
    session, since the current day's bars don't exist yet pre-open.

    day_aggs_di requires a partition filter on trade_date, so we bound the
    scan to the last ``lookback_days`` (enough to span weekends/holidays).
    """
    from google.cloud import bigquery

    upper = as_of or date.today().strftime("%Y-%m-%d")
    lower = (datetime.fromisoformat(upper) - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    sql = f"""
        SELECT MAX(trade_date) AS d
        FROM {fq(DAY_TABLE)}
        WHERE trade_date BETWEEN @lower AND @upper
    """
    params = [
        bigquery.ScalarQueryParameter("lower", "DATE", lower),
        bigquery.ScalarQueryParameter("upper", "DATE", upper),
    ]
    d = run_query(sql, params)["d"].iloc[0]
    if pd.isna(d):
        raise ValueError(f"no trading day in day_aggs_di within {lower}..{upper}")
    return pd.to_datetime(d).strftime("%Y-%m-%d")


def previous_trading_day(session_date: str, lookback_days: int = 14) -> str:
    """The last completed session strictly before ``session_date`` (YYYY-MM-DD).

    This is the data basis visible at ``session_date`` pre-open: prices/clustering
    must use the prior close, never ``session_date``'s own bars (which may already
    be in BQ if run post-close).
    """
    prev = (datetime.fromisoformat(session_date[:10]) - timedelta(days=1)).strftime("%Y-%m-%d")
    return latest_trading_day(prev, lookback_days=lookback_days)


def load_daily_close(
    tickers: list[str],
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Long close table ``[ticker, trade_date, close]`` from day_aggs_di.

    ``tickers`` should include the market proxy (SPY/QQQ) used for
    de-marketing, since day_aggs_di carries ETFs/indices.
    """
    from google.cloud import bigquery

    # DISTINCT drops exact duplicate rows (day_aggs_di has occasional fully
    # re-ingested days, e.g. 2025-10-16). A genuine (ticker, trade_date) close
    # conflict survives and fails loudly downstream — we don't silently merge.
    sql = f"""
        SELECT DISTINCT ticker, trade_date, close
        FROM {fq(DAY_TABLE)}
        WHERE ticker IN UNNEST(@tickers)
          AND trade_date BETWEEN @start_date AND @end_date
        ORDER BY trade_date
    """
    params = [
        bigquery.ArrayQueryParameter("tickers", "STRING", tickers),
        bigquery.ScalarQueryParameter("start_date", "DATE", start_date),
        bigquery.ScalarQueryParameter("end_date", "DATE", end_date),
    ]
    df = run_query(sql, params)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    return df


def load_daily_ohlc(
    tickers: list[str],
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Long table ``[ticker, trade_date, open, high, low, close]`` from day_aggs_di.

    Used by the post-hoc evaluator (Module A), which anchors forward returns at
    the session **open** (the first fill after a pre-market judgment) and exits
    at later closes. high/low feed the ATR-based volatility band. ``tickers``
    should include the market proxy (SPY/QQQ).
    """
    from google.cloud import bigquery

    sql = f"""
        SELECT DISTINCT ticker, trade_date, open, high, low, close
        FROM {fq(DAY_TABLE)}
        WHERE ticker IN UNNEST(@tickers)
          AND trade_date BETWEEN @start_date AND @end_date
        ORDER BY trade_date
    """
    params = [
        bigquery.ArrayQueryParameter("tickers", "STRING", tickers),
        bigquery.ScalarQueryParameter("start_date", "DATE", start_date),
        bigquery.ScalarQueryParameter("end_date", "DATE", end_date),
    ]
    df = run_query(sql, params)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    return df


def load_daily_ohlcv(
    tickers: list[str],
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Long table ``[ticker, trade_date, open, high, low, close, volume]``.

    Same as ``load_daily_ohlc`` but carries ``volume``, which the price-in
    event study needs to compare pre/post-publication trading activity.
    """
    from google.cloud import bigquery

    sql = f"""
        SELECT DISTINCT ticker, trade_date, open, high, low, close, volume
        FROM {fq(DAY_TABLE)}
        WHERE ticker IN UNNEST(@tickers)
          AND trade_date BETWEEN @start_date AND @end_date
        ORDER BY trade_date
    """
    params = [
        bigquery.ArrayQueryParameter("tickers", "STRING", tickers),
        bigquery.ScalarQueryParameter("start_date", "DATE", start_date),
        bigquery.ScalarQueryParameter("end_date", "DATE", end_date),
    ]
    df = run_query(sql, params)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    return df


def load_minute_close(
    tickers: list[str],
    trade_date: str,
) -> pd.DataFrame:
    """Intraday minute bars for ``trade_date`` from minute_aggs_di.

    Returns ``[ticker, trade_minute, close, volume]``. ``trade_minute`` is a
    string like ``"2025-08-25 09:59"`` (US/Eastern), parsed to datetime.
    """
    from google.cloud import bigquery

    sql = f"""
        SELECT DISTINCT ticker, trade_minute, close, volume
        FROM {fq(MINUTE_TABLE)}
        WHERE ticker IN UNNEST(@tickers)
          AND trade_date = @trade_date
        ORDER BY trade_minute
    """
    params = [
        bigquery.ArrayQueryParameter("tickers", "STRING", tickers),
        bigquery.ScalarQueryParameter("trade_date", "DATE", trade_date),
    ]
    df = run_query(sql, params)
    df["trade_minute"] = pd.to_datetime(df["trade_minute"])
    return df
