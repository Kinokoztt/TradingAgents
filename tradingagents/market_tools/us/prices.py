"""US price loaders (BigQuery: day_aggs_di / minute_aggs_di).

Prices are unadjusted; apply split correction downstream (see
concept_graph/sources/comovement.py and concept-graph-design.md §4.2).
"""

from __future__ import annotations

import pandas as pd

from ._bigquery import DAY_TABLE, MINUTE_TABLE, fq, run_query


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
