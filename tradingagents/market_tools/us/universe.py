"""US candidate universe (BigQuery: valid_ticker_v3_pure_cs).

Default screen: common stock only. The volume/price floors are optional and
parameterized so the pool can be widened (e.g. avg_volume 5M -> 3M).
"""

from __future__ import annotations

from ._bigquery import UNIVERSE_TABLE, fq, run_query


def load_candidate_universe(
    min_avg_volume: float = 3_000_000,
    min_avg_price: float = 5.0,
    ticker_type: str | None = "CS",
) -> list[str]:
    """Return tickers passing the screen, sorted ascending.

    Pass ``ticker_type=None`` to drop the type filter, or set the floors to 0
    to disable the liquidity/price screen.
    """
    from google.cloud import bigquery

    conditions = ["avg_volume >= @min_avg_volume", "avg_price >= @min_avg_price"]
    params = [
        bigquery.ScalarQueryParameter("min_avg_volume", "FLOAT64", min_avg_volume),
        bigquery.ScalarQueryParameter("min_avg_price", "FLOAT64", min_avg_price),
    ]
    if ticker_type is not None:
        conditions.append("ticker_type = @ticker_type")
        params.append(bigquery.ScalarQueryParameter("ticker_type", "STRING", ticker_type))

    sql = f"""
        SELECT ticker
        FROM {fq(UNIVERSE_TABLE)}
        WHERE {' AND '.join(conditions)}
        ORDER BY ticker
    """
    df = run_query(sql, params)
    return df["ticker"].tolist()
