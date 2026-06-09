"""US stock splits (Massive /v3/reference/splits).

Feeds split correction in the concept graph's co-movement builder: an
unadjusted close drops ~split ratio on the execution day, injecting a phantom
return we neutralise (see concept_graph/sources/comovement.py).
"""

from __future__ import annotations

import pandas as pd

from tradingagents.dataflows import massive


def load_splits(
    tickers: list[str] | None,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Splits with execution_date in [start_date, end_date].

    Returns ``[ticker, execution_date, split_from, split_to]`` with
    ``execution_date`` as datetime. Pulls market-wide and filters to
    ``tickers`` (None = keep all); split counts in a window are small.
    """
    raw = massive.fetch_splits(start_date, end_date)
    df = pd.DataFrame(raw, columns=["ticker", "execution_date", "split_from", "split_to"])
    if tickers is not None:
        df = df[df["ticker"].isin(set(tickers))]
    df["execution_date"] = pd.to_datetime(df["execution_date"])
    return df.reset_index(drop=True)
