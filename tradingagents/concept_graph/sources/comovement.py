"""Co-movement edges: de-marketed residual return correlation.

Why de-market: raw return correlation is dominated by the common market
factor, so in a bull tape *every* stock looks correlated and community
detection degenerates. We regress each stock's return on a market proxy
(SPY/QQQ, which live in day_aggs_di) and correlate the residuals, which
captures genuine sector/theme co-movement beyond the index.

Why split-adjust: day_aggs_di is unadjusted, so a split injects a ~-90%
phantom return on the execution day that wrecks that stock's correlations.
We restore the affected day's return using the split ratio.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import GraphConfig


def compute_returns(
    close: pd.DataFrame,
    splits: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Daily simple returns from an unadjusted close matrix (date x ticker).

    ``splits`` (optional) has columns ``ticker, execution_date, split_from,
    split_to``. On a 2-for-1 split Polygon reports split_from=1, split_to=2,
    and the unadjusted close halves; we multiply that day's gross return by
    ``split_to / split_from`` to neutralise the phantom move.
    """
    if not isinstance(close.index, pd.DatetimeIndex):
        close = close.copy()
        close.index = pd.to_datetime(close.index)
    close = close.sort_index()

    returns = close.pct_change()

    if splits is not None and not splits.empty:
        for col in ("ticker", "execution_date", "split_from", "split_to"):
            if col not in splits.columns:
                raise ValueError(f"splits is missing required column '{col}'")
        for row in splits.itertuples(index=False):
            exec_date = pd.to_datetime(row.execution_date)
            if row.ticker not in returns.columns or exec_date not in returns.index:
                continue
            factor = float(row.split_to) / float(row.split_from)
            gross = 1.0 + returns.at[exec_date, row.ticker]
            returns.at[exec_date, row.ticker] = gross * factor - 1.0

    return returns.iloc[1:]


def _demarket(returns: pd.DataFrame, market: pd.Series, min_periods: int) -> pd.DataFrame:
    """Return residuals of each column regressed on the market series.

    Intercept (alpha) is dropped on purpose: correlation is invariant to a
    constant shift, so only the slope (beta * market) needs removing.
    """
    resid = pd.DataFrame(index=returns.index, columns=returns.columns, dtype=float)
    for ticker in returns.columns:
        r = returns[ticker]
        mask = r.notna() & market.notna()
        if int(mask.sum()) < min_periods:
            continue
        rc = r[mask]
        mc = market[mask]
        mc_centered = mc - mc.mean()
        var = float((mc_centered ** 2).sum())
        if var == 0.0:
            raise ValueError("market series has zero variance; cannot de-market")
        beta = float(((rc - rc.mean()) * mc_centered).sum() / var)
        resid[ticker] = r - beta * market
    return resid


def _winsorize(df: pd.DataFrame, q: float) -> pd.DataFrame:
    lower = df.quantile(q)
    upper = df.quantile(1.0 - q)
    return df.clip(lower=lower, upper=upper, axis=1)


def build_comovement_edges(
    prices: pd.DataFrame,
    config: GraphConfig,
    splits: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build co-movement edges from a long price table.

    ``prices`` columns: ``ticker, trade_date, close``. Returns a long edge
    table ``[src, dst, comovement]`` with ``src < dst`` (undirected).
    """
    for col in ("ticker", "trade_date", "close"):
        if col not in prices.columns:
            raise ValueError(f"prices is missing required column '{col}'")

    close = prices.pivot(index="trade_date", columns="ticker", values="close")
    close = close.tail(config.comovement_window + 1)

    returns = compute_returns(close, splits)

    # Fail fast on insufficient price history: with fewer return rows than
    # comovement_min_periods, every pairwise correlation falls below min_periods,
    # co-movement collapses to an empty edge table, and the graph silently
    # degenerates into a co-mention-only graph (fuse_beta weight becomes dead).
    # This typically means day_aggs_di lacks deep enough history for as_of
    # (e.g. early-2024 dates against a table that starts 2023-12-26) — surface it
    # instead of emitting a half-built graph.
    if len(returns) < config.comovement_min_periods:
        raise ValueError(
            f"insufficient price history for co-movement: only {len(returns)} return "
            f"rows in the window but comovement_min_periods={config.comovement_min_periods}. "
            f"Backfill day_aggs_di earlier or move the rollback start date forward."
        )

    if config.market_ticker not in returns.columns:
        raise ValueError(
            f"market_ticker '{config.market_ticker}' not found in price data"
        )

    market = returns[config.market_ticker]
    resid = _demarket(returns, market, config.comovement_min_periods)
    resid = resid.drop(columns=[config.market_ticker])

    if config.winsorize_quantile is not None:
        resid = _winsorize(resid, config.winsorize_quantile)

    corr = resid.corr(
        method=config.comovement_method,
        min_periods=config.comovement_min_periods,
    )

    corr = corr.rename_axis(index="src", columns="dst")
    edges = (
        corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
        .stack()
        .rename("comovement")
        .reset_index()
    )
    edges = edges.dropna(subset=["comovement"])

    if config.keep_positive_only:
        edges = edges[edges["comovement"] > 0.0]

    return edges.reset_index(drop=True)
