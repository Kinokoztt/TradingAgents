"""Unit tests for the concept-graph M1 edge builders (synthetic data).

These exercise the pure algorithm layer only — no BigQuery, no networkx —
so they run fast and offline.
"""

import numpy as np
import pandas as pd
import pytest

from tradingagents.concept_graph import GraphConfig
from tradingagents.concept_graph.build_graph import fuse_and_prune
from tradingagents.concept_graph.sources.comention import build_comention_edges
from tradingagents.concept_graph.sources.comovement import (
    build_comovement_edges,
    compute_returns,
)

pytestmark = pytest.mark.unit


def _synthetic_prices(n: int = 150, seed: int = 0) -> pd.DataFrame:
    """Two co-moving groups (A: AAA/BBB/CCC, B: DDD/EEE) plus a market proxy.

    After de-marketing, within-group residuals should correlate strongly and
    cross-group residuals should not.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-01", periods=n)
    mkt = rng.normal(0, 0.01, n)
    f_a = rng.normal(0, 0.02, n)
    f_b = rng.normal(0, 0.02, n)

    def mk(beta, group):
        return beta * mkt + group + rng.normal(0, 0.003, n)

    returns = {
        "SPY": mkt,
        "AAA": mk(1.0, f_a),
        "BBB": mk(1.1, f_a),
        "CCC": mk(0.9, f_a),
        "DDD": mk(1.0, f_b),
        "EEE": mk(0.8, f_b),
    }
    close = pd.DataFrame(
        {t: 100 * np.cumprod(1 + r) for t, r in returns.items()}, index=dates
    )
    long = (
        close.reset_index()
        .melt(id_vars="index", var_name="ticker", value_name="close")
        .rename(columns={"index": "trade_date"})
    )
    return long


def _edge_value(edges, a, b, col):
    lo, hi = sorted([a, b])
    row = edges[(edges["src"] == lo) & (edges["dst"] == hi)]
    return None if row.empty else float(row[col].iloc[0])


def test_comovement_recovers_group_structure():
    prices = _synthetic_prices()
    config = GraphConfig(comovement_window=120, comovement_min_periods=40)
    edges = build_comovement_edges(prices, config)

    # SPY is the de-market basis and must not appear as a node.
    assert "SPY" not in set(edges["src"]) | set(edges["dst"])

    within = [
        _edge_value(edges, "AAA", "BBB", "comovement"),
        _edge_value(edges, "AAA", "CCC", "comovement"),
        _edge_value(edges, "BBB", "CCC", "comovement"),
        _edge_value(edges, "DDD", "EEE", "comovement"),
    ]
    cross = [
        _edge_value(edges, "AAA", "DDD", "comovement"),
        _edge_value(edges, "BBB", "EEE", "comovement"),
    ]
    within = [w for w in within if w is not None]
    assert len(within) >= 3
    cross_mean = np.mean([c for c in cross if c is not None]) if any(
        c is not None for c in cross
    ) else 0.0
    assert np.mean(within) > 0.5
    assert np.mean(within) > cross_mean + 0.3


def test_split_adjustment_neutralises_phantom_drop():
    dates = pd.bdate_range("2024-01-01", periods=5)
    # Flat price, then a 2-for-1 split halves the unadjusted close on day 3.
    close = pd.DataFrame({"XYZ": [100.0, 100.0, 50.0, 50.0, 50.0]}, index=dates)

    raw = compute_returns(close)
    assert raw["XYZ"].iloc[1] == pytest.approx(-0.5)  # phantom -50% without adj

    splits = pd.DataFrame(
        {
            "ticker": ["XYZ"],
            "execution_date": [dates[2]],
            "split_from": [1],
            "split_to": [2],
        }
    )
    adj = compute_returns(close, splits)
    assert adj["XYZ"].iloc[1] == pytest.approx(0.0, abs=1e-9)  # neutralised


def test_comention_jaccard_and_gainers_filter():
    base = pd.Timestamp("2024-06-01")
    rows = []
    # 10 articles co-mentioning AAA & BBB within the window
    for i in range(10):
        rows.append({"date": base + pd.Timedelta(days=i), "tickers": ["AAA", "BBB"]})
    # AAA also appears solo a few times (lowers Jaccard but keeps an edge)
    for i in range(3):
        rows.append({"date": base + pd.Timedelta(days=i), "tickers": ["AAA"]})
    # one oversized "gainers list" naming 20 tickers -> must be dropped
    rows.append(
        {"date": base + pd.Timedelta(days=5), "tickers": [f"Z{i}" for i in range(20)]}
    )
    articles = pd.DataFrame(rows)

    config = GraphConfig(comention_window_days=90, comention_max_tickers_per_article=15)
    edges = build_comention_edges(articles, base + pd.Timedelta(days=11), config)

    assert _edge_value(edges, "AAA", "BBB", "comention") > 0.0
    # Z-tickers only co-occur inside the oversized article -> no edge
    assert _edge_value(edges, "Z0", "Z1", "comention") is None


def test_fuse_and_prune_respects_universe_and_weights():
    prices = _synthetic_prices()
    config = GraphConfig(prune_min_weight=0.0, prune_top_k=None)
    cm = build_comovement_edges(prices, config)

    base = pd.Timestamp("2024-06-01")
    articles = pd.DataFrame(
        [{"date": base, "tickers": ["AAA", "BBB"]} for _ in range(5)]
    )
    cn = build_comention_edges(articles, base + pd.Timedelta(days=1), config)

    universe = {"AAA", "BBB", "CCC", "DDD", "EEE"}
    fused = fuse_and_prune(cm, cn, config, node_universe=universe)

    nodes = set(fused["src"]) | set(fused["dst"])
    assert nodes <= universe  # SPY excluded
    assert ((fused["weight"] >= 0.0) & (fused["weight"] <= 1.0)).all()
    assert {"weight", "comovement", "comention"} <= set(fused.columns)
