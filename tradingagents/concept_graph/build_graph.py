"""Fuse co-movement + co-mention edges into a single pruned graph.

Each signal is normalised to [0,1], outer-merged on the (src,dst) pair,
combined with the configured weights, then pruned by a minimum weight and
an optional per-node top-K to keep the graph sparse enough for community
detection (M2).
"""

from __future__ import annotations

import pandas as pd

from .config import GraphConfig


def _normalize(s: pd.Series, method: str) -> pd.Series:
    if s.empty:
        return s
    if method == "minmax":
        lo, hi = s.min(), s.max()
        if hi == lo:
            return pd.Series(1.0, index=s.index)
        return (s - lo) / (hi - lo)
    if method == "rank":
        return s.rank(pct=True)
    raise ValueError(f"unknown normalize_method '{method}'")


def _canonicalize(edges: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """Ensure src < dst so the two signals merge on the same pair key."""
    if edges.empty:
        return edges[["src", "dst", value_col]]
    src = edges[["src", "dst"]].min(axis=1)
    dst = edges[["src", "dst"]].max(axis=1)
    out = pd.DataFrame({"src": src, "dst": dst, value_col: edges[value_col].values})
    return out


def fuse_and_prune(
    comovement_edges: pd.DataFrame,
    comention_edges: pd.DataFrame,
    config: GraphConfig,
    node_universe: set[str] | None = None,
) -> pd.DataFrame:
    """Return a fused, pruned edge table ``[src, dst, weight, comovement, comention]``.

    ``node_universe`` (e.g. the CS tickers from valid_ticker_v3_pure_cs)
    restricts the graph to genuine stock nodes, dropping any edge that
    touches an ETF/index proxy.
    """
    cm = _canonicalize(comovement_edges, "comovement")
    cn = _canonicalize(comention_edges, "comention")

    merged = pd.merge(cm, cn, on=["src", "dst"], how="outer")
    merged["comovement"] = merged["comovement"].fillna(0.0)
    merged["comention"] = merged["comention"].fillna(0.0)

    merged["comovement_n"] = _normalize(merged["comovement"], config.normalize_method)
    merged["comention_n"] = _normalize(merged["comention"], config.normalize_method)

    merged["weight"] = (
        config.fuse_alpha * merged["comention_n"]
        + config.fuse_beta * merged["comovement_n"]
    )

    if node_universe is not None:
        keep = merged["src"].isin(node_universe) & merged["dst"].isin(node_universe)
        merged = merged[keep]

    merged = merged[merged["weight"] >= config.prune_min_weight]

    if config.prune_top_k is not None:
        merged = _prune_top_k(merged, config.prune_top_k)

    result = merged[["src", "dst", "weight", "comovement", "comention"]]
    return result.sort_values("weight", ascending=False).reset_index(drop=True)


def _prune_top_k(edges: pd.DataFrame, k: int) -> pd.DataFrame:
    """Keep an edge if it is in the top-K by weight for either endpoint (KNN graph)."""
    if edges.empty:
        return edges
    stacked = pd.concat(
        [
            edges[["src", "weight"]].rename(columns={"src": "node"}),
            edges[["dst", "weight"]].rename(columns={"dst": "node"}),
        ]
    )
    stacked = stacked.assign(_idx=list(edges.index) * 2)
    keep_idx = (
        stacked.sort_values("weight", ascending=False)
        .groupby("node")
        .head(k)["_idx"]
        .unique()
    )
    return edges.loc[edges.index.isin(keep_idx)]


def to_networkx(edges: pd.DataFrame):
    """Build an undirected weighted networkx graph (networkx imported lazily)."""
    import networkx as nx

    g = nx.Graph()
    for row in edges.itertuples(index=False):
        g.add_edge(
            row.src,
            row.dst,
            weight=row.weight,
            comovement=row.comovement,
            comention=row.comention,
        )
    return g
