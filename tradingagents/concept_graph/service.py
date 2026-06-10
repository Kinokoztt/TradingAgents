"""Concept-graph service: build the graph, detect concept clusters, persist,
and serve the query interface consumed by the regime gate (L1/L2).

Pipeline: data -> edges -> fused graph (M1) -> community detection (M2) ->
local-JSON snapshot -> get_cluster_map / get_cluster_label / get_neighbors.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd

from . import store
from .build_graph import fuse_and_prune, to_networkx
from .community import detect_communities
from .config import CommunityConfig, GraphConfig
from .naming import DEFAULT_NAMING_MODEL, name_clusters
from .schemas import Cluster, Membership
from .sources.comention import build_comention_edges
from .sources.comovement import build_comovement_edges


def _days_before(date_str: str, days: int) -> str:
    return (datetime.fromisoformat(date_str[:10]) - timedelta(days=days)).strftime("%Y-%m-%d")


def build_concept_graph(
    as_of_date: str,
    universe: list[str] | None = None,
    config: GraphConfig | None = None,
    market: str = "US",
    splits: pd.DataFrame | bool | None = None,
    max_news_articles: int = 20000,
):
    """Build the concept graph as of ``as_of_date`` (YYYY-MM-DD).

    Returns ``(edges_df, nx.Graph)``. ``edges_df`` columns:
    ``[src, dst, weight, comovement, comention]``.

    ``universe`` defaults to the market's candidate pool. ``splits`` defaults
    to the market's splits over the price window (set ``splits=False`` to skip
    correction); it feeds co-movement so unadjusted split days don't inject
    phantom returns (see comovement.py).
    """
    from tradingagents.market_tools import get_market_tools

    config = config or GraphConfig()
    tools = get_market_tools(market)

    if universe is None:
        universe = tools.load_candidate_universe()
    node_universe = set(universe)

    # Price window: need comovement_window+1 trading days; pad calendar days
    # generously to cover weekends/holidays.
    price_tickers = sorted(node_universe | {config.market_ticker})
    price_start = _days_before(as_of_date, int(config.comovement_window * 2) + 10)
    prices = tools.load_daily_close(price_tickers, price_start, as_of_date)

    # splits: None -> auto-load over the price window; False -> skip; else use given
    if splits is None:
        splits = tools.load_splits(price_tickers, price_start, as_of_date)
    elif splits is False:
        splits = None
    comovement_edges = build_comovement_edges(prices, config, splits=splits)

    # News window for co-mention.
    news_start = _days_before(as_of_date, config.comention_window_days)
    articles = tools.load_news_articles(news_start, as_of_date, max_articles=max_news_articles)
    comention_edges = build_comention_edges(articles, as_of_date, config)

    edges = fuse_and_prune(comovement_edges, comention_edges, config, node_universe=node_universe)
    graph = to_networkx(edges)
    return edges, graph


def detect_and_save(
    as_of_date: str,
    edges: pd.DataFrame,
    graph,
    community_config: CommunityConfig | None = None,
    out_dir: str = store.DEFAULT_OUT_DIR,
    label_date: str | None = None,
):
    """Run community detection on ``graph`` and persist the snapshot.

    Returns ``(memberships, clusters)`` and writes edges/memberships/clusters
    under ``{out_dir}/{label_date or as_of_date}/`` — ``label_date`` lets the
    snapshot be named by the trading session while ``as_of_date`` is the data date.
    """
    memberships, clusters = detect_communities(graph, community_config or CommunityConfig())
    store.save_snapshot(label_date or as_of_date, edges, memberships, clusters, out_dir)
    return memberships, clusters


def build_detect_save(
    as_of_date: str,
    universe: list[str] | None = None,
    config: GraphConfig | None = None,
    community_config: CommunityConfig | None = None,
    market: str = "US",
    out_dir: str = store.DEFAULT_OUT_DIR,
    label_date: str | None = None,
):
    """Full M1+M2 batch: build graph (data as of ``as_of_date``), detect clusters,
    persist under ``label_date or as_of_date``. Returns
    ``(edges, graph, memberships, clusters)``."""
    edges, graph = build_concept_graph(as_of_date, universe=universe, config=config, market=market)
    memberships, clusters = detect_and_save(as_of_date, edges, graph, community_config, out_dir, label_date)
    return edges, graph, memberships, clusters


def name_and_save_clusters(
    as_of_date: str,
    provider: str = "google",
    model: str = DEFAULT_NAMING_MODEL,
    out_dir: str = store.DEFAULT_OUT_DIR,
) -> dict[str, Cluster]:
    """Load the snapshot's clusters, name them via LLM, re-save. Returns named clusters."""
    clusters = store.load_clusters(as_of_date, out_dir)
    named = name_clusters(clusters, provider=provider, model=model)
    store.save_clusters(as_of_date, named, out_dir)
    return named


# --- Query interface (consumed by the regime gate L1/L2) ---


def get_cluster_map(as_of_date: str, out_dir: str = store.DEFAULT_OUT_DIR) -> dict[str, list[Membership]]:
    """ticker -> [Membership]: primary + weighted secondary memberships."""
    return store.load_memberships(as_of_date, out_dir)


def get_cluster_label(as_of_date: str, cluster_id: str, out_dir: str = store.DEFAULT_OUT_DIR) -> Cluster:
    """Cluster metadata (members, parent_sector, representatives, label)."""
    return store.load_clusters(as_of_date, out_dir)[cluster_id]


def get_neighbors(
    as_of_date: str,
    ticker: str,
    top_k: int = 10,
    out_dir: str = store.DEFAULT_OUT_DIR,
) -> list[tuple[str, float]]:
    """Strongest neighbours of ``ticker`` by edge weight, for L1 catalyst propagation."""
    edges = store.load_edges(as_of_date, out_dir)
    hit = edges[(edges["src"] == ticker) | (edges["dst"] == ticker)].copy()
    hit["other"] = hit.apply(lambda r: r["dst"] if r["src"] == ticker else r["src"], axis=1)
    hit = hit.sort_values("weight", ascending=False).head(top_k)
    return list(zip(hit["other"], hit["weight"]))
