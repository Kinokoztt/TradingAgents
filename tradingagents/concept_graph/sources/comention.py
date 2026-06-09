"""Co-mention edges: time-decayed news co-occurrence.

Two tickers mentioned in the same article are weakly related; doing this
across a window builds a co-mention graph. Raw counts are biased toward
mega-caps that appear everywhere, so we normalise with Jaccard (or PMI).
"Gainers list" style articles that name dozens of tickers are dropped
because they manufacture spurious co-occurrence.
"""

from __future__ import annotations

import math
from collections import defaultdict
from itertools import combinations

import pandas as pd

from ..config import GraphConfig


def build_comention_edges(
    articles: pd.DataFrame,
    as_of_date,
    config: GraphConfig,
) -> pd.DataFrame:
    """Build co-mention edges from an article table.

    ``articles`` columns: ``date`` (date/datetime) and ``tickers``
    (list[str] per row). ``as_of_date`` is the inclusive right edge of the
    lookback window (strict no-lookahead: articles after it are ignored).
    Returns a long edge table ``[src, dst, comention]`` with ``src < dst``.
    """
    for col in ("date", "tickers"):
        if col not in articles.columns:
            raise ValueError(f"articles is missing required column '{col}'")

    as_of = pd.to_datetime(as_of_date)
    cutoff = as_of - pd.Timedelta(days=config.comention_window_days)

    dates = pd.to_datetime(articles["date"])
    in_window = (dates > cutoff) & (dates <= as_of)

    weighted_count: dict[str, float] = defaultdict(float)   # n_i
    pair_weight: dict[tuple[str, str], float] = defaultdict(float)  # n_ij
    pair_articles: dict[tuple[str, str], int] = defaultdict(int)    # raw co-occur count

    for date, tickers in zip(dates[in_window], articles["tickers"][in_window]):
        unique = sorted(set(tickers))
        if len(unique) < 2 or len(unique) > config.comention_max_tickers_per_article:
            # singletons add no edges; oversized lists are noise
            for t in unique:
                weighted_count[t] += 0.0
            continue
        age_days = (as_of - date).days
        w = math.exp(-config.comention_decay_lambda * age_days)
        for t in unique:
            weighted_count[t] += w
        for i, j in combinations(unique, 2):
            pair_weight[(i, j)] += w
            pair_articles[(i, j)] += 1

    rows = []
    for (i, j), n_ij in pair_weight.items():
        if pair_articles[(i, j)] < config.comention_min_articles:
            continue
        if config.comention_method == "jaccard":
            denom = weighted_count[i] + weighted_count[j] - n_ij
            score = n_ij / denom if denom > 0 else 0.0
        elif config.comention_method == "pmi":
            total = sum(weighted_count.values())
            if total <= 0:
                score = 0.0
            else:
                p_ij = n_ij / total
                p_i = weighted_count[i] / total
                p_j = weighted_count[j] / total
                score = math.log(p_ij / (p_i * p_j)) if p_ij > 0 else 0.0
        else:
            raise ValueError(f"unknown comention_method '{config.comention_method}'")
        rows.append((i, j, score))

    edges = pd.DataFrame(rows, columns=["src", "dst", "comention"])
    if config.comention_method == "pmi":
        edges = edges[edges["comention"] > 0.0]
    return edges.reset_index(drop=True)
