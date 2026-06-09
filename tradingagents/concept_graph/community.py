"""Community detection (M2): hierarchical Leiden + anti-bloat + multi-membership.

Standard Leiden is a hard partition (one cluster per node), which doesn't fit
"a ticker can belong to several themes". So we build a hard-partition skeleton
at two resolutions (L_sector coarse, L_theme fine) and then layer controlled
multi-membership on top via edge-weight affinity. See
docs/concept-graph-design.md §5.

leidenalg + python-igraph are imported lazily; missing them raises a clear
install hint rather than silently falling back to a different algorithm.
"""

from __future__ import annotations

from collections import Counter, defaultdict

from .config import UNCLUSTERED, CommunityConfig
from .schemas import Cluster, Membership


def _require_leiden():
    try:
        import igraph
        import leidenalg
    except ImportError as exc:  # fail fast with an actionable message
        raise ImportError(
            "Community detection needs leidenalg + python-igraph. Install:\n"
            "  pip install leidenalg python-igraph"
        ) from exc
    return leidenalg, igraph


def _to_igraph(graph, igraph):
    """Convert an undirected weighted networkx graph to igraph (+ node list)."""
    nodes = list(graph.nodes())
    index = {n: i for i, n in enumerate(nodes)}
    edges = [(index[u], index[v]) for u, v in graph.edges()]
    weights = [graph[u][v].get("weight", 1.0) for u, v in graph.edges()]
    ig = igraph.Graph(n=len(nodes), edges=edges)
    ig.es["weight"] = weights
    return ig, nodes


def _leiden(ig, resolution, seed, leidenalg):
    """Return a membership list[int] (one cluster id per vertex)."""
    part = leidenalg.find_partition(
        ig,
        leidenalg.RBConfigurationVertexPartition,
        weights="weight",
        resolution_parameter=resolution,
        seed=seed,
    )
    return list(part.membership)


def _num_clusters(membership) -> int:
    return len(set(membership))


def _search_theme_resolution(ig, config: CommunityConfig, leidenalg) -> float:
    """Binary-search resolution so the cluster count lands in the target range.

    The *total* cluster count rises monotonically with resolution (unlike the
    count of clusters above ``min_cluster_size``, which is non-monotonic —
    high resolution shatters groups into sub-min fragments). So the search
    targets total count; ``min_cluster_size`` cleanup happens afterwards.

    If even the highest resolution can't reach target_min (small graphs),
    return hi; if the lowest already exceeds target_max, return lo.
    """
    lo, hi = config.theme_resolution_lo, config.theme_resolution_hi
    if _num_clusters(_leiden(ig, hi, config.seed, leidenalg)) < config.theme_target_min:
        return hi
    if _num_clusters(_leiden(ig, lo, config.seed, leidenalg)) > config.theme_target_max:
        return lo
    best = lo
    for _ in range(config.theme_search_iters):
        mid = (lo + hi) / 2.0
        count = _num_clusters(_leiden(ig, mid, config.seed, leidenalg))
        if count < config.theme_target_min:
            lo = mid
        elif count > config.theme_target_max:
            hi = mid
        else:
            return mid
        best = mid
    return best


def _apply_min_cluster_size(graph, labels: dict[str, str], config: CommunityConfig) -> dict[str, str]:
    """Reassign members of undersized clusters to their strongest valid cluster.

    Uses the original labels for neighbour lookup so reassignment order doesn't
    matter; nodes with no edge into any valid cluster become UNCLUSTERED.
    """
    sizes = Counter(labels.values())
    valid = {c for c, s in sizes.items() if s >= config.min_cluster_size}
    orig = dict(labels)
    out = dict(labels)
    for node in graph.nodes():
        if orig[node] in valid:
            continue
        scores: dict[str, float] = defaultdict(float)
        for nb in graph.neighbors(node):
            c = orig[nb]
            if c in valid:
                scores[c] += graph[node][nb].get("weight", 1.0)
        out[node] = max(scores, key=scores.get) if scores else UNCLUSTERED
    return out


def _weighted_degree(graph, node) -> float:
    return sum(graph[node][nb].get("weight", 1.0) for nb in graph.neighbors(node))


def _multi_membership(graph, theme_labels: dict[str, str], config: CommunityConfig):
    """Primary (hard) membership + edge-weight-affinity secondary memberships."""
    members_by_cluster: dict[str, list[str]] = defaultdict(list)
    for node, c in theme_labels.items():
        members_by_cluster[c].append(node)

    memberships: dict[str, list[Membership]] = {}
    for node in graph.nodes():
        primary = theme_labels[node]
        result = [Membership(cluster_id=primary, weight=1.0, is_primary=True)]

        if primary != UNCLUSTERED:
            total_w = _weighted_degree(graph, node)
            if total_w > 0:
                affinity: dict[str, float] = defaultdict(float)
                for nb in graph.neighbors(node):
                    c = theme_labels[nb]
                    if c in (primary, UNCLUSTERED):
                        continue
                    affinity[c] += graph[node][nb].get("weight", 1.0)
                ranked = sorted(
                    ((c, w / total_w) for c, w in affinity.items()),
                    key=lambda x: x[1],
                    reverse=True,
                )
                for c, share in ranked:
                    if len(result) >= config.multi_membership_k:
                        break
                    if share >= config.multi_membership_tau:
                        result.append(Membership(cluster_id=c, weight=round(share, 4), is_primary=False))
        memberships[node] = result
    return memberships


def _build_clusters(graph, theme_labels, sector_labels, config: CommunityConfig) -> dict[str, Cluster]:
    members_by_cluster: dict[str, list[str]] = defaultdict(list)
    for node, c in theme_labels.items():
        members_by_cluster[c].append(node)

    clusters: dict[str, Cluster] = {}
    for cid, members in members_by_cluster.items():
        if cid == UNCLUSTERED:
            continue
        # parent sector = the sector most members fall into
        parent = Counter(sector_labels[m] for m in members).most_common(1)[0][0]
        reps = sorted(members, key=lambda m: _weighted_degree(graph, m), reverse=True)
        clusters[cid] = Cluster(
            cluster_id=cid,
            level="theme",
            parent_sector=parent,
            members=sorted(members),
            representatives=reps[: config.representatives_top_n],
        )
    return clusters


def detect_communities(graph, config: CommunityConfig | None = None):
    """Detect concept clusters on a fused concept graph.

    Returns ``(memberships, clusters)``:
      - memberships: ``dict[ticker, list[Membership]]`` (primary + secondary)
      - clusters: ``dict[cluster_id, Cluster]`` (theme level, with parent sector)
    """
    config = config or CommunityConfig()
    if graph.number_of_nodes() == 0:
        return {}, {}

    leidenalg, igraph = _require_leiden()
    ig, nodes = _to_igraph(graph, igraph)

    sector_part = _leiden(ig, config.sector_resolution, config.seed, leidenalg)
    sector_labels = {nodes[i]: f"SEC_{c}" for i, c in enumerate(sector_part)}

    theme_res = _search_theme_resolution(ig, config, leidenalg)
    theme_part = _leiden(ig, theme_res, config.seed, leidenalg)
    theme_labels = {nodes[i]: f"TH_{c}" for i, c in enumerate(theme_part)}
    theme_labels = _apply_min_cluster_size(graph, theme_labels, config)

    memberships = _multi_membership(graph, theme_labels, config)
    clusters = _build_clusters(graph, theme_labels, sector_labels, config)
    return memberships, clusters
