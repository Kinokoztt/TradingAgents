"""Concept-graph subsystem: build a per-stock correlation graph from
co-mention (news) and co-movement (price) signals, then surface dynamic
concept clusters for the regime gate (see docs/concept-graph-design.md).

M1 scope (this package so far): edge construction + fusion + pruning.
Community detection, naming, and the query service land in M2/M3.
"""

from .community import detect_communities
from .config import CommunityConfig, GraphConfig
from .naming import name_clusters
from .schemas import Cluster, Membership
from .service import (
    build_concept_graph,
    build_detect_save,
    detect_and_save,
    get_cluster_label,
    get_cluster_map,
    get_neighbors,
    name_and_save_clusters,
)

__all__ = [
    "GraphConfig",
    "CommunityConfig",
    "Membership",
    "Cluster",
    "build_concept_graph",
    "detect_communities",
    "detect_and_save",
    "build_detect_save",
    "name_clusters",
    "name_and_save_clusters",
    "get_cluster_map",
    "get_cluster_label",
    "get_neighbors",
]
