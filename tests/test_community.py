"""Community detection (M2) tests on a synthetic planted-cluster graph.

leidenalg/python-igraph are optional, so skip if not installed. The persistence
round-trip test is pure-Python and always runs.
"""

import importlib.util

import pandas as pd
import pytest

from tradingagents.concept_graph import store
from tradingagents.concept_graph.config import UNCLUSTERED, CommunityConfig
from tradingagents.concept_graph.schemas import Cluster, Membership

pytestmark = pytest.mark.unit

_HAS_LEIDEN = (
    importlib.util.find_spec("leidenalg") is not None
    and importlib.util.find_spec("igraph") is not None
)
requires_leiden = pytest.mark.skipif(
    not _HAS_LEIDEN, reason="leidenalg/python-igraph not installed"
)


def _planted_graph():
    """Three dense 4-cliques (themes) with a couple of weak bridges."""
    import networkx as nx

    g = nx.Graph()
    groups = [["A1", "A2", "A3", "A4"], ["B1", "B2", "B3", "B4"], ["C1", "C2", "C3", "C4"]]
    for grp in groups:
        for i in range(len(grp)):
            for j in range(i + 1, len(grp)):
                g.add_edge(grp[i], grp[j], weight=0.9)
    # weak inter-group bridges (should not merge clusters)
    g.add_edge("A1", "B1", weight=0.05)
    g.add_edge("B4", "C1", weight=0.05)
    return g, groups


def _cfg():
    # small graph: aim for ~3 theme clusters
    return CommunityConfig(min_cluster_size=3, theme_target_min=2, theme_target_max=4, multi_membership_tau=0.1)


@requires_leiden
def test_detect_recovers_planted_clusters():
    from tradingagents.concept_graph.community import detect_communities

    g, groups = _planted_graph()
    memberships, clusters = detect_communities(g, _cfg())

    # each planted group should share one primary cluster, distinct across groups
    primaries = {}
    for grp in groups:
        prim = {next(m.cluster_id for m in memberships[n] if m.is_primary) for n in grp}
        assert len(prim) == 1, f"group {grp} split across clusters: {prim}"
        primaries[grp[0]] = prim.pop()
    assert len(set(primaries.values())) == 3  # three distinct themes
    assert all(c.parent_sector.startswith("SEC_") for c in clusters.values())


@requires_leiden
def test_primary_weight_is_one_and_sorted():
    from tradingagents.concept_graph.community import detect_communities

    g, _ = _planted_graph()
    memberships, _ = detect_communities(g, _cfg())
    for ms in memberships.values():
        assert ms[0].is_primary and ms[0].weight == 1.0
        secondary = [m.weight for m in ms[1:]]
        assert secondary == sorted(secondary, reverse=True)
        assert len(ms) <= _cfg().multi_membership_k


@requires_leiden
def test_min_cluster_size_unclusters_tiny_groups():
    from tradingagents.concept_graph.community import detect_communities

    import networkx as nx

    g = nx.Graph()
    for i in range(5):
        for j in range(i + 1, 5):
            g.add_edge(f"X{i}", f"X{j}", weight=0.9)
    g.add_edge("Y1", "Y2", weight=0.9)  # 2-node group below min_cluster_size=3
    memberships, clusters = detect_communities(g, _cfg())
    y_primary = next(m.cluster_id for m in memberships["Y1"] if m.is_primary)
    assert y_primary == UNCLUSTERED  # no valid cluster to join
    assert all(cid != UNCLUSTERED for cid in clusters)


def test_name_clusters_fills_label_and_sector():
    from tradingagents.concept_graph.naming import _ClusterName, _ClusterNames, name_clusters

    class _FakeStructured:
        def invoke(self, prompt):
            return _ClusterNames(clusters=[
                _ClusterName(cluster_id="TH_1", label="存储/HBM", parent_sector="Semiconductor"),
            ])

    class _FakeLLM:
        def with_structured_output(self, schema):
            return _FakeStructured()

    clusters = {
        "TH_1": Cluster(cluster_id="TH_1", parent_sector="SEC_2",
                        members=["MU", "WDC"], representatives=["MU"]),
        "TH_9": Cluster(cluster_id="TH_9", parent_sector="SEC_3",
                        members=["JPM", "BAC"], representatives=["JPM"]),
    }
    out = name_clusters(clusters, llm=_FakeLLM())
    assert out["TH_1"].label == "存储/HBM"
    assert out["TH_1"].parent_sector == "Semiconductor"
    # cluster missing from LLM response is left unchanged (no crash)
    assert out["TH_9"].label is None
    assert out["TH_9"].parent_sector == "SEC_3"


def test_gcs_upload_snapshot(tmp_path):
    from unittest.mock import MagicMock, patch

    from tradingagents.concept_graph import gcs

    snap = tmp_path / "2026-06-05"
    snap.mkdir(parents=True)
    for name in ("edges.json", "memberships.json", "clusters.json"):
        (snap / name).write_text("{}")

    blobs = []
    fake_bucket = MagicMock()
    fake_bucket.blob.side_effect = lambda p: blobs.append(p) or MagicMock()
    fake_client = MagicMock()
    fake_client.bucket.return_value = fake_bucket

    fake_storage = MagicMock()
    fake_storage.Client.return_value = fake_client
    # the function does `from google.cloud import storage`; override that submodule
    with patch.dict("sys.modules", {"google.cloud.storage": fake_storage}):
        uris = gcs.upload_snapshot("2026-06-05", "my-bucket", prefix="cg", out_dir=str(tmp_path))

    assert uris == [
        "gs://my-bucket/cg/2026-06-05/edges.json",
        "gs://my-bucket/cg/2026-06-05/memberships.json",
        "gs://my-bucket/cg/2026-06-05/clusters.json",
    ]
    assert blobs == [
        "cg/2026-06-05/edges.json",
        "cg/2026-06-05/memberships.json",
        "cg/2026-06-05/clusters.json",
    ]


def test_store_roundtrip(tmp_path):
    edges = pd.DataFrame(
        {"src": ["A", "A"], "dst": ["B", "C"], "weight": [0.8, 0.5],
         "comovement": [0.7, 0.4], "comention": [0.1, 0.1]}
    )
    memberships = {
        "A": [Membership(cluster_id="TH_0", weight=1.0, is_primary=True)],
        "B": [Membership(cluster_id="TH_0", weight=1.0, is_primary=True),
              Membership(cluster_id="TH_1", weight=0.3, is_primary=False)],
    }
    clusters = {
        "TH_0": Cluster(cluster_id="TH_0", parent_sector="SEC_0",
                        members=["A", "B"], representatives=["A"]),
    }
    store.save_snapshot("2026-06-05", edges, memberships, clusters, out_dir=str(tmp_path))

    assert store.load_memberships("2026-06-05", str(tmp_path))["B"][1].cluster_id == "TH_1"
    assert store.load_clusters("2026-06-05", str(tmp_path))["TH_0"].parent_sector == "SEC_0"
    assert len(store.load_edges("2026-06-05", str(tmp_path))) == 2
