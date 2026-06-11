"""Tests for L2 concept aggregation + catalyst propagation (no network/files)."""

import pytest

from tradingagents.concept_graph.schemas import Cluster, Membership
from tradingagents.regime import (
    Direction,
    Strength,
    StockSignal,
    aggregate_concepts,
    propagate_catalysts,
)

pytestmark = pytest.mark.unit


def _cluster_map():
    return {
        "NVDA": [Membership(cluster_id="TH_0", weight=1.0, is_primary=True)],
        "AMD": [Membership(cluster_id="TH_0", weight=1.0, is_primary=True)],
        "MU": [Membership(cluster_id="TH_0", weight=0.8, is_primary=True)],
        "XOM": [Membership(cluster_id="TH_1", weight=1.0, is_primary=True)],
    }


def _clusters():
    return {
        "TH_0": Cluster(cluster_id="TH_0", parent_sector="Technology",
                        members=["NVDA", "AMD", "MU"], representatives=["NVDA"], label="AI Chips"),
        "TH_1": Cluster(cluster_id="TH_1", parent_sector="Energy",
                        members=["XOM", "CVX"], representatives=["XOM"], label="Oil & Gas"),
    }


def test_aggregate_strong_coherent_bullish_cluster():
    signals = [
        StockSignal(ticker="NVDA", direction=Direction.LONG, catalyst_confidence=0.9, reason="beat"),
        StockSignal(ticker="AMD", direction=Direction.LONG, catalyst_confidence=0.8, reason="guide up"),
        StockSignal(ticker="MU", direction=Direction.LONG, catalyst_confidence=0.7, reason="HBM"),
    ]
    out = aggregate_concepts("2026-06-08", signals, cluster_map=_cluster_map(), clusters=_clusters())
    assert len(out) == 1
    cs = out[0]
    assert cs.cluster_id == "TH_0"
    assert cs.concept == "AI Chips"
    assert cs.strength is Strength.STRONG  # high conf + perfect coherence
    assert cs.member_tickers == ["NVDA", "AMD", "MU"]


def test_conflicting_directions_lower_strength():
    signals = [
        StockSignal(ticker="NVDA", direction=Direction.LONG, catalyst_confidence=0.8, reason="up"),
        StockSignal(ticker="AMD", direction=Direction.SHORT, catalyst_confidence=0.8, reason="down"),
        StockSignal(ticker="MU", direction=Direction.LONG, catalyst_confidence=0.2, reason="meh"),
    ]
    out = aggregate_concepts("2026-06-08", signals, cluster_map=_cluster_map(), clusters=_clusters())
    cs = out[0]
    assert cs.strength is not Strength.STRONG  # incoherent -> downgraded


def test_min_members_skips_singletons():
    signals = [StockSignal(ticker="XOM", direction=Direction.SHORT, catalyst_confidence=0.9, reason="oil")]
    out = aggregate_concepts("2026-06-08", signals, cluster_map=_cluster_map(), clusters=_clusters(), min_members=2)
    assert out == []  # TH_1 has only one signalled member


def test_block_signals_excluded():
    signals = [
        StockSignal(ticker="NVDA", direction=Direction.BLOCK, catalyst_confidence=0.1, reason="halt"),
        StockSignal(ticker="AMD", direction=Direction.BLOCK, catalyst_confidence=0.1, reason="halt"),
    ]
    out = aggregate_concepts("2026-06-08", signals, cluster_map=_cluster_map(), clusters=_clusters())
    assert out == []


def test_propagate_fills_neighbours_only():
    signals = [StockSignal(ticker="NVDA", direction=Direction.LONG, catalyst_confidence=0.9, reason="beat")]
    neighbors = {"NVDA": [("AMD", 0.8), ("MU", 0.4)], "AMD": [], "MU": []}
    out = propagate_catalysts(
        "2026-06-08", signals, neighbors_fn=lambda t: neighbors.get(t, []), decay=0.5, max_boost=0.3
    )
    by = {s.ticker: s for s in out}
    assert set(by) == {"NVDA", "AMD", "MU"}
    assert by["NVDA"].catalyst_confidence == 0.9            # source untouched
    assert by["AMD"].direction is Direction.LONG
    assert by["AMD"].catalyst_confidence == pytest.approx(0.3)   # 0.9*0.8*0.5=0.36 capped at 0.3
    assert by["MU"].catalyst_confidence == pytest.approx(0.18)   # 0.9*0.4*0.5


def test_propagate_does_not_override_existing():
    signals = [
        StockSignal(ticker="NVDA", direction=Direction.LONG, catalyst_confidence=0.9, reason="beat"),
        StockSignal(ticker="AMD", direction=Direction.SHORT, catalyst_confidence=0.5, reason="own view"),
    ]
    out = propagate_catalysts("2026-06-08", signals, neighbors_fn=lambda t: {"NVDA": [("AMD", 0.9)]}.get(t, []))
    by = {s.ticker: s for s in out}
    assert by["AMD"].direction is Direction.SHORT  # existing signal preserved
    assert by["AMD"].catalyst_confidence == 0.5


def test_propagate_skips_share_class_sibling_of_existing():
    # GOOG has a real signal; its graph neighbour GOOGL canonicalizes to GOOG,
    # so it must NOT be resurrected as a separate propagated signal.
    signals = [StockSignal(ticker="GOOG", direction=Direction.LONG, catalyst_confidence=0.9, reason="stake")]
    out = propagate_catalysts(
        "2026-06-08", signals, neighbors_fn=lambda t: {"GOOG": [("GOOGL", 0.9), ("MSFT", 0.8)]}.get(t, [])
    )
    tickers = {s.ticker for s in out}
    assert "GOOGL" not in tickers          # sibling collapsed
    assert tickers == {"GOOG", "MSFT"}     # genuine neighbour still propagated
