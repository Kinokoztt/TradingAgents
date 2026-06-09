"""Local-JSON persistence for concept-graph snapshots (M2 / G7).

Layout: ``{out_dir}/{as_of_date}/{edges,memberships,clusters}.json``. A
BigQuery results table can be added later; the service reads through this
store so the swap is transparent.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .schemas import Cluster, Membership

DEFAULT_OUT_DIR = "concept_graph_output"


def _snapshot_dir(as_of_date: str, out_dir: str) -> Path:
    return Path(out_dir) / as_of_date


def save_snapshot(
    as_of_date: str,
    edges: pd.DataFrame,
    memberships: dict[str, list[Membership]],
    clusters: dict[str, Cluster],
    out_dir: str = DEFAULT_OUT_DIR,
) -> Path:
    """Persist edges + memberships + clusters for ``as_of_date``."""
    path = _snapshot_dir(as_of_date, out_dir)
    path.mkdir(parents=True, exist_ok=True)

    edges.to_json(path / "edges.json", orient="records", indent=2)
    (path / "memberships.json").write_text(
        json.dumps(
            {tkr: [m.model_dump() for m in ms] for tkr, ms in memberships.items()},
            indent=2,
        )
    )
    save_clusters(as_of_date, clusters, out_dir)
    return path


def save_clusters(
    as_of_date: str,
    clusters: dict[str, Cluster],
    out_dir: str = DEFAULT_OUT_DIR,
) -> None:
    """Persist (or overwrite) just clusters.json, e.g. after naming."""
    path = _snapshot_dir(as_of_date, out_dir)
    path.mkdir(parents=True, exist_ok=True)
    (path / "clusters.json").write_text(
        json.dumps({cid: c.model_dump() for cid, c in clusters.items()}, indent=2)
    )


def load_memberships(as_of_date: str, out_dir: str = DEFAULT_OUT_DIR) -> dict[str, list[Membership]]:
    raw = json.loads((_snapshot_dir(as_of_date, out_dir) / "memberships.json").read_text())
    return {tkr: [Membership(**m) for m in ms] for tkr, ms in raw.items()}


def load_clusters(as_of_date: str, out_dir: str = DEFAULT_OUT_DIR) -> dict[str, Cluster]:
    raw = json.loads((_snapshot_dir(as_of_date, out_dir) / "clusters.json").read_text())
    return {cid: Cluster(**c) for cid, c in raw.items()}


def load_edges(as_of_date: str, out_dir: str = DEFAULT_OUT_DIR) -> pd.DataFrame:
    return pd.read_json(_snapshot_dir(as_of_date, out_dir) / "edges.json", orient="records")
