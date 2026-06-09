"""Pydantic schemas for the concept-graph query interface (M2)."""

from __future__ import annotations

from pydantic import BaseModel


class Membership(BaseModel):
    """A ticker's membership in one concept cluster.

    ``weight`` is 1.0 for the primary (hard-partition) cluster and the
    edge-weight share (affinity) for secondary clusters.
    """

    cluster_id: str
    weight: float
    is_primary: bool


class Cluster(BaseModel):
    """A theme-level concept cluster (the consumable L_theme layer)."""

    cluster_id: str
    level: str = "theme"
    parent_sector: str
    members: list[str]
    representatives: list[str]      # top weighted-degree members
    label: str | None = None        # filled by G5 naming (Gemini); None until then
