"""Cluster naming (G5): turn anonymous theme clusters into semantic labels.

Uses the framework's LLM factory (Gemini via structured output) to assign each
theme cluster a concise label and a parent sector, from its representative
tickers. Light task → defaults to a cheap Flash model. The LLM is injectable
so tests can pass a fake (no network).
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from .schemas import Cluster

DEFAULT_NAMING_MODEL = "gemini-3.1-flash-lite"


class _ClusterName(BaseModel):
    cluster_id: str
    label: str = Field(description="Concise English theme name, e.g. 'HBM/Storage' or 'AI Inference Chips'")
    parent_sector: str = Field(description="Broad English sector, e.g. 'Semiconductor', 'Energy'")


class _ClusterNames(BaseModel):
    clusters: list[_ClusterName]


def _build_prompt(clusters: dict[str, Cluster]) -> str:
    lines = [
        "You label stock-market concept clusters. For each cluster below, give a",
        "concise English theme `label` (the specific sub-theme its members share)",
        "and a broad English `parent_sector`. Use English only. Return every",
        "cluster_id exactly once.",
        "",
        "Clusters (cluster_id: representative tickers | all members):",
    ]
    for cid, c in clusters.items():
        reps = ", ".join(c.representatives)
        members = ", ".join(c.members)
        lines.append(f"- {cid}: reps=[{reps}] | members=[{members}]")
    return "\n".join(lines)


def name_clusters(
    clusters: dict[str, Cluster],
    provider: str = "google",
    model: str = DEFAULT_NAMING_MODEL,
    llm=None,
) -> dict[str, Cluster]:
    """Return clusters with ``label`` and semantic ``parent_sector`` filled.

    ``llm`` may be injected (must support ``with_structured_output``); otherwise
    a client is created via the framework factory using GOOGLE_API_KEY from env.
    """
    if not clusters:
        return dict(clusters)

    if llm is None:
        import os

        from tradingagents.llm_clients import create_llm_client

        client = create_llm_client(provider, model, google_api_key=os.getenv("GOOGLE_API_KEY"))
        llm = client.get_llm()

    structured = llm.with_structured_output(_ClusterNames)
    result = structured.invoke(_build_prompt(clusters))
    names = {n.cluster_id: n for n in result.clusters}

    out: dict[str, Cluster] = {}
    for cid, c in clusters.items():
        n = names.get(cid)
        out[cid] = (
            c.model_copy(update={"label": n.label, "parent_sector": n.parent_sector})
            if n
            else c
        )
    return out
