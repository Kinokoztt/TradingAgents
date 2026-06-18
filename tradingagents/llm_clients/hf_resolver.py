"""Resolve catalog entries to the *current* best Hugging Face repo, live.

Pinning a repo id in the catalog goes stale fast (new quants/generations land
weekly). Instead, each catalog entry carries a family query + quant preference,
and this module queries the HF Hub at runtime to pick the newest/most-downloaded
matching build. It also exposes a `discover` helper to browse what's trending.

The ranking logic (`rank_candidates`) is a pure function so it is unit-testable
without network; only `resolve_latest` / `discover` hit the Hub API.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from . import config
from .local_models import LocalModelSpec

# Quant tokens we recognize in repo ids, used to derive prefer_terms when a
# spec doesn't set them explicitly.
_QUANT_TOKENS = ("awq", "gptq", "int4", "int8", "mxfp4", "fp8", "marlin", "autoround")

# HfApi.list_models sort keys (string values accepted by the Hub API).
_SORT_KEYS = {
    "trending": "trending_score",
    "downloads": "downloads",
    "likes": "likes",
    "modified": "last_modified",
    "created": "created_at",
}


@dataclass(frozen=True)
class Candidate:
    """A resolved HF repo with the signals used to rank it."""

    repo_id: str
    downloads: int = 0
    likes: int = 0
    last_modified: str = ""


def _tokens(text: str) -> list[str]:
    """Lowercased alphanumeric tokens (e.g. 'Qwen3-32B' -> ['qwen3','32b'])."""
    return [t for t in re.split(r"[\s/_\-]+", text.lower()) if t]


def effective_query(spec: LocalModelSpec) -> str:
    """HF search string for ``spec`` (explicit, else derived from served name)."""
    return spec.hf_query or " ".join(_tokens(spec.served_name))


def effective_match_terms(spec: LocalModelSpec) -> tuple[str, ...]:
    """Substrings a candidate repo id must contain (lowercased)."""
    if spec.match_terms:
        return tuple(t.lower() for t in spec.match_terms)
    # Derive from served-name tokens, dropping a generic trailing quant token.
    toks = [t for t in _tokens(spec.served_name) if t not in _QUANT_TOKENS]
    return tuple(toks)


def effective_prefer_terms(spec: LocalModelSpec) -> tuple[str, ...]:
    """Quant tokens to rank higher (explicit, else derived from ``quant``)."""
    if spec.prefer_terms:
        return tuple(t.lower() for t in spec.prefer_terms)
    return tuple(t for t in _QUANT_TOKENS if t in spec.quant.lower())


def is_repo_compatible(repo_id: str, blocked_terms: tuple[str, ...]) -> bool:
    """False if the repo id contains any hardware-incompatible quant token."""
    rid = repo_id.lower()
    return not any(t in rid for t in blocked_terms)


def rank_candidates(
    candidates: list[Candidate],
    match_terms: tuple[str, ...],
    prefer_terms: tuple[str, ...],
    blocked_terms: tuple[str, ...] = (),
) -> list[Candidate]:
    """Filter to repos matching ALL ``match_terms`` and NONE of ``blocked_terms``
    (incompatible quant formats), then rank by (#prefer hits, downloads, likes).
    Pure — no network."""
    matched = [
        c for c in candidates
        if all(t in c.repo_id.lower() for t in match_terms)
        and is_repo_compatible(c.repo_id, blocked_terms)
    ]

    def score(c: Candidate) -> tuple[int, int, int]:
        rid = c.repo_id.lower()
        prefer_hits = sum(1 for t in prefer_terms if t in rid)
        return (prefer_hits, c.downloads, c.likes)

    return sorted(matched, key=score, reverse=True)


def _to_candidate(info) -> Candidate:
    return Candidate(
        repo_id=info.id,
        downloads=getattr(info, "downloads", 0) or 0,
        likes=getattr(info, "likes", 0) or 0,
        last_modified=str(getattr(info, "last_modified", "") or ""),
    )


def resolve_candidates(
    spec: LocalModelSpec, *, limit: int = 50, blocked_terms: tuple[str, ...] | None = None,
) -> list[Candidate]:
    """Live HF search for ``spec``, ranked best-first and filtered to checkpoints
    compatible with this box's GPU. Empty if nothing compatible matches.

    ``blocked_terms`` defaults to ``config.blocked_quant_terms()`` (derived from
    ``gpu_arch``); pass an explicit tuple to override.
    """
    from huggingface_hub import HfApi

    if blocked_terms is None:
        blocked_terms = config.blocked_quant_terms()

    infos = HfApi().list_models(
        search=effective_query(spec),
        sort="downloads",  # descending by default in current huggingface_hub
        limit=limit,
        full=True,  # populate last_modified so the recency signal is visible
    )
    candidates = [_to_candidate(i) for i in infos]
    return rank_candidates(
        candidates, effective_match_terms(spec), effective_prefer_terms(spec), blocked_terms,
    )


def resolve_latest(
    spec: LocalModelSpec, *, limit: int = 50, blocked_terms: tuple[str, ...] | None = None,
) -> Candidate | None:
    """Best GPU-compatible HF repo for ``spec``, or None if none match."""
    ranked = resolve_candidates(spec, limit=limit, blocked_terms=blocked_terms)
    return ranked[0] if ranked else None


def discover(*, sort: str = "downloads", task: str = "text-generation", limit: int = 20,
             query: str = "") -> list[Candidate]:
    """Browse the Hub: top models for ``task`` by ``sort``.

    ``downloads`` is the default since it is supported across all hub versions;
    ``trending`` (trending_score) is newer and may not be honored by older
    clients — passing it then is the caller's choice and will fail loudly.
    """
    from huggingface_hub import HfApi

    infos = HfApi().list_models(
        search=query or None,
        filter=task,
        sort=_SORT_KEYS.get(sort, sort),  # descending by default in current huggingface_hub
        limit=limit,
        full=True,
    )
    return [_to_candidate(i) for i in infos]
