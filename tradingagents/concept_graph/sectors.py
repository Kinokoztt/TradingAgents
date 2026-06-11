"""Canonical sector taxonomy + normalization.

The cluster namer (G5) lets the LLM free-form a ``parent_sector``, which drifts
(e.g. "Semiconductor" as a peer of "Technology", "Health Care" vs "Healthcare").
That fragments the sector layer (S3) and double-counts overlapping sectors. We
pin a single 11-bucket taxonomy (Yahoo/Morningstar style, matching the existing
snapshots) and fold synonyms / sub-industries into it.
"""

from __future__ import annotations

# The 11 canonical sectors (matches the taxonomy already in the snapshots).
CANONICAL_SECTORS: tuple[str, ...] = (
    "Technology",
    "Financials",
    "Healthcare",
    "Consumer Cyclical",
    "Consumer Defensive",
    "Industrials",
    "Energy",
    "Basic Materials",
    "Communication Services",
    "Utilities",
    "Real Estate",
)

# Synonyms / sub-industries -> canonical. Keys are lower-cased.
_ALIASES: dict[str, str] = {
    # Technology (incl. semiconductors as a sub-industry, not a peer sector)
    "semiconductor": "Technology",
    "semiconductors": "Technology",
    "semiconductors & semiconductor equipment": "Technology",
    "semis": "Technology",
    "information technology": "Technology",
    "infotech": "Technology",
    "tech": "Technology",
    "software": "Technology",
    "hardware": "Technology",
    "it": "Technology",
    # Financials
    "financial services": "Financials",
    "financial": "Financials",
    "finance": "Financials",
    "banks": "Financials",
    "banking": "Financials",
    "insurance": "Financials",
    # Healthcare
    "health care": "Healthcare",
    "healthcare & pharmaceuticals": "Healthcare",
    "pharmaceuticals": "Healthcare",
    "pharma": "Healthcare",
    "biotech": "Healthcare",
    "biotechnology": "Healthcare",
    # Consumer Cyclical / Discretionary
    "consumer discretionary": "Consumer Cyclical",
    "consumer cyclicals": "Consumer Cyclical",
    "retail": "Consumer Cyclical",
    "automotive": "Consumer Cyclical",
    # Consumer Defensive / Staples
    "consumer staples": "Consumer Defensive",
    "consumer defensives": "Consumer Defensive",
    "staples": "Consumer Defensive",
    # Industrials
    "industrial": "Industrials",
    "aerospace & defense": "Industrials",
    "transportation": "Industrials",
    # Energy
    "oil & gas": "Energy",
    "oil and gas": "Energy",
    "energy infrastructure": "Energy",
    # Basic Materials
    "materials": "Basic Materials",
    "basic resources": "Basic Materials",
    "mining": "Basic Materials",
    "metals & mining": "Basic Materials",
    "chemicals": "Basic Materials",
    # Communication Services
    "communications": "Communication Services",
    "communication": "Communication Services",
    "telecommunications": "Communication Services",
    "telecom": "Communication Services",
    "media": "Communication Services",
    # Utilities
    "utility": "Utilities",
    # Real Estate
    "reit": "Real Estate",
    "reits": "Real Estate",
    "real estate investment trusts": "Real Estate",
}

_CANONICAL_LOWER = {s.lower(): s for s in CANONICAL_SECTORS}


def normalize_sector(name: str | None) -> str | None:
    """Map a free-form sector name to the canonical taxonomy.

    Exact (case-insensitive) canonical match wins, then the synonym map. Unknown
    names are returned title-cased (kept visible rather than silently merged).
    """
    if not name:
        return name
    key = name.strip().lower()
    if key in _CANONICAL_LOWER:
        return _CANONICAL_LOWER[key]
    if key in _ALIASES:
        return _ALIASES[key]
    return name.strip()
