"""Publisher -> reliability tier, and a tagger for NewsEvents.

The news pipeline keeps the publisher name on every article (Massive
``publisher.name``) but never used it. This is a small, hand-maintained tier
table so events from primary wires / top-tier outlets can be weighted above
aggregators and promotional sources. It is intentionally conservative and
static; a later iteration can *learn* reliability from realized price-in
outcomes (see docs/nn-pipeline-roadmap.md).

Matching is case-insensitive: exact publisher name first, then substring
(so "Reuters" matches "Thomson Reuters"). Unknown publishers stay Unknown
rather than being guessed.
"""

from __future__ import annotations

from .events import NewsEvent, SourceReliability

# Tier assignments. HIGH = primary wires + top-tier financial press;
# MEDIUM = mainstream analysis/aggregators with editorial standards;
# LOW = promotional / low-signal aggregators. Keys are matched lowercased.
_TIERS: dict[SourceReliability, tuple[str, ...]] = {
    SourceReliability.HIGH: (
        "reuters", "bloomberg", "the wall street journal", "wall street journal",
        "dow jones", "associated press", "financial times", "barron's", "barrons",
        "cnbc", "marketwatch", "the new york times",
        # primary press-release wires (the company's own official disclosure)
        "globenewswire", "business wire", "businesswire", "pr newswire", "prnewswire",
        "accesswire", "globe newswire",
    ),
    SourceReliability.MEDIUM: (
        "zacks", "the motley fool", "motley fool", "investing.com", "benzinga",
        "seeking alpha", "tipranks", "forbes", "yahoo", "investor's business daily",
        "investors business daily", "thefly", "the fly", "kiplinger", "morningstar",
    ),
    SourceReliability.LOW: (
        "simply wall st", "investorsobserver", "insider monkey", "etf daily news",
        "stocknews", "gurufocus", "247wallst", "24/7 wall st", "pennystocks",
        "newsfilecorp", "newsfile corp",
    ),
}

# Flattened lookup built once at import.
_EXACT: dict[str, SourceReliability] = {}
for _tier, _names in _TIERS.items():
    for _n in _names:
        _EXACT[_n] = _tier


def classify_source(publisher: str) -> SourceReliability:
    """Reliability tier for a publisher name (exact then substring match)."""
    if not publisher:
        return SourceReliability.UNKNOWN
    key = publisher.strip().lower()
    if key in _EXACT:
        return _EXACT[key]
    for name, tier in _EXACT.items():
        if name in key:
            return tier
    return SourceReliability.UNKNOWN


def tag_source_reliability(events: list[NewsEvent]) -> list[NewsEvent]:
    """Set ``source_reliability`` on each event in place; returns the list."""
    for ev in events:
        ev.source_reliability = classify_source(ev.source)
    return events
