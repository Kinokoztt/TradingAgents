"""Publisher -> reliability tier, source filtering, and rule-based certainty.

Every article keeps its publisher name; this small, hand-maintained table lets us
(1) weight/keep events from primary wires + real journalism above opinion mills,
(2) drop low-signal sources from the NN corpus, and (3) derive ``certainty``
deterministically from the source instead of asking the LLM (a primary corporate
disclosure is Confirmed; a media report is Unconfirmed).

Matching is case-insensitive: exact publisher name first, then substring (so
"zacks" matches "Zacks Investment Research"). Unknown publishers stay Unknown.
Tiers reflect the FMP ``news/stock`` publisher mix; opinion mills (Motley Fool,
24/7 Wall Street, GuruFocus, Invezz, Finbold, …) are LOW on purpose.
"""

from __future__ import annotations

from .events import Certainty, NewsEvent, SourceReliability

# Tier assignments (keys matched lowercased). HIGH = primary wires + top-tier
# financial press; MEDIUM = mainstream analysis/journalism with editorial
# standards; LOW = promotional / opinion-mill / low-signal aggregators.
_TIERS: dict[SourceReliability, tuple[str, ...]] = {
    SourceReliability.HIGH: (
        "reuters", "bloomberg", "the wall street journal", "wall street journal", "wsj",
        "dow jones", "associated press", "financial times", "barron's", "barrons",
        "cnbc", "cnbc television", "marketwatch", "market watch", "the new york times",
        # primary press-release wires (the company's own official disclosure)
        "globenewswire", "globe newswire", "business wire", "businesswire",
        "pr newswire", "prnewswire", "accesswire",
    ),
    SourceReliability.MEDIUM: (
        "zacks", "benzinga", "seeking alpha", "tipranks", "forbes", "yahoo",
        "investor's business daily", "investors business daily", "thefly", "the fly",
        "kiplinger", "morningstar", "investing.com", "investopedia", "techcrunch",
        "business insider", "proactive investors", "schwab network", "cnet",
        "reuters - finance",
    ),
    SourceReliability.LOW: (
        # opinion mills / listicle factories
        "the motley fool", "motley fool", "fool - investing news", "fool.com",
        "24/7 wall street", "24/7 wall st", "247 wallst", "247wallst",
        "gurufocus", "invezz", "finbold", "stocktwits",
        # promotional / low-signal aggregators
        "simply wall st", "investorsobserver", "insider monkey", "etf daily news",
        "stocknews", "247wallst", "pennystocks", "newsfilecorp", "newsfile corp",
    ),
}

# Primary-disclosure wires: an item from these is a company's own filing/release,
# so it is treated as Confirmed regardless of tier. Everything else is a report.
_PRIMARY_WIRES: tuple[str, ...] = (
    "globenewswire", "globe newswire", "business wire", "businesswire",
    "pr newswire", "prnewswire", "accesswire", "newsfile corp", "newsfilecorp",
)

_TIER_RANK = {
    SourceReliability.UNKNOWN: 0,
    SourceReliability.LOW: 1,
    SourceReliability.MEDIUM: 2,
    SourceReliability.HIGH: 3,
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


def tier_rank(tier: SourceReliability) -> int:
    """Orderable rank (HIGH=3 .. UNKNOWN=0) for min-tier comparisons."""
    return _TIER_RANK[tier]


def meets_min_tier(publisher: str, min_tier: SourceReliability) -> bool:
    """True if ``publisher`` classifies at or above ``min_tier``."""
    return tier_rank(classify_source(publisher)) >= tier_rank(min_tier)


def certainty_for_source(publisher: str) -> Certainty:
    """Rule-based certainty: a primary-wire disclosure is Confirmed; any other
    publisher (journalism/aggregator/opinion) is a report -> Unconfirmed.

    Deterministic on purpose — the LLM cannot reliably separate confirmed facts
    from reported claims, so certainty is derived from the source, not generated.
    """
    if not publisher:
        return Certainty.UNCONFIRMED
    key = publisher.strip().lower()
    if any(wire in key for wire in _PRIMARY_WIRES):
        return Certainty.CONFIRMED
    return Certainty.UNCONFIRMED


def tag_source_reliability(events: list[NewsEvent]) -> list[NewsEvent]:
    """Set ``source_reliability`` on each event in place; returns the list."""
    for ev in events:
        ev.source_reliability = classify_source(ev.source)
    return events
