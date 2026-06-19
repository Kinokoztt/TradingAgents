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

import re

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


# Securities class-action *solicitation* press releases flood the wires
# (GlobeNewswire/PRNewswire): law firms blast near-duplicate "lead plaintiff
# deadline / investors urged to contact" notices for every name that dropped.
# They are promotional, not company disclosures, get marked Confirmed (wire),
# and duplicate heavily — pure noise for the event corpus. Drop them pre-LLM.
# The signatures below essentially never appear in genuine legal *reporting*
# (e.g. a DOJ/SEC action or a trial covered by Reuters), so real legal events
# are kept.
_LAW_FIRMS: tuple[str, ...] = (
    "rosen law", "hagens berman", "pomerantz", "bragar eagel", "levi & korsinsky",
    "levi and korsinsky", "berger montague", "glancy prongay", "robbins geller",
    "kessler topaz", "faruqi", "bernstein liebhard", "kirby mcinerney", "schall law",
    "johnson fistel", "kahn swick", "scott+scott", "block & leviton", "gross law",
    "howard g. smith", "holzer", "kaskela", "bronstein, gewirtz", "wolf haldenstein",
    "labaton", "pawar law", "thornton law", "the law offices of", "rigrodsky",
)

_SOLICITATION_RE = re.compile(
    r"lead plaintiff"
    r"|class period"
    r"|securities class action"
    r"|investors?\s+who\s+(?:purchased|bought|acquired)"
    r"|investors?\s+(?:are\s+)?(?:encouraged|urged|reminded|notified|alerted)"
    r"|(?:encourages|urges|reminds|notifies|alerts)\s+(?:[\w.,&'\- ]+?\s+)?investors",
    re.IGNORECASE,
)


def is_litigation_solicitation(text: str) -> bool:
    """True if ``text`` (article title + snippet) is a securities class-action
    solicitation / shareholder-alert press release (noise to drop pre-LLM).

    Three nets: (1) a named plaintiff law firm, (2) explicit solicitation phrasing
    (lead plaintiff / investors urged to contact / class period), (3) the
    securities-class-action genre itself ("class action" + "securit*"/"10b-5").
    Genuine legal *reporting* — a DOJ/SEC charge, a product-liability or injury
    suit, an antitrust trial — matches none of these and is kept.
    """
    if not text:
        return False
    low = text.lower()
    if any(firm in low for firm in _LAW_FIRMS):
        return True
    if _SOLICITATION_RE.search(text):
        return True
    if "class action" in low and ("securit" in low or "10b-5" in low):
        return True
    return False


def tag_source_reliability(events: list[NewsEvent]) -> list[NewsEvent]:
    """Set ``source_reliability`` on each event in place; returns the list."""
    for ev in events:
        ev.source_reliability = classify_source(ev.source)
    return events
