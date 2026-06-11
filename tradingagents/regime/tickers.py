"""Share-class canonicalization for the regime gate.

Dual-class names (GOOG/GOOGL, BRK.A/BRK.B, …) reference the same underlying
company and share the same catalyst, yet news/universe carry both — so L1 would
analyse and whitelist them twice. We collapse known siblings to one canonical
ticker (the more liquid / primary class). This is a curated map: there is no
reliable way to detect share-class pairs from the symbol alone.
"""

from __future__ import annotations

# sibling ticker -> canonical (primary class). Canonicals omitted (map to self).
SHARE_CLASS_CANONICAL: dict[str, str] = {
    "GOOGL": "GOOG",
    "BRK.A": "BRK.B",
    "BRKA": "BRK.B",
    "BRKB": "BRK.B",
    "BF.A": "BF.B",
    "BF.B": "BF.B",
    "LEN.B": "LEN",
    "HEI.A": "HEI",
    "LBRDK": "LBRDA",
    "FOX": "FOXA",
    "NWS": "NWSA",
    "UAA": "UA",
    "GEF.B": "GEF",
}


def canonical_ticker(ticker: str) -> str:
    """Map a ticker to its canonical share class (identity if not a known sibling)."""
    return SHARE_CLASS_CANONICAL.get(ticker.upper(), ticker.upper())


def canonicalize_tickers(tickers: list[str]) -> list[str]:
    """Canonicalize share classes and dedupe, preserving first-seen order."""
    seen: dict[str, None] = {}
    for t in tickers:
        seen.setdefault(canonical_ticker(t), None)
    return list(seen)
