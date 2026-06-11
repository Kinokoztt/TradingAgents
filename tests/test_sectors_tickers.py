"""Tests for sector normalization and share-class canonicalization (A work)."""

import pytest

from tradingagents.concept_graph.sectors import CANONICAL_SECTORS, normalize_sector
from tradingagents.regime.tickers import canonical_ticker, canonicalize_tickers

pytestmark = pytest.mark.unit


def test_normalize_folds_subindustries_and_synonyms():
    assert normalize_sector("Semiconductor") == "Technology"
    assert normalize_sector("Semiconductors") == "Technology"
    assert normalize_sector("Information Technology") == "Technology"
    assert normalize_sector("Health Care") == "Healthcare"
    assert normalize_sector("Materials") == "Basic Materials"
    assert normalize_sector("Consumer Discretionary") == "Consumer Cyclical"
    assert normalize_sector("Consumer Staples") == "Consumer Defensive"
    assert normalize_sector("Telecom") == "Communication Services"


def test_normalize_is_case_insensitive_and_canonical_stable():
    assert normalize_sector("technology") == "Technology"
    for s in CANONICAL_SECTORS:
        assert normalize_sector(s) == s


def test_normalize_unknown_kept_visible():
    assert normalize_sector("Cryptocurrency") == "Cryptocurrency"
    assert normalize_sector(None) is None
    assert normalize_sector("") == ""


def test_canonical_ticker_collapses_share_classes():
    assert canonical_ticker("GOOGL") == "GOOG"
    assert canonical_ticker("goog") == "GOOG"
    assert canonical_ticker("BRK.A") == "BRK.B"
    assert canonical_ticker("AAPL") == "AAPL"  # untouched


def test_canonicalize_tickers_dedupes_preserving_order():
    assert canonicalize_tickers(["NVDA", "GOOG", "GOOGL", "AMD", "GOOGL"]) == ["NVDA", "GOOG", "AMD"]
    assert canonicalize_tickers(["GOOGL", "GOOG"]) == ["GOOG"]
