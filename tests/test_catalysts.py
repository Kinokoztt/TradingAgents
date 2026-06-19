"""Tests for line-1 structured catalyst transforms (pure, no network)."""

from __future__ import annotations

import pytest

from tradingagents.regime.catalysts import (
    Catalyst,
    CatalystType,
    dividends_to_catalysts,
    earnings_to_catalysts,
    grades_to_catalysts,
    mergers_to_catalysts,
    price_target_to_catalysts,
)
from tradingagents.regime.events import Certainty, Polarity

pytestmark = pytest.mark.unit

R = ("2026-05-01", "2026-05-31")


def test_earnings_beat_miss_and_future_skipped():
    rows = [
        {"date": "2026-07-30", "epsActual": None, "epsEstimated": 1.9},  # future -> skip
        {"date": "2026-05-05", "epsActual": 2.01, "epsEstimated": 1.95,
         "revenueActual": 111, "revenueEstimated": 109},
        {"date": "2026-05-06", "epsActual": 1.0, "epsEstimated": 1.5},
        {"date": "2026-01-29", "epsActual": 2.85, "epsEstimated": 2.67},  # out of range
    ]
    out = earnings_to_catalysts("AAPL", rows, *R)
    assert len(out) == 2
    beat = next(c for c in out if c.effective_date == "2026-05-05")
    assert beat.polarity is Polarity.POSITIVE
    assert beat.certainty is Certainty.CONFIRMED
    assert beat.data["eps_surprise"] == pytest.approx(0.06, abs=1e-6)
    miss = next(c for c in out if c.effective_date == "2026-05-06")
    assert miss.polarity is Polarity.NEGATIVE


def test_grades_only_up_down():
    rows = [
        {"date": "2026-05-05", "gradingCompany": "MS", "previousGrade": "Hold",
         "newGrade": "Buy", "action": "upgrade"},
        {"date": "2026-05-06", "gradingCompany": "GS", "previousGrade": "Buy",
         "newGrade": "Hold", "action": "downgrade"},
        {"date": "2026-05-07", "gradingCompany": "JPM", "previousGrade": "Buy",
         "newGrade": "Buy", "action": "maintain"},  # dropped
    ]
    out = grades_to_catalysts("AAPL", rows, *R)
    assert [c.polarity for c in out] == [Polarity.POSITIVE, Polarity.NEGATIVE]
    assert all(c.catalyst_type is CatalystType.ANALYST_GRADE for c in out)


def test_price_target_direction_and_upside():
    rows = [
        {"publishedDate": "2026-05-09T12:28:00.000Z", "newsTitle": "AAPL PT Raised to $350 at Maxim",
         "priceTarget": 350, "priceWhenPosted": 280, "analystCompany": "Maxim"},
        {"publishedDate": "2026-05-10T09:00:00.000Z", "newsTitle": "AAPL PT Lowered to $200 at XYZ",
         "priceTarget": 200, "priceWhenPosted": 250, "analystCompany": "XYZ"},
        {"publishedDate": "2026-05-11T09:00:00.000Z", "newsTitle": "XYZ Reiterates Overweight on AAPL",
         "priceTarget": 325, "priceWhenPosted": 300, "analystCompany": "XYZ"},
    ]
    out = price_target_to_catalysts("AAPL", rows, *R)
    assert out[0].polarity is Polarity.POSITIVE
    assert out[0].published_utc == "2026-05-09T12:28:00.000Z"
    assert out[0].data["implied_upside"] == pytest.approx(0.25, abs=1e-6)
    assert out[1].polarity is Polarity.NEGATIVE
    assert out[2].polarity is Polarity.NEUTRAL


def test_dividends_change_across_boundary():
    # newest-first like FMP; the in-range 0.27 must be diffed vs the prior 0.26.
    rows = [
        {"declarationDate": "2026-04-30", "date": "2026-05-11", "dividend": 0.27,
         "adjDividend": 0.27, "yield": 0.36, "frequency": "Quarterly"},
        {"declarationDate": "2026-01-29", "date": "2026-02-09", "dividend": 0.26, "frequency": "Quarterly"},
        {"declarationDate": "2025-10-30", "date": "2025-11-10", "dividend": 0.26, "frequency": "Quarterly"},
    ]
    out = dividends_to_catalysts("AAPL", rows, "2026-04-01", "2026-05-31")
    assert len(out) == 1
    assert out[0].effective_date == "2026-04-30"
    assert out[0].polarity is Polarity.POSITIVE
    assert out[0].data["change"] == "raised"


def test_mergers_role_polarity_and_universe_filter():
    rows = [{
        "symbol": "BIG", "companyName": "Big Co", "targetedSymbol": "SMALL",
        "targetedCompanyName": "Small Co", "transactionDate": "2026-05-10",
        "acceptedDate": "2026-05-10 16:06:28", "link": "http://x",
    }]
    out = mergers_to_catalysts(rows, {"BIG", "SMALL"}, *R)
    by_t = {c.ticker: c for c in out}
    assert by_t["SMALL"].polarity is Polarity.POSITIVE   # target gets the premium
    assert by_t["BIG"].polarity is Polarity.NEUTRAL      # acquirer ambiguous
    assert by_t["SMALL"].published_utc and by_t["SMALL"].published_utc.endswith("Z")
    # universe filter: a deal touching no in-universe ticker yields nothing
    assert mergers_to_catalysts(rows, {"OTHER"}, *R) == []


def test_to_dict_flattens_numerics():
    c = Catalyst("AAPL", CatalystType.EARNINGS, "2026-05-05", Polarity.POSITIVE,
                 Certainty.CONFIRMED, "s", "FMP:earnings", None, {"eps_surprise": 0.06})
    d = c.to_dict()
    assert d["catalyst_type"] == "Earnings"
    assert d["polarity"] == "Positive"
    assert d["certainty"] == "Confirmed"
    assert d["eps_surprise"] == 0.06  # flattened to top level
