"""Point-in-time fundamentals builder: acceptedDate cutoff + self-computed ratios.

No network / BQ: FMP statements and the BigQuery price stats are both stubbed,
so these tests pin the leak-free filtering and the ratio arithmetic.
"""

from __future__ import annotations

import pytest

from tradingagents.market_tools.us import fundamentals as fund


def _income_q(date: str, accepted: str, *, revenue=100.0, net=10.0, gross=40.0,
              oi=20.0, ebitda=25.0, eps=0.5, shares=1000.0) -> dict:
    return {
        "date": date, "period": "Q1", "acceptedDate": accepted,
        "revenue": revenue, "netIncome": net, "grossProfit": gross,
        "operatingIncome": oi, "ebitda": ebitda, "epsDiluted": eps,
        "weightedAverageShsOutDil": shares,
    }


def _balance_q(date: str, accepted: str) -> dict:
    return {
        "date": date, "period": "Q1", "acceptedDate": accepted,
        "totalStockholdersEquity": 500.0, "totalAssets": 1000.0, "totalDebt": 200.0,
        "totalCurrentAssets": 300.0, "totalCurrentLiabilities": 150.0,
        "cashAndCashEquivalents": 50.0,
    }


def _cashflow_q(date: str, accepted: str) -> dict:
    return {"date": date, "period": "Q1", "acceptedDate": accepted, "freeCashFlow": 30.0}


# 5 quarters, newest first. The most recent (2026-03-31) is accepted 2026-04-25.
_QUARTERS = [
    ("2026-03-31", "2026-04-25 16:30:00"),
    ("2025-12-31", "2026-01-28 16:30:00"),
    ("2025-09-30", "2025-10-28 16:30:00"),
    ("2025-06-30", "2025-07-25 16:30:00"),
    ("2025-03-31", "2025-04-25 16:30:00"),
]

_STMTS = {
    "income": [_income_q(d, a) for d, a in _QUARTERS],
    "balance": [_balance_q(d, a) for d, a in _QUARTERS],
    "cashflow": [_cashflow_q(d, a) for d, a in _QUARTERS],
}

_PX = {"as_of": "2026-06-08", "price": 50.0, "ma50": 48.0, "ma200": 45.0, "hi52": 60.0, "lo52": 30.0}


@pytest.fixture(autouse=True)
def _stub(monkeypatch):
    monkeypatch.setattr(fund.fmp, "get_financial_statements", lambda *a, **k: _STMTS)
    monkeypatch.setattr(fund, "_price_stats", lambda ticker, session: dict(_PX))


def test_visible_drops_missing_or_future_accepted():
    cutoff = "2026-06-09 09:00:00"
    reports = [
        {"date": "2026-03-31", "acceptedDate": "2026-04-25 16:30:00"},
        {"date": "2026-06-30", "acceptedDate": "2026-07-25 16:30:00"},  # future -> drop
        {"date": "2025-12-31", "acceptedDate": None},                   # no timestamp -> drop
    ]
    out = fund._visible(reports, cutoff)
    assert [r["date"] for r in out] == ["2026-03-31"]


def test_point_in_time_cutoff_changes_latest_filing():
    # Session after the Q1-2026 filing: it is visible.
    block_after = fund.get_fundamentals("TEST", "2026-06-09")
    assert "period 2026-03-31" in block_after

    # Session before that filing was accepted (2026-04-25): Q1-2026 not yet public.
    block_before = fund.get_fundamentals("TEST", "2026-04-10")
    assert "period 2025-12-31" in block_before
    assert "2026-03-31" not in block_before


def test_ratio_math():
    m = fund._compute_metrics(_STMTS["income"], _STMTS["balance"], _STMTS["cashflow"], dict(_PX))
    assert m["ttm_quarters"] == 4
    assert m["revenue_ttm"] == pytest.approx(400.0)
    assert m["eps_ttm"] == pytest.approx(2.0)
    assert m["gross_margin"] == pytest.approx(0.40)
    assert m["operating_margin"] == pytest.approx(0.20)
    assert m["net_margin"] == pytest.approx(0.10)
    assert m["roe"] == pytest.approx(0.08)
    assert m["roa"] == pytest.approx(0.04)
    assert m["debt_to_equity"] == pytest.approx(0.40)
    assert m["current_ratio"] == pytest.approx(2.0)
    assert m["book_value_per_share"] == pytest.approx(0.5)
    assert m["pe_ttm"] == pytest.approx(25.0)
    assert m["price_to_book"] == pytest.approx(100.0)
    assert m["market_cap"] == pytest.approx(50000.0)


def test_negative_eps_yields_na_pe():
    income = [_income_q(d, a, net=-10.0, eps=-0.5) for d, a in _QUARTERS]
    m = fund._compute_metrics(income, _STMTS["balance"], _STMTS["cashflow"], dict(_PX))
    assert m["eps_ttm"] == pytest.approx(-2.0)
    assert m["pe_ttm"] is None  # trailing loss -> no meaningful P/E


def test_no_visible_filing_returns_skip_message():
    # All filings are accepted in 2025+, a 2024 session sees none.
    block = fund.get_fundamentals("TEST", "2024-01-15")
    assert "No point-in-time fundamentals" in block


def test_partial_ttm_when_fewer_than_4_quarters():
    m = fund._compute_metrics(_STMTS["income"][:3], _STMTS["balance"], _STMTS["cashflow"], None)
    assert m["ttm_quarters"] == 3
    assert m["revenue_ttm"] is None
    assert m["net_margin"] is None
    assert "price" not in m  # px None -> no price-derived fields
