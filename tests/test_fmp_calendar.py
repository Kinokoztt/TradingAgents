"""Economic calendar point-in-time: blank `Actual` for events after the cutoff.

FMP's calendar is a live table — on a rollback it back-fills realized actuals for
events that have since occurred, leaking the macro prints being forecast. These
tests pin that the cutoff hides exactly those, while keeping estimate/previous.
"""

from __future__ import annotations

import pytest

from tradingagents.dataflows import fmp

_EVENTS = [
    # before 09:00 cutoff -> already released -> actual kept
    {"date": "2026-06-09 08:30:00", "event": "CPI", "actual": "0.4", "estimate": "0.3", "previous": "0.2", "impact": "High"},
    # same day, after cutoff -> not yet out -> actual blanked
    {"date": "2026-06-09 10:00:00", "event": "Fed Speak", "actual": "hawkish", "estimate": "", "previous": "", "impact": "Medium"},
    # future day -> actual blanked
    {"date": "2026-06-11 08:30:00", "event": "PPI", "actual": "0.5", "estimate": "0.4", "previous": "0.3", "impact": "High"},
]


@pytest.fixture(autouse=True)
def _stub(monkeypatch):
    monkeypatch.setattr(fmp, "_get", lambda endpoint, params: list(_EVENTS))


def _row(block: str, event: str) -> str:
    return next(ln for ln in block.splitlines() if f"| {event} |" in ln)


def test_cutoff_hides_future_actuals_keeps_estimates():
    block = fmp.get_economic_calendar("2026-06-09", "2026-06-16", cutoff="2026-06-09 09:00:00")

    cpi = _row(block, "CPI")
    assert "0.4" in cpi  # released pre-cutoff -> actual visible

    fed = _row(block, "Fed Speak")
    assert "hawkish" not in fed  # after cutoff -> actual hidden

    ppi = _row(block, "PPI")
    assert "0.5" not in ppi      # future -> actual hidden
    assert "0.4" in ppi          # but estimate kept
    assert "0.3" in ppi          # and previous kept


def test_no_cutoff_keeps_all_actuals():
    block = fmp.get_economic_calendar("2026-06-09", "2026-06-16")
    assert "hawkish" in block
    assert "0.5" in _row(block, "PPI")
