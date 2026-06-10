"""Test FMP market-news pre-market cutoff visibility (backfill safety)."""

import pytest

from tradingagents.dataflows.fmp import _visible_premarket

pytestmark = pytest.mark.unit

CUTOFF = "2026-06-09 09:00:00"


def test_timestamped_kept_before_cutoff():
    assert _visible_premarket("2026-06-09 08:31:00", CUTOFF) is True


def test_timestamped_dropped_after_cutoff():
    assert _visible_premarket("2026-06-09 10:15:00", CUTOFF) is False


def test_prior_day_kept():
    assert _visible_premarket("2026-06-08 20:00:00", CUTOFF) is True
    assert _visible_premarket("2026-06-08", CUTOFF) is True  # date-only prior day


def test_same_day_date_only_dropped():
    # ambiguous on a backfill (can't tell pre/post open) -> drop
    assert _visible_premarket("2026-06-09", CUTOFF) is False


def test_empty_dropped():
    assert _visible_premarket("", CUTOFF) is False
