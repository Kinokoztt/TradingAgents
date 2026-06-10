"""Test regime report local persistence roundtrip (no network)."""

import pytest

from tradingagents.regime import Direction, MarketRegime, RegimeReport, StockSignal
from tradingagents.regime.store import load_report, save_report

pytestmark = pytest.mark.unit


def test_save_load_roundtrip(tmp_path):
    report = RegimeReport(
        as_of_date="2026-06-08",
        market_state=MarketRegime.BULLISH,
        macro_summary="利率平稳，VIX 偏低。",  # unicode must survive
        stock_signals=[StockSignal(ticker="NVDA", direction=Direction.LONG, catalyst_confidence=0.8, reason="beat")],
    )
    path = save_report("2026-06-08", report, out_dir=str(tmp_path))
    assert path.endswith("2026-06-08/regime_report.json")

    restored = load_report("2026-06-08", out_dir=str(tmp_path))
    assert restored.market_state is MarketRegime.BULLISH
    assert restored.macro_summary == "利率平稳，VIX 偏低。"
    assert restored.long_whitelist == ["NVDA"]
