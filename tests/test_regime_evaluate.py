"""Tests for Module A post-hoc evaluation (pure, no network)."""

import pandas as pd
import pytest

from tradingagents.regime import (
    ConceptSignal,
    Direction,
    HorizonOutlook,
    MarketRegime,
    RegimeReport,
    StockSignal,
    evaluate_report,
)

pytestmark = pytest.mark.unit

# 8 business days; session = first. Open-anchor: baseline = open[i0]; horizon N
# exits at close[i0+N-1] (session itself = h1). SPY rises => Bullish hits.
CALENDAR = pd.bdate_range("2026-06-08", periods=8)

# (open[0], close[...]) per ticker — only open[0] is the baseline; closes drive exits.
OPEN0 = {"SPY": 100, "WIN": 50, "LOSE": 50, "DOWN": 80, "BLK": 10}
CLOSES = {
    "SPY":  [102, 103, 104, 105, 106, 107, 108, 109],   # h1=+2% h3=104/100=+4% h5=106/100=+6%
    "WIN":  [53, 55, 57, 59, 62, 64, 66, 68],           # Long, beats proxy (h1=+6%)
    "LOSE": [49, 48, 47, 46, 45, 44, 43, 42],           # Long, falls (h1=-2%)
    "DOWN": [78, 76, 74, 72, 70, 68, 66, 64],           # Short, falls (h1=-2.5%)
    "BLK":  [10.5, 11, 11, 11, 11, 11, 11, 11],         # Block, rose (h1=+5%)
}


def _price_df(open0=OPEN0, closes=CLOSES, calendar=CALENDAR) -> pd.DataFrame:
    rows = []
    for ticker, cl in closes.items():
        for i, (dt, c) in enumerate(zip(calendar, cl)):
            # open only matters at i0; use the baseline open there, close elsewhere.
            op = open0[ticker] if i == 0 else c
            rows.append({"ticker": ticker, "trade_date": dt, "open": float(op), "close": float(c)})
    return pd.DataFrame(rows)


def _report() -> RegimeReport:
    return RegimeReport(
        as_of_date="2026-06-08",
        market_state=MarketRegime.BULLISH,
        macro_summary="x",
        stock_signals=[
            StockSignal(ticker="WIN", direction=Direction.LONG, catalyst_confidence=0.9, reason="r"),
            StockSignal(ticker="LOSE", direction=Direction.LONG, catalyst_confidence=0.3, reason="r"),
            StockSignal(ticker="DOWN", direction=Direction.SHORT, catalyst_confidence=0.7, reason="r"),
            StockSignal(ticker="BLK", direction=Direction.BLOCK, catalyst_confidence=0.0, reason="r"),
        ],
        concept_signals=[
            ConceptSignal(concept="Sec", level="sector", direction=Direction.LONG,
                          confidence=0.6, member_tickers=["WIN", "LOSE"]),
            ConceptSignal(concept="Thm", level="theme", direction=Direction.SHORT,
                          confidence=0.6, member_tickers=["DOWN"]),
        ],
    )


def test_market_state_and_whitelists():
    sc = evaluate_report(_report(), _price_df(), proxy="SPY", horizons=(1, 3, 5))
    assert sc.complete is True
    h1 = sc.horizons[0]

    assert h1.target_date == "2026-06-08"  # h1 exits at the session's own close
    assert h1.market_return == pytest.approx(0.02)
    assert h1.market_hit is True

    # long bucket: WIN(+6%) hit, LOSE(-2%) miss -> hit_rate 0.5, WIN beats proxy
    assert h1.long.count == 2
    assert h1.long.hit_rate == pytest.approx(0.5)
    assert h1.long.win_rate == pytest.approx(0.5)  # only WIN beats SPY
    assert h1.long.avg_alpha is not None

    # short bucket: DOWN fell and underperformed -> hit + win
    assert h1.short.count == 1
    assert h1.short.hit_rate == pytest.approx(1.0)
    assert h1.short.win_rate == pytest.approx(1.0)
    assert h1.short.avg_alpha == pytest.approx(0.02 - (78 / 80 - 1))


def test_confusion_precision_and_brier():
    h1 = evaluate_report(_report(), _price_df(), horizons=(1,)).horizons[0]
    assert h1.dir_confusion["Long"] == {"up": 1, "down": 1}   # WIN up, LOSE down
    assert h1.dir_confusion["Short"] == {"up": 0, "down": 1}  # DOWN down
    assert h1.long_precision == pytest.approx(0.5)
    assert h1.short_precision == pytest.approx(1.0)
    # brier over WIN(conf .9, outcome 1), LOSE(conf .3, outcome 0), DOWN(conf .7, outcome 1)
    expected = ((0.9 - 1) ** 2 + (0.3 - 0) ** 2 + (0.7 - 1) ** 2) / 3
    assert h1.confidence_brier == pytest.approx(expected)


def test_concept_hits():
    h1 = evaluate_report(_report(), _price_df(), horizons=(1,)).horizons[0]
    # sector Long [WIN+6%, LOSE-2%] avg +2% > 0 -> hit; theme Short [DOWN] < 0 -> hit
    assert h1.sector_hit_rate == pytest.approx(1.0)
    assert h1.theme_hit_rate == pytest.approx(1.0)


def test_regime_veto_is_rule_not_overwrite():
    # Bearish report: raw Longs stay Long, but the veto flags them; we score how
    # the vetoed names actually moved (rule applied at eval time, no overwrite).
    rpt = _report()
    rpt.market_state = MarketRegime.BEARISH
    assert rpt.long_whitelist == ["WIN", "LOSE"]          # raw preserved
    assert rpt.tradable_long_whitelist == []              # all vetoed
    h1 = evaluate_report(rpt, _price_df(), horizons=(1,)).horizons[0]
    rv = h1.regime_veto
    assert rv["vetoed_long_count"] == 2                   # WIN, LOSE
    assert rv["evaluated"] == 2
    assert rv["vetoed_avg_return"] == pytest.approx((0.06 + -0.02) / 2)  # WIN +6%, LOSE -2%
    assert rv["vetoed_rose_rate"] == pytest.approx(0.5)


def test_not_elapsed_horizon_is_null():
    sc = evaluate_report(_report(), _price_df(), horizons=(1, 10))
    h10 = sc.horizons[1]
    assert h10.evaluable is False
    assert h10.target_date is None
    assert h10.market_return is None
    assert h10.market_hit is None
    assert sc.complete is False  # not every horizon elapsed


def test_bearish_state_hit_logic():
    rpt = _report()
    rpt.market_state = MarketRegime.BEARISH
    h1 = evaluate_report(rpt, _price_df(), horizons=(1,)).horizons[0]
    assert h1.market_hit is False  # proxy rose +2%, bearish call wrong


def test_missing_session_raises():
    rpt = _report()
    rpt.as_of_date = "2026-05-01"  # not in the price calendar
    with pytest.raises(ValueError, match="not in price data"):
        evaluate_report(rpt, _price_df())


def test_missing_proxy_raises():
    with pytest.raises(ValueError, match="proxy"):
        evaluate_report(_report(), _price_df(), proxy="QQQ")


def test_fixed_band_used_when_no_presession_history():
    # Synthetic data starts at the session => no pre-session bars => ATR falls back
    # to the fixed band, so the band actually applied is range_band.
    sc = evaluate_report(_report(), _price_df(), horizons=(1,), band_mode="atr", range_band=0.01)
    assert sc.atr_pct is None
    assert sc.horizons[0].range_band_used == pytest.approx(0.01)


def test_concept_hit_uses_path_trend_on_reversal():
    # A Long concept whose only member spikes day-1 then reverses: endpoint +2%
    # (would fake a Long "hit"), but the fitted drift is ~flat/negative -> miss
    # under the path-aware metric (same caliber as the market trend).
    cal = pd.bdate_range("2026-06-08", periods=5)
    closes = [110.0, 101.0, 102.0, 102.0, 102.0]
    rows = []
    for i, dt in enumerate(cal):
        op = 100.0 if i == 0 else closes[i]
        rows.append({"ticker": "SPY", "trade_date": dt, "open": 100.0 + i, "high": 0, "low": 0, "close": 100.0 + i})
        rows.append({"ticker": "ZZ", "trade_date": dt, "open": op, "high": 0, "low": 0, "close": closes[i]})
    df = pd.DataFrame(rows)
    rpt = RegimeReport(
        as_of_date="2026-06-08", market_state=MarketRegime.RANGE, macro_summary="x",
        concept_signals=[ConceptSignal(concept="Spk", level="sector", direction=Direction.LONG,
                                        confidence=0.6, member_tickers=["ZZ"])],
    )
    h_slope = evaluate_report(rpt, df, horizons=(3,), band_mode="fixed").horizons[0]
    assert h_slope.sector_hit_rate == pytest.approx(0.0)   # reversal -> not a real Long win
    h_ep = evaluate_report(rpt, df, horizons=(3,), band_mode="fixed", trend_metric="endpoint").horizons[0]
    assert h_ep.sector_hit_rate == pytest.approx(1.0)      # endpoint +2% fakes a hit


def test_slope_metric_beats_endpoint_on_reversal():
    # Day-1 spike to 110 then settles at 102: endpoint reads +2% (looks directional),
    # but the fitted drift over the path is ~flat -> correctly Range under "slope".
    cal = pd.bdate_range("2026-06-08", periods=5)
    closes = [110.0, 101.0, 102.0, 102.0, 102.0]
    rows = []
    for i, dt in enumerate(cal):
        op = 100.0 if i == 0 else closes[i]
        rows.append({"ticker": "SPY", "trade_date": dt, "open": op,
                     "high": closes[i], "low": closes[i], "close": closes[i]})
    df = pd.DataFrame(rows)
    rpt = RegimeReport(as_of_date="2026-06-08", market_state=MarketRegime.RANGE, macro_summary="x")

    sc = evaluate_report(rpt, df, horizons=(3,), band_mode="fixed", range_band=0.01)  # slope (default)
    h = sc.horizons[0]
    assert h.market_return == pytest.approx(0.02)        # endpoint: 102/100 - 1
    assert abs(h.market_trend) < 0.01                    # fitted drift ~flat
    assert h.market_hit is True                          # slope metric: correctly Range

    sc_ep = evaluate_report(rpt, df, horizons=(3,), band_mode="fixed", range_band=0.01, trend_metric="endpoint")
    assert sc_ep.horizons[0].market_hit is False         # endpoint metric misfires on the +2% close


def test_per_horizon_outlook_is_graded_separately():
    # SPY rises steadily (+2%/+4%/+6%). The report's near-term market_state is
    # Range, but the multi-horizon outlook calls 1d=Range, 3d/5d=Bullish. Each
    # horizon must be graded against ITS OWN call, not the near-term anchor.
    rpt = RegimeReport(
        as_of_date="2026-06-08", market_state=MarketRegime.RANGE, macro_summary="x",
        outlook=[
            HorizonOutlook(horizon="1d", direction=MarketRegime.RANGE, confidence=0.7),
            HorizonOutlook(horizon="3d", direction=MarketRegime.BULLISH, confidence=0.6),
            HorizonOutlook(horizon="5d", direction=MarketRegime.BULLISH, confidence=0.5),
        ],
    )
    sc = evaluate_report(rpt, _price_df(), horizons=(1, 3, 5), band_mode="fixed", range_band=0.01)
    h1, h3, h5 = sc.horizons
    assert (h1.graded_state, h1.from_outlook) == ("Range", True)
    assert h1.outlook_confidence == pytest.approx(0.7)
    assert h1.market_hit is False                 # +2% drift clears the 1% band, not Range
    assert (h3.graded_state, h3.market_hit) == ("Bullish", True)
    assert (h5.graded_state, h5.market_hit) == ("Bullish", True)


def test_missing_outlook_falls_back_to_market_state():
    # No outlook on the report => every horizon grades the near-term market_state.
    sc = evaluate_report(_report(), _price_df(), horizons=(1, 3), band_mode="fixed", range_band=0.01)
    for h in sc.horizons:
        assert h.from_outlook is False
        assert h.graded_state == "Bullish"        # _report() is Bullish
        assert h.outlook_confidence is None
        assert h.market_hit is True               # SPY rises -> Bullish hits


def test_atr_band_from_presession_bars():
    # 16 pre-session days of high=101/low=99/close=100 => Wilder ATR=2, ATR%=0.02.
    cal = pd.bdate_range("2026-05-01", periods=20)
    session = cal[16]
    rows = []
    for i, dt in enumerate(cal):
        op = 100.0 if i != 16 else 100.0
        cl = 100.0 if i != 16 else 100.5      # session h1 return = +0.5%
        rows.append({"ticker": "SPY", "trade_date": dt, "open": op,
                     "high": 101.0, "low": 99.0, "close": cl})
    price_df = pd.DataFrame(rows)
    rpt = RegimeReport(as_of_date=session.strftime("%Y-%m-%d"),
                       market_state=MarketRegime.RANGE, macro_summary="x")
    sc = evaluate_report(rpt, price_df, horizons=(1,), band_mode="atr", atr_k=1.0)
    assert sc.atr_pct == pytest.approx(0.02, abs=1e-6)
    h1 = sc.horizons[0]
    assert h1.range_band_used == pytest.approx(0.02)      # atr_k * 0.02 * sqrt(1)
    assert h1.market_return == pytest.approx(0.005)
    assert h1.market_hit is True                          # |0.5%| <= 2% => Range hit
