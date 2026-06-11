"""Module A: post-hoc accuracy evaluation of a RegimeReport (zero LLM cost).

Given a persisted ``RegimeReport`` (the pre-market judgment for a session) and a
long OHLC table covering the session + forward window, score the judgment
against realized forward returns. No LLM, no look-ahead into the report — we
only read prices on/after the session open (the judgment was made pre-open, so
the session open is the first fill the quant layer can act on).

Forward returns are anchored at the **session open** and measure holding N
trading days, *counting the session itself as day 1*: horizon N exits at the
close of calendar index ``i0 + (N-1)``. So for a 2026-06-09 judgment, h1 =
open(06-09)→close(06-09) (the session's own reaction), h3 → close(06-11), h5 →
close(06-15). The market proxy (SPY/QQQ) supplies both the de-marketing baseline
and the trading calendar (it prints every session). Horizons that haven't
elapsed yet score as ``None`` — expected, not an error; rerun once prices exist.

The Bullish/Range/Bearish call is graded against a **volatility-adaptive band**:
instead of a hardcoded ±1%, the flat ("Range") zone is ``atr_k × ATR% × √N`` of
the proxy, where ATR% is the proxy's daily ATR (Wilder) computed *strictly from
pre-session bars* (no look-ahead) as a fraction of price, and √N scales it to an
N-day horizon. A ±1% day means something very different at VIX 13 vs VIX 25, so
the threshold tracks the regime's own volatility. Falls back to a fixed band if
pre-session history is too short.

The Bullish/Range/Bearish call is graded **path-aware**, not on a single endpoint.
An endpoint return (open→exit close) misjudges trend: an up-then-down reversal
ending flat would read as Range only by luck, and a down-all-week-then-bounce
would read as up. So ``market_state`` is scored against the **OLS slope of log
price over the held window** (the fitted drift), with R² reporting how trend-like
vs choppy the path was. A reversal collapses the slope toward zero (correctly
Range); a steady move keeps it (Bullish/Bearish). The raw endpoint return is still
reported (it's the buy-and-hold P&L) and drives the whitelist economics.

The regime "circuit breaker" is evaluated as a **rule, not an overwrite**: raw
L1 Longs stay Long in the report; we ask ``report.regime_blocked_longs`` which
Longs the regime vetoes and check whether those vetoed names actually fell.

**Multi-horizon (module B):** when the report carries a per-horizon ``outlook``
(a separate Bullish/Range/Bearish call for 1d/3d/5d), each horizon is graded
against *its own* call (``graded_state``, ``from_outlook=True``); horizons with no
outlook fall back to the near-term ``market_state``. This lets calibration be
tracked per holding period instead of grading one near-term call across windows.

See docs/regime-gate-feedback-design.md §2 (Module A) and §3 (Module B).
"""

from __future__ import annotations

import math
from datetime import date

import numpy as np
import pandas as pd
from pydantic import BaseModel

from .schemas import Direction, MarketRegime, RegimeReport

DEFAULT_HORIZONS = (1, 3, 5)
DEFAULT_PROXY = "SPY"
DEFAULT_RANGE_BAND = 0.01  # fixed fallback when ATR history is unavailable
DEFAULT_ATR_WINDOW = 14
DEFAULT_ATR_K = 1.0        # flat zone = atr_k * dayATR% * sqrt(horizon)
DEFAULT_TREND_METRIC = "slope"  # "slope" (path-aware) | "endpoint"


class WhitelistScore(BaseModel):
    count: int               # signals in this bucket
    evaluated: int           # of those, how many had forward prices
    hit_rate: float | None   # fraction realized in the bet's direction
    avg_return: float | None
    avg_alpha: float | None  # vs proxy (short flips sign)
    win_rate: float | None   # fraction beating the proxy (short: underperforming)


class HorizonScore(BaseModel):
    horizon: int                       # trading days held (session counts as day 1)
    target_date: str | None            # exit close date; None if not elapsed yet
    evaluable: bool
    graded_state: str                  # regime graded here: this horizon's outlook, else market_state
    from_outlook: bool                 # True if a per-horizon outlook (B) supplied graded_state
    outlook_confidence: float | None   # the outlook's stated confidence (for calibration), if any
    market_return: float | None        # endpoint: proxy close(exit)/open(session)-1 (buy-hold P&L)
    market_trend: float | None         # path-aware: fitted drift of proxy log-price over the window
    market_trend_r2: float | None      # 0-1; how trend-like (high) vs choppy (low) the path was
    market_hit: bool | None            # did graded_state match the graded trend (see trend_metric)
    range_band_used: float             # flat-zone half-width applied at this horizon
    long: WhitelistScore
    short: WhitelistScore
    dir_confusion: dict[str, dict[str, int]]  # {"Long": {"up":n,"down":n}, ...}
    long_precision: float | None
    short_precision: float | None
    confidence_brier: float | None     # over directional signals
    regime_veto: dict                  # effectiveness of the regime Long-veto (rule, not overwrite)
    sector_hit_rate: float | None
    theme_hit_rate: float | None


class Scorecard(BaseModel):
    session: str
    market_state: str
    evaluated_at: str
    proxy: str
    band_mode: str                     # "atr" | "fixed"
    atr_pct: float | None              # proxy daily ATR% used for the band (pre-session)
    atr_k: float
    range_band: float                  # fixed fallback / used when band_mode="fixed"
    trend_metric: str                  # "slope" (path-aware) | "endpoint"
    complete: bool                     # every horizon was evaluable
    horizons: list[HorizonScore]


def _safe_mean(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def _wide(price_df: pd.DataFrame, field: str) -> pd.DataFrame:
    """Pivot the long OHLC table to date x ticker for ``field``."""
    return price_df.pivot_table(index="trade_date", columns="ticker", values=field).sort_index()


def _atr_pct(price_df: pd.DataFrame, ticker: str, session_ts: pd.Timestamp, window: int) -> float | None:
    """Wilder ATR% of ``ticker`` from bars STRICTLY before the session (no look-ahead).

    ATR% = ATR(window) / last pre-session close. Returns None if there aren't
    enough pre-session bars to seed a ``window``-period ATR.
    """
    s = (price_df[price_df["ticker"] == ticker]
         .loc[lambda d: d["trade_date"] < session_ts]
         .sort_values("trade_date"))
    if len(s) < window + 1:
        return None
    high, low, close = s["high"].to_numpy(), s["low"].to_numpy(), s["close"].to_numpy()
    prev_close = close[:-1]
    tr = pd.Series(
        [max(high[i] - low[i], abs(high[i] - prev_close[i - 1]), abs(low[i] - prev_close[i - 1]))
         for i in range(1, len(close))]
    )
    atr = tr.ewm(alpha=1.0 / window, adjust=False).mean().iloc[-1]  # Wilder smoothing
    last_close = close[-1]
    if last_close == 0:
        return None
    return float(atr / last_close)


def _horizon_band(n: int, atr_pct: float | None, atr_k: float, range_band: float) -> float:
    """Flat-zone half-width for horizon ``n``: atr_k*ATR%*sqrt(n), or fixed fallback."""
    if atr_pct is None:
        return range_band
    return atr_k * atr_pct * math.sqrt(n)


def _exit_index(i0: int, n: int) -> int:
    """Exit calendar index for holding ``n`` trading days, session = day 1."""
    return i0 + (n - 1)


def _forward_return(opens, closes, calendar: list, i0: int, n: int, ticker: str) -> float | None:
    """Hold ``n`` trading days from the session open: close[i0+n-1]/open[i0]-1.

    Returns None if the horizon hasn't elapsed or the ticker lacks prices.
    """
    exit_i = _exit_index(i0, n)
    if exit_i >= len(calendar) or ticker not in opens.columns or ticker not in closes.columns:
        return None
    base = opens.at[calendar[i0], ticker]
    tgt = closes.at[calendar[exit_i], ticker]
    if pd.isna(base) or pd.isna(tgt) or base == 0:
        return None
    return float(tgt / base - 1.0)


def _price_path(opens, closes, calendar: list, i0: int, n: int, ticker: str) -> list[float] | None:
    """Held-window price path ``[open(D), close(D), …, close(exit)]`` (length n+1).

    Anchored at the session open (the fill) and walking the daily closes through
    the exit; this is what the path-aware trend fit consumes. None on missing data.
    """
    exit_i = _exit_index(i0, n)
    if exit_i >= len(calendar) or ticker not in opens.columns or ticker not in closes.columns:
        return None
    o = opens.at[calendar[i0], ticker]
    if pd.isna(o) or o <= 0:
        return None
    path = [float(o)]
    for k in range(i0, exit_i + 1):
        c = closes.at[calendar[k], ticker]
        if pd.isna(c) or c <= 0:
            return None
        path.append(float(c))
    return path


def _trend(prices: list[float]) -> tuple[float | None, float | None]:
    """OLS of log price vs time → (fitted drift over the window, R²).

    The drift is the fitted simple return from t=0 to t=last, so it's directly
    comparable to the endpoint return / ATR band, but robust: an up-then-down
    reversal flattens the slope toward 0, a steady move preserves it. R² (on log
    prices) tells trend (→1) from chop (→0). None if fewer than 2 points.
    """
    if not prices or len(prices) < 2:
        return None, None
    logp = np.log(np.asarray(prices, dtype=float))
    t = np.arange(len(logp), dtype=float)
    slope, intercept = np.polyfit(t, logp, 1)
    fit = slope * t + intercept
    ss_res = float(((logp - fit) ** 2).sum())
    ss_tot = float(((logp - logp.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    drift = float(np.expm1(slope * (len(prices) - 1)))
    return drift, r2


def _grade_regime(state: MarketRegime, graded: float | None, band: float) -> bool | None:
    """Did the regime call match the realized (path-aware or endpoint) move?

    Bullish hits when the move clears +band, Bearish when it clears -band, Range
    when it stays inside the flat band. None if the move is unavailable.
    """
    if graded is None:
        return None
    if state is MarketRegime.BULLISH:
        return graded > band
    if state is MarketRegime.BEARISH:
        return graded < -band
    return abs(graded) <= band


def _whitelist_score(rets: list[float], mkt: float | None, *, short: bool) -> WhitelistScore:
    """Aggregate forward returns for one directional bucket.

    For shorts the bet wins when the stock *falls* / underperforms, so hit and
    alpha flip sign relative to longs.
    """
    if not rets:
        return WhitelistScore(count=0, evaluated=0, hit_rate=None, avg_return=None,
                              avg_alpha=None, win_rate=None)
    if short:
        hit = _safe_mean([1.0 if r < 0 else 0.0 for r in rets])
        alpha = _safe_mean([mkt - r for r in rets]) if mkt is not None else None
        win = _safe_mean([1.0 if r < mkt else 0.0 for r in rets]) if mkt is not None else None
    else:
        hit = _safe_mean([1.0 if r > 0 else 0.0 for r in rets])
        alpha = _safe_mean([r - mkt for r in rets]) if mkt is not None else None
        win = _safe_mean([1.0 if r > mkt else 0.0 for r in rets]) if mkt is not None else None
    return WhitelistScore(count=len(rets), evaluated=len(rets), hit_rate=hit,
                          avg_return=_safe_mean(rets), avg_alpha=alpha, win_rate=win)


def _score_horizon(
    report: RegimeReport,
    opens: pd.DataFrame,
    closes: pd.DataFrame,
    calendar: list,
    i0: int,
    n: int,
    proxy: str,
    band: float,
    trend_metric: str,
) -> HorizonScore:
    mkt = _forward_return(opens, closes, calendar, i0, n, proxy)
    market_trend, market_trend_r2 = _trend(_price_path(opens, closes, calendar, i0, n, proxy) or [])
    exit_i = _exit_index(i0, n)
    evaluable = exit_i < len(calendar)
    target_date = calendar[exit_i].strftime("%Y-%m-%d") if evaluable else None

    # B (multi-horizon): grade this horizon against its own outlook call if the
    # commander emitted one; otherwise fall back to the near-term market_state.
    outlook = report.outlook_for(n)
    graded_state = outlook.direction if outlook else report.market_state

    # Grade against the path-aware fitted drift (default), falling back to the
    # endpoint return if the slope is unavailable (e.g. <2 points).
    graded = market_trend if (trend_metric == "slope" and market_trend is not None) else mkt
    market_hit = _grade_regime(graded_state, graded, band)

    # per-stock returns + confusion matrix (raw L1 directions, un-gated)
    confusion = {d.value: {"up": 0, "down": 0} for d in Direction}
    long_rets: list[float] = []
    short_rets: list[float] = []
    brier_terms: list[float] = []

    for sig in report.stock_signals:
        r = _forward_return(opens, closes, calendar, i0, n, sig.ticker)
        if r is None:
            continue
        bucket = "up" if r >= 0 else "down"
        confusion[sig.direction.value][bucket] += 1
        if sig.direction is Direction.LONG:
            long_rets.append(r)
            brier_terms.append((sig.catalyst_confidence - (1.0 if r > 0 else 0.0)) ** 2)
        elif sig.direction is Direction.SHORT:
            short_rets.append(r)
            brier_terms.append((sig.catalyst_confidence - (1.0 if r < 0 else 0.0)) ** 2)

    long_n = confusion["Long"]["up"] + confusion["Long"]["down"]
    short_n = confusion["Short"]["up"] + confusion["Short"]["down"]
    long_precision = confusion["Long"]["up"] / long_n if long_n else None
    short_precision = confusion["Short"]["down"] / short_n if short_n else None

    # Regime veto effectiveness (rule, not overwrite): of the raw Longs the regime
    # vetoes, how did they actually do? Negative avg => the veto avoided losses.
    vetoed = report.regime_blocked_longs
    veto_rets = [r for t in vetoed if (r := _forward_return(opens, closes, calendar, i0, n, t)) is not None]
    regime_veto = {
        "vetoed_long_count": len(vetoed),
        "evaluated": len(veto_rets),
        "vetoed_avg_return": _safe_mean(veto_rets),
        "vetoed_rose_rate": _safe_mean([1.0 if r > 0 else 0.0 for r in veto_rets]),
    }

    sector_hits, theme_hits = _concept_hits(report, opens, closes, calendar, i0, n, trend_metric)

    return HorizonScore(
        horizon=n,
        target_date=target_date,
        evaluable=evaluable,
        graded_state=graded_state.value,
        from_outlook=outlook is not None,
        outlook_confidence=outlook.confidence if outlook else None,
        market_return=mkt,
        market_trend=market_trend,
        market_trend_r2=market_trend_r2,
        market_hit=market_hit,
        range_band_used=band,
        long=_whitelist_score(long_rets, mkt, short=False),
        short=_whitelist_score(short_rets, mkt, short=True),
        dir_confusion=confusion,
        long_precision=long_precision,
        short_precision=short_precision,
        confidence_brier=_safe_mean(brier_terms),
        regime_veto=regime_veto,
        sector_hit_rate=sector_hits,
        theme_hit_rate=theme_hits,
    )


def _member_move(opens, closes, calendar, i0, n, ticker, trend_metric) -> float | None:
    """One member's directional move over the window — path-aware drift (default)
    or endpoint return, matching how the market trend is graded."""
    if trend_metric == "slope":
        drift, _ = _trend(_price_path(opens, closes, calendar, i0, n, ticker) or [])
        return drift
    return _forward_return(opens, closes, calendar, i0, n, ticker)


def _concept_hits(report, opens, closes, calendar, i0, n, trend_metric) -> tuple[float | None, float | None]:
    """Per-level hit rate: a directional concept hits if its members' mean move
    agrees with the lean (Long>0 / Short<0). Uses the same path-aware trend as the
    market call (``trend_metric``), so a member that spikes then reverses doesn't
    fake a directional win. Block concepts are skipped."""
    sector, theme = [], []
    for c in report.concept_signals:
        if c.direction is Direction.BLOCK or not c.member_tickers:
            continue
        moves = [m for t in c.member_tickers
                 if (m := _member_move(opens, closes, calendar, i0, n, t, trend_metric)) is not None]
        if not moves:
            continue
        avg = sum(moves) / len(moves)
        hit = 1.0 if ((c.direction is Direction.LONG and avg > 0) or
                      (c.direction is Direction.SHORT and avg < 0)) else 0.0
        (sector if c.level == "sector" else theme).append(hit)
    return _safe_mean(sector), _safe_mean(theme)


def evaluate_report(
    report: RegimeReport,
    price_df: pd.DataFrame,
    *,
    proxy: str = DEFAULT_PROXY,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    band_mode: str = "atr",
    atr_window: int = DEFAULT_ATR_WINDOW,
    atr_k: float = DEFAULT_ATR_K,
    range_band: float = DEFAULT_RANGE_BAND,
    trend_metric: str = DEFAULT_TREND_METRIC,
    evaluated_at: str | None = None,
) -> Scorecard:
    """Score ``report`` against realized forward returns in ``price_df``.

    ``price_df`` is the long ``[ticker, trade_date, open, high, low, close]`` table
    and MUST include ``proxy`` (defines the trading calendar + de-marketing baseline)
    and cover from before the session (for ATR) through the longest horizon. The
    session date must be present (it's the open-price baseline) — otherwise we
    raise. Returns are anchored at the session open, session = horizon day 1.

    ``band_mode="atr"`` grades the market call against ``atr_k × proxyATR% × √N``
    (ATR from pre-session bars, no look-ahead); ``"fixed"`` uses ``range_band``.
    Falls back to the fixed band per-horizon if ATR history is too short.
    """
    if price_df.empty:
        raise ValueError(
            f"price data empty for session {report.as_of_date}: the daily table has no rows in the "
            f"requested window — the session/forward bars likely haven't been ingested yet. "
            f"Check the latest available trading day and rerun once it covers the session."
        )
    opens = _wide(price_df, "open")
    closes = _wide(price_df, "close")
    calendar = list(closes.index)
    session_ts = pd.Timestamp(report.as_of_date)
    if session_ts not in closes.index:
        raise ValueError(
            f"session {report.as_of_date} not in price data (no baseline); "
            f"calendar present spans {calendar[0].date()}..{calendar[-1].date()}. "
            f"Either the session bar isn't ingested yet, or {report.as_of_date} wasn't a trading day."
        )
    if proxy not in closes.columns or proxy not in opens.columns:
        raise ValueError(f"proxy {proxy!r} missing from price_df; needed for market truth + calendar")

    i0 = calendar.index(session_ts)
    atr_pct = _atr_pct(price_df, proxy, session_ts, atr_window) if band_mode == "atr" else None
    horizon_scores = [
        _score_horizon(report, opens, closes, calendar, i0, n, proxy,
                       _horizon_band(n, atr_pct, atr_k, range_band), trend_metric)
        for n in horizons
    ]
    return Scorecard(
        session=report.as_of_date,
        market_state=report.market_state.value,
        evaluated_at=evaluated_at or date.today().strftime("%Y-%m-%d"),
        proxy=proxy,
        band_mode=band_mode,
        atr_pct=atr_pct,
        atr_k=atr_k,
        range_band=range_band,
        trend_metric=trend_metric,
        complete=all(h.evaluable for h in horizon_scores),
        horizons=horizon_scores,
    )
