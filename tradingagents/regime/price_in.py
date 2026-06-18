"""price-in labeling: did the market already absorb the news?

For each ``NewsEvent`` with a publication timestamp, run a small daily event
study against real prices and decide whether the information was already priced
in, partially priced in, fresh, or just a post-hoc recap of a move that had
already happened.

Daily-bar limitation (honest by design): we cannot see the intraday tick when
news hit, so the split between "before" and "after" is at the *reaction
session* — the first session that could trade on the news (next session if the
article published after the US close). Moves are measured in ATR units so the
threshold adapts to each name's volatility; all parameters are returned/recorded
so a label is reproducible.

``label_price_in`` is a pure function over a single ticker's OHLCV frame
(injectable, no I/O) so it is unit-testable without BigQuery. ``tag_price_in``
is the batch enricher that loads prices via MarketDataTools and mutates events.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd

from tradingagents.market_tools import MarketDataTools, get_market_tools

from .events import NewsEvent, PriceInStatus

# US equity close is 16:00 ET ~ 20:00-21:00 UTC depending on DST. News with a
# UTC hour at/after this is treated as after-hours: its first tradable session
# is the next one. A conservative single cutoff avoids a tz database lookup.
_AFTER_HOURS_UTC_HOUR = 20


def _reaction_session(pub_dt: datetime, sessions: list[pd.Timestamp]) -> pd.Timestamp | None:
    """First session that could trade on news published at ``pub_dt``.

    Intraday news reacts the same session; after-hours (>= ~US close) reacts
    the next session.
    """
    pub_date = pd.Timestamp(pub_dt.date())
    after_hours = pub_dt.hour >= _AFTER_HOURS_UTC_HOUR
    for s in sessions:
        if after_hours and s <= pub_date:
            continue
        if not after_hours and s < pub_date:
            continue
        return s
    return None


def _atr_pct(ohlcv: pd.DataFrame, end_idx: int, window: int) -> float | None:
    """Average true range as a fraction of close, over ``window`` sessions
    ending at ``end_idx`` (inclusive). None if insufficient history."""
    if end_idx < window:
        return None
    seg = ohlcv.iloc[end_idx - window + 1 : end_idx + 1]
    prev_close = ohlcv["close"].shift(1).iloc[end_idx - window + 1 : end_idx + 1]
    tr = pd.concat(
        [
            seg["high"] - seg["low"],
            (seg["high"] - prev_close).abs(),
            (seg["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.mean()
    last_close = ohlcv["close"].iloc[end_idx]
    if last_close <= 0 or pd.isna(atr):
        return None
    return float(atr / last_close)


def label_price_in(
    event: NewsEvent,
    ohlcv: pd.DataFrame,
    *,
    pre_days: int = 3,
    post_days: int = 2,
    atr_window: int = 14,
    sig_atr: float = 1.0,
) -> dict:
    """Event study for one event. Returns a dict of metrics + ``price_in``.

    ``ohlcv`` is a single ticker's daily bars with a ``trade_date`` column (or
    index) and open/high/low/close/volume. Returns keys: ``price_in``,
    ``pre_return``, ``post_return``, ``pre_volume_ratio``, ``post_volume_ratio``.
    Insufficient data yields ``PriceInStatus.UNKNOWN`` with None metrics.
    """
    unknown = {
        "price_in": PriceInStatus.UNKNOWN,
        "pre_return": None,
        "post_return": None,
        "pre_volume_ratio": None,
        "post_volume_ratio": None,
    }
    if not event.published_utc or ohlcv is None or ohlcv.empty:
        return unknown

    df = ohlcv.copy()
    if "trade_date" in df.columns:
        df = df.sort_values("trade_date").reset_index(drop=True)
        sessions = list(df["trade_date"])
    else:
        df = df.sort_index()
        sessions = list(df.index)

    pub_dt = pd.to_datetime(event.published_utc).to_pydatetime()
    reaction = _reaction_session(pub_dt, sessions)
    if reaction is None:
        return unknown
    r = sessions.index(reaction)

    # Need pre_days history before the reaction and post_days after it.
    if r - pre_days < 0 or r + post_days >= len(df):
        return unknown

    atr_pct = _atr_pct(df, r - 1, atr_window)
    if atr_pct is None or atr_pct <= 0:
        return unknown

    close = df["close"]
    open_ = df["open"]
    vol = df["volume"]

    # Move leading INTO the event (close before reaction vs pre_days earlier).
    pre_return = float(close.iloc[r - 1] / close.iloc[r - 1 - pre_days] - 1.0)
    # Reaction: enter at reaction-session open, exit post_days later at close.
    post_return = float(close.iloc[r + post_days] / open_.iloc[r] - 1.0)

    baseline_vol = float(vol.iloc[max(0, r - atr_window) : r].mean())
    pre_volume_ratio = (
        float(vol.iloc[r - pre_days : r].mean() / baseline_vol) if baseline_vol > 0 else None
    )
    post_volume_ratio = (
        float(vol.iloc[r : r + 1].mean() / baseline_vol) if baseline_vol > 0 else None
    )

    pre_sig = abs(pre_return) / atr_pct >= sig_atr
    post_sig = abs(post_return) / atr_pct >= sig_atr

    if post_sig and not pre_sig:
        status = PriceInStatus.NOT_PRICED_IN
    elif post_sig and pre_sig:
        status = PriceInStatus.PARTIAL
    elif pre_sig and not post_sig:
        # Move happened before the news could be traded. After-hours publication
        # of an already-moved name with no follow-through reads as a recap.
        status = PriceInStatus.POST_HOC if pub_dt.hour >= _AFTER_HOURS_UTC_HOUR else PriceInStatus.PRICED_IN
    else:
        # Neither side moved beyond noise: nothing left to capture.
        status = PriceInStatus.PRICED_IN

    return {
        "price_in": status,
        "pre_return": round(pre_return, 6),
        "post_return": round(post_return, 6),
        "pre_volume_ratio": round(pre_volume_ratio, 4) if pre_volume_ratio is not None else None,
        "post_volume_ratio": round(post_volume_ratio, 4) if post_volume_ratio is not None else None,
    }


def tag_price_in(
    events: list[NewsEvent],
    *,
    market: str = "US",
    tools: MarketDataTools | None = None,
    pre_days: int = 3,
    post_days: int = 2,
    atr_window: int = 14,
    sig_atr: float = 1.0,
) -> list[NewsEvent]:
    """Label every event in place with its price-in status + metrics.

    Loads one daily OHLCV window per ticker (covering ATR history + the
    post-event horizon) and applies ``label_price_in``. ``tools`` is injectable
    for tests. Events without a usable timestamp/price stay Unknown.
    """
    if not events:
        return events
    tools = tools or get_market_tools(market)

    by_ticker: dict[str, list[NewsEvent]] = {}
    for ev in events:
        by_ticker.setdefault(ev.ticker, []).append(ev)

    for ticker, evs in by_ticker.items():
        pubs = [pd.to_datetime(e.published_utc) for e in evs if e.published_utc]
        if not pubs:
            continue
        # Window: ATR history + pre buffer before the earliest event, post
        # buffer (calendar-padded for weekends/holidays) after the latest.
        start = (min(pubs) - timedelta(days=atr_window + pre_days + 7)).strftime("%Y-%m-%d")
        end = (max(pubs) + timedelta(days=post_days + 7)).strftime("%Y-%m-%d")
        ohlcv = tools.load_daily_ohlcv([ticker], start, end)
        if ohlcv is None or ohlcv.empty:
            continue
        ohlcv = ohlcv[ohlcv["ticker"] == ticker] if "ticker" in ohlcv.columns else ohlcv
        for ev in evs:
            metrics = label_price_in(
                ev, ohlcv, pre_days=pre_days, post_days=post_days, atr_window=atr_window, sig_atr=sig_atr
            )
            ev.price_in = metrics["price_in"]
            ev.pre_return = metrics["pre_return"]
            ev.post_return = metrics["post_return"]
            ev.pre_volume_ratio = metrics["pre_volume_ratio"]
            ev.post_volume_ratio = metrics["post_volume_ratio"]
    return events
