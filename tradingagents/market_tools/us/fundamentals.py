"""US fundamentals — point-in-time, leak-free, vendor-reliable.

Replaces the old yfinance ``.info`` snapshot (unofficial scraper, current-only,
ignored the session date -> leaked post-session info on a rollback). Instead we
assemble fundamentals from sources that respect the trading session:

  * Statement line items: FMP income / balance-sheet / cash-flow statements,
    kept only when their SEC ``acceptedDate`` is on/before the session's
    pre-market cutoff (so a rolled-back session sees only what was public then).
  * Price-type fields (price, 50/200-day MA, 52-week high/low): computed from
    our own BigQuery daily bars up to the prior session close.
  * Ratios (PE, P/B, margins, ROE/ROA, D/E, current ratio): computed here from
    the above, never read from a current-only vendor snapshot.

No yfinance/AV fallback: if FMP has no filing visible at the cutoff, we say so
and the caller simply omits fundamentals for that ticker.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from tradingagents.dataflows import fmp

from .prices import load_daily_ohlc, previous_trading_day

# Filings accepted before this wall-clock time on the session day are visible
# pre-open. 09:00 ET matches the news cutoff: conservative, before the bell.
_PREMARKET_CUTOFF = "09:00:00"


def _visible(reports: list[dict], cutoff: str) -> list[dict]:
    """Periods whose SEC acceptedDate <= ``cutoff`` ("YYYY-MM-DD HH:MM:SS"), newest first.

    A missing acceptedDate is treated as *not provably public* and dropped — we
    never include a filing we can't time-stamp against the session cutoff.
    """
    keep = [r for r in reports if (r.get("acceptedDate") or "") and str(r["acceptedDate"])[:19] <= cutoff]
    keep.sort(key=lambda r: str(r.get("date", "")), reverse=True)
    return keep


def _f(d: dict | None, *keys: str) -> float | None:
    """First present numeric field among ``keys`` (tolerates v3/stable name drift)."""
    if not d:
        return None
    for k in keys:
        v = d.get(k)
        if v is not None:
            try:
                return float(v)
            except (TypeError, ValueError):
                continue
    return None


def _ttm(reports: list[dict], *keys: str) -> tuple[float | None, int]:
    """Trailing-twelve-month sum over the latest 4 quarters.

    Returns ``(sum, n)``. If fewer than 4 quarters are visible, or any quarter
    is missing the field, the sum is None (we don't fabricate a partial TTM).
    """
    vals: list[float] = []
    for r in reports[:4]:
        v = _f(r, *keys)
        if v is None:
            return None, len(vals)
        vals.append(v)
    if len(vals) < 4:
        return None, len(vals)
    return sum(vals), len(vals)


def _ratio(a: float | None, b: float | None) -> float | None:
    if a is None or b is None or b == 0:
        return None
    return a / b


def _price_stats(ticker: str, session: str) -> dict | None:
    """Price-type fields from BigQuery daily bars up to the prior session close.

    Point-in-time by construction: nothing on/after ``session`` is read.
    """
    prev = previous_trading_day(session)
    start = (datetime.fromisoformat(prev) - timedelta(days=400)).strftime("%Y-%m-%d")
    df = load_daily_ohlc([ticker], start, prev)
    if df.empty:
        return None
    df = df.sort_values("trade_date")
    closes = df["close"].astype(float)
    win = df.tail(252)
    return {
        "as_of": prev,
        "price": float(closes.iloc[-1]),
        "ma50": float(closes.tail(50).mean()) if len(closes) >= 50 else None,
        "ma200": float(closes.tail(200).mean()) if len(closes) >= 200 else None,
        "hi52": float(win["high"].astype(float).max()),
        "lo52": float(win["low"].astype(float).min()),
    }


def _compute_metrics(
    income_v: list[dict],
    balance_v: list[dict],
    cashflow_v: list[dict],
    px: dict | None,
) -> dict:
    """Derive absolute figures + ratios from point-in-time statements and prices."""
    inc0 = income_v[0] if income_v else {}
    bal0 = balance_v[0] if balance_v else {}

    rev, nq = _ttm(income_v, "revenue")
    ni, _ = _ttm(income_v, "netIncome")
    gp, _ = _ttm(income_v, "grossProfit")
    oi, _ = _ttm(income_v, "operatingIncome")
    ebitda, _ = _ttm(income_v, "ebitda")
    eps, _ = _ttm(income_v, "epsDiluted", "epsdiluted", "eps")
    fcf, _ = _ttm(cashflow_v, "freeCashFlow")

    shares = _f(inc0, "weightedAverageShsOutDil", "weightedAverageShsOut")
    equity = _f(bal0, "totalStockholdersEquity")
    assets = _f(bal0, "totalAssets")
    debt = _f(bal0, "totalDebt")
    cur_a = _f(bal0, "totalCurrentAssets")
    cur_l = _f(bal0, "totalCurrentLiabilities")
    cash = _f(bal0, "cashAndCashEquivalents")
    price = px["price"] if px else None
    bvps = _ratio(equity, shares)

    m: dict = {
        "ttm_quarters": nq,
        "revenue_ttm": rev,
        "net_income_ttm": ni,
        "ebitda_ttm": ebitda,
        "fcf_ttm": fcf,
        "eps_ttm": eps,
        "shares_out": shares,
        "total_equity": equity,
        "total_debt": debt,
        "cash": cash,
        "book_value_per_share": bvps,
        "gross_margin": _ratio(gp, rev),
        "operating_margin": _ratio(oi, rev),
        "net_margin": _ratio(ni, rev),
        "roe": _ratio(ni, equity),
        "roa": _ratio(ni, assets),
        "debt_to_equity": _ratio(debt, equity),
        "current_ratio": _ratio(cur_a, cur_l),
    }
    if px:
        m["price"] = price
        m["ma50"] = px.get("ma50")
        m["ma200"] = px.get("ma200")
        m["hi52"] = px.get("hi52")
        m["lo52"] = px.get("lo52")
        m["market_cap"] = price * shares if shares else None
        # A trailing loss makes P/E meaningless; report n/a rather than a negative.
        m["pe_ttm"] = _ratio(price, eps) if (eps is not None and eps > 0) else None
        m["price_to_book"] = _ratio(price, bvps)
        m["pct_below_52w_high"] = _ratio(price - px["hi52"], px["hi52"]) if px.get("hi52") else None
    return m


def _fmt(v: float | None, pct: bool = False, money: bool = False) -> str:
    if v is None:
        return "n/a"
    if pct:
        return f"{v * 100:.2f}%"
    if money:
        return f"{v:,.0f}"
    return f"{v:,.2f}"


def _format_block(ticker: str, session: str, income_v: list[dict], m: dict, px: dict | None) -> str:
    inc0 = income_v[0]
    lines = [
        f"# Fundamentals for {ticker} (point-in-time, session {session})",
        f"# Source: FMP statements (acceptedDate <= {session} {_PREMARKET_CUTOFF} ET) + BigQuery prices",
        f"# Latest visible filing: period {inc0.get('date')} ({inc0.get('period')}), "
        f"accepted {inc0.get('acceptedDate')}; TTM over {m['ttm_quarters']} quarter(s)",
    ]
    if px:
        lines.append(f"# Price fields as of {px['as_of']} close (prior session)")
    lines.append("")
    rows = [
        ("Price", _fmt(m.get("price"))),
        ("Market Cap", _fmt(m.get("market_cap"), money=True)),
        ("PE (TTM)", _fmt(m.get("pe_ttm"))),
        ("EPS (TTM)", _fmt(m.get("eps_ttm"))),
        ("Price/Book", _fmt(m.get("price_to_book"))),
        ("Book Value/Share", _fmt(m.get("book_value_per_share"))),
        ("Revenue (TTM)", _fmt(m.get("revenue_ttm"), money=True)),
        ("Net Income (TTM)", _fmt(m.get("net_income_ttm"), money=True)),
        ("EBITDA (TTM)", _fmt(m.get("ebitda_ttm"), money=True)),
        ("Free Cash Flow (TTM)", _fmt(m.get("fcf_ttm"), money=True)),
        ("Gross Margin", _fmt(m.get("gross_margin"), pct=True)),
        ("Operating Margin", _fmt(m.get("operating_margin"), pct=True)),
        ("Net Margin", _fmt(m.get("net_margin"), pct=True)),
        ("ROE", _fmt(m.get("roe"), pct=True)),
        ("ROA", _fmt(m.get("roa"), pct=True)),
        ("Debt/Equity", _fmt(m.get("debt_to_equity"))),
        ("Current Ratio", _fmt(m.get("current_ratio"))),
        ("50 / 200-day MA", f"{_fmt(m.get('ma50'))} / {_fmt(m.get('ma200'))}"),
        ("52-week High / Low", f"{_fmt(m.get('hi52'))} / {_fmt(m.get('lo52'))}"),
    ]
    lines += [f"{label}: {val}" for label, val in rows]
    return "\n".join(lines)


def get_fundamentals(ticker: str, curr_date: str) -> str:
    """Point-in-time fundamentals block for ``ticker`` as of trading session ``curr_date``.

    ``curr_date`` is the session date (YYYY-MM-DD); it is the cutoff, not a hint.
    Statements are filtered to filings accepted before the session open and ratios
    are recomputed here, so this is safe to roll back to any past session.
    """
    if not curr_date:
        raise ValueError(
            "get_fundamentals requires curr_date (the trading session date) for point-in-time fundamentals"
        )
    cutoff = f"{curr_date[:10]} {_PREMARKET_CUTOFF}"
    stmts = fmp.get_financial_statements(ticker)
    income_v = _visible(stmts["income"], cutoff)
    balance_v = _visible(stmts["balance"], cutoff)
    cashflow_v = _visible(stmts["cashflow"], cutoff)
    if not income_v:
        return (
            f"No point-in-time fundamentals for {ticker} as of {curr_date}: "
            f"no FMP filing accepted on/before {cutoff}."
        )
    px = _price_stats(ticker, curr_date)
    m = _compute_metrics(income_v, balance_v, cashflow_v, px)
    return _format_block(ticker, curr_date, income_v, m, px)
