"""Line-1 structured catalysts: deterministic events straight from FMP feeds.

Unlike the news track (LLM reads vendor summaries -> NewsEvent), these come from
*structured* endpoints — earnings, analyst grade actions, price-target changes,
dividends, M&A — so there is no LLM, no hallucination, and no opinion: the
``event_type``/``polarity`` are derived by rule and ``certainty`` is always
Confirmed. Full numeric payloads (eps surprise, PT change, dividend amount, …)
are preserved so the NN can use magnitude, not just a category.

Output is one ``Catalyst`` per (ticker, event) with an ``effective_date`` and,
where the feed provides a timestamp, a precise ``published_utc``. Point-in-time
alignment to trading sessions is the downstream feature-join's job; here we just
record the event honestly. The corpus is written to ``catalysts.jsonl`` and
joins the news ``events.jsonl`` on (ticker, session) at feature time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from tradingagents.dataflows import fmp

from .events import Certainty, Polarity


class CatalystType(str, Enum):
    EARNINGS = "Earnings"
    ANALYST_GRADE = "AnalystGrade"
    PRICE_TARGET = "PriceTarget"
    DIVIDEND = "Dividend"
    MNA = "MnA"


@dataclass
class Catalyst:
    ticker: str
    catalyst_type: CatalystType
    effective_date: str  # event's own date (YYYY-MM-DD)
    polarity: Polarity
    certainty: Certainty
    summary: str
    source: str
    published_utc: str | None = None  # precise instant when the feed gives one
    data: dict = field(default_factory=dict)  # type-specific numeric payload

    def to_dict(self) -> dict:
        base = {
            "ticker": self.ticker,
            "catalyst_type": self.catalyst_type.value,
            "effective_date": self.effective_date,
            "polarity": self.polarity.value,
            "certainty": self.certainty.value,
            "summary": self.summary,
            "source": self.source,
            "published_utc": self.published_utc,
        }
        base.update(self.data)  # flatten numerics to top level for easy analysis
        return base


def _in_range(d: str, start: str, end: str) -> bool:
    return bool(d) and start <= d[:10] <= end


# --- earnings ----------------------------------------------------------------

def _eps_polarity(actual: float | None, est: float | None) -> Polarity:
    """Beat/miss with a ~1% tolerance band around the estimate."""
    if actual is None or est is None:
        return Polarity.NEUTRAL
    diff = actual - est
    tol = 0.01 * abs(est)
    if diff > tol:
        return Polarity.POSITIVE
    if diff < -tol:
        return Polarity.NEGATIVE
    return Polarity.NEUTRAL


def earnings_to_catalysts(symbol: str, rows: list[dict], start: str, end: str) -> list[Catalyst]:
    out: list[Catalyst] = []
    for r in rows:
        actual = r.get("epsActual")
        if actual is None:  # future/unreported quarter — not yet an event
            continue
        d = r.get("date", "")
        if not _in_range(d, start, end):
            continue
        est = r.get("epsEstimated")
        rev_a, rev_e = r.get("revenueActual"), r.get("revenueEstimated")
        surprise = (actual - est) if est is not None else None
        surprise_pct = (surprise / abs(est)) if (surprise is not None and est) else None
        rev_pct = ((rev_a - rev_e) / abs(rev_e)) if (rev_a is not None and rev_e) else None
        summary = f"{symbol} reported EPS {actual} vs est {est}"
        if rev_a is not None:
            summary += f"; revenue {rev_a} vs est {rev_e}"
        out.append(Catalyst(
            symbol, CatalystType.EARNINGS, d[:10], _eps_polarity(actual, est),
            Certainty.CONFIRMED, summary, "FMP:earnings", None,
            {
                "eps_actual": actual, "eps_estimated": est,
                "eps_surprise": round(surprise, 4) if surprise is not None else None,
                "eps_surprise_pct": round(surprise_pct, 4) if surprise_pct is not None else None,
                "revenue_actual": rev_a, "revenue_estimated": rev_e,
                "revenue_surprise_pct": round(rev_pct, 4) if rev_pct is not None else None,
            },
        ))
    return out


# --- analyst grade actions ---------------------------------------------------

def grades_to_catalysts(symbol: str, rows: list[dict], start: str, end: str) -> list[Catalyst]:
    """Only upgrades/downgrades — "maintain" reiterations dominate (~88%) and
    carry little catalyst signal, so they are dropped."""
    out: list[Catalyst] = []
    for r in rows:
        action = (r.get("action") or "").lower()
        if action not in ("upgrade", "downgrade"):
            continue
        d = r.get("date", "")
        if not _in_range(d, start, end):
            continue
        pol = Polarity.POSITIVE if action == "upgrade" else Polarity.NEGATIVE
        gc, pg, ng = r.get("gradingCompany", "?"), r.get("previousGrade"), r.get("newGrade")
        out.append(Catalyst(
            symbol, CatalystType.ANALYST_GRADE, d[:10], pol, Certainty.CONFIRMED,
            f"{gc} {action}d {symbol} from {pg} to {ng}", "FMP:grades", None,
            {"grading_company": gc, "previous_grade": pg, "new_grade": ng, "action": action},
        ))
    return out


# --- price-target changes ----------------------------------------------------

_PT_UP = ("raised", "raises", "increased", "increases", "boost", "hiked", "lifted")
_PT_DOWN = ("lowered", "lowers", " cut", "cuts", "decreased", "slashed", "reduced", "trimmed")


def _pt_polarity(title: str) -> Polarity:
    t = (title or "").lower()
    if any(k in t for k in _PT_DOWN):
        return Polarity.NEGATIVE
    if any(k in t for k in _PT_UP):
        return Polarity.POSITIVE
    return Polarity.NEUTRAL  # reiterations / no direction in the headline


def price_target_to_catalysts(symbol: str, rows: list[dict], start: str, end: str) -> list[Catalyst]:
    out: list[Catalyst] = []
    for r in rows:
        pub = r.get("publishedDate", "")  # ISO-8601 with 'Z' (UTC)
        if not _in_range(pub, start, end):
            continue
        pt, pwp = r.get("priceTarget"), r.get("priceWhenPosted")
        upside = (pt / pwp - 1) if (pt and pwp) else None
        title = r.get("newsTitle", "")
        out.append(Catalyst(
            symbol, CatalystType.PRICE_TARGET, pub[:10], _pt_polarity(title),
            Certainty.CONFIRMED, title or f"{symbol} price target {pt}", "FMP:price-target", pub,
            {
                "analyst_company": r.get("analystCompany"), "price_target": pt,
                "price_when_posted": pwp,
                "implied_upside": round(upside, 4) if upside is not None else None,
            },
        ))
    return out


# --- dividends ---------------------------------------------------------------

def dividends_to_catalysts(symbol: str, rows: list[dict], start: str, end: str) -> list[Catalyst]:
    """Anchored on ``declarationDate`` (when it becomes public). Polarity is the
    change vs the previous declared amount (raised/cut/flat), diffed over the
    full history so an in-range row gets the correct change across the boundary."""
    hist = sorted((r for r in rows if r.get("declarationDate")), key=lambda r: r["declarationDate"])
    out: list[Catalyst] = []
    prev: float | None = None
    for r in hist:
        decl, amt = r["declarationDate"], r.get("dividend")
        change = None
        pol = Polarity.NEUTRAL
        if prev is not None and amt is not None:
            if amt > prev:
                pol, change = Polarity.POSITIVE, "raised"
            elif amt < prev:
                pol, change = Polarity.NEGATIVE, "cut"
            else:
                change = "flat"
        if amt is not None:
            prev = amt
        if not _in_range(decl, start, end):
            continue
        summary = f"{symbol} declared {r.get('frequency', '')} dividend {amt}".strip()
        if change:
            summary += f" ({change})"
        out.append(Catalyst(
            symbol, CatalystType.DIVIDEND, decl[:10], pol, Certainty.CONFIRMED,
            summary, "FMP:dividends", None,
            {
                "dividend": amt, "adj_dividend": r.get("adjDividend"), "yield": r.get("yield"),
                "frequency": r.get("frequency"), "change": change, "ex_date": r.get("date"),
                "record_date": r.get("recordDate"), "payment_date": r.get("paymentDate"),
            },
        ))
    return out


# --- M&A (market-wide feed matched to the universe) --------------------------

def mergers_to_catalysts(rows: list[dict], universe: set[str], start: str, end: str) -> list[Catalyst]:
    uni = {t.upper() for t in universe}
    out: list[Catalyst] = []
    seen: set[tuple] = set()
    for r in rows:
        td = r.get("transactionDate", "")
        if not _in_range(td, start, end):
            continue
        acq, tgt = (r.get("symbol") or "").upper(), (r.get("targetedSymbol") or "").upper()
        pub = fmp._news_dt_to_utc(r.get("acceptedDate", "")) or None
        for tk, role in ((acq, "acquirer"), (tgt, "target")):
            if not tk or tk not in uni:
                continue
            key = (tk, acq, tgt, td[:10])
            if key in seen:
                continue
            seen.add(key)
            pol = Polarity.POSITIVE if role == "target" else Polarity.NEUTRAL
            summary = (f"{r.get('companyName')} ({acq}) to acquire "
                       f"{r.get('targetedCompanyName')} ({tgt})")
            out.append(Catalyst(
                tk, CatalystType.MNA, td[:10], pol, Certainty.CONFIRMED, summary, "FMP:mergers", pub,
                {
                    "role": role, "acquirer": acq, "target": tgt,
                    "acquirer_name": r.get("companyName"), "target_name": r.get("targetedCompanyName"),
                    "link": r.get("link"),
                },
            ))
    return out


# --- per-ticker driver -------------------------------------------------------

def build_ticker_catalysts(ticker: str, start: str, end: str) -> list[Catalyst]:
    """All per-ticker structured catalysts for ``ticker`` in [start, end].

    (M&A is a market-wide feed handled once at the universe level, not here.)
    """
    out: list[Catalyst] = []
    out += earnings_to_catalysts(ticker, fmp.fetch_earnings(ticker), start, end)
    out += grades_to_catalysts(ticker, fmp.fetch_grades(ticker), start, end)
    out += price_target_to_catalysts(
        ticker, fmp.fetch_price_target_news(ticker, stop_before=start), start, end)
    out += dividends_to_catalysts(ticker, fmp.fetch_dividends(ticker), start, end)
    return out
