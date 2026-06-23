"""L3: market-regime verdict (the strategic commander).

L3 reads only macro/calendar/market-news (no per-stock LLM cost), asks the deep
model for a market regime + macro synthesis, then assembles the top-level
``RegimeReport``.

The "circuit breaker" is **not** applied by overwriting lower-layer signals.
L1/L2 judgments are kept raw; the regime veto (Bearish ⇒ no Longs, Range ⇒ only
high-confidence Longs) is a *consumption-time rule* exposed via
``RegimeReport.tradable_long_whitelist`` / ``regime_blocked_longs``. L3 just
records ``range_min_confidence`` so that rule is reproducible. See
docs/regime-gate-main-flow.md.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from pydantic import BaseModel, Field

from tradingagents.market_tools import MarketDataTools, get_market_tools

from .schemas import ConceptSignal, HorizonOutlook, MarketRegime, RegimeReport, StockSignal

DEFAULT_REGIME_MODEL = "gemini-3.1-pro-preview"  # deep thinking for the commander
OUTLOOK_HORIZONS = ("1d", "3d", "5d")  # B: forward windows the commander forecasts


class _L3Verdict(BaseModel):
    """What the commander LLM returns; the rest of RegimeReport is assembled locally."""

    market_state: MarketRegime
    macro_summary: str = Field(description="Macro/fundamental synthesis, micro-noise stripped")
    key_drivers: list[str] = Field(default_factory=list, description="Top regime drivers")
    outlook: list[HorizonOutlook] = Field(
        default_factory=list,
        description="Multi-horizon market call: one HorizonOutlook per 1d/3d/5d forward window",
    )


def _build_prompt(
    as_of_date: str,
    macro_summary: str,
    market_news: str,
    calendar: str,
    concept_signals: list[ConceptSignal],
) -> str:
    concept_block = (
        "\n".join(f"- {c.concept} [{c.strength.value}]: {', '.join(c.member_tickers[:8])}" for c in concept_signals)
        if concept_signals
        else "(none provided this run)"
    )
    return f"""You are the strategic commander and circuit breaker of a trading system.
Decide the short-term US market regime as of {as_of_date}. Reason macro-first,
strip micro-noise, and be willing to call Bearish/Range when systemic risk is
present. Output `market_state` (Bullish/Range/Bearish), a concise `macro_summary`,
and the `key_drivers`.

Also give a multi-horizon `outlook`: a separate Bullish/Range/Bearish call for
each of the 1d, 3d, and 5d forward windows (trading days, the session itself
counts as day 1), each with a 0-1 `confidence` and a one-line `rationale`. These
horizons can diverge — e.g. an overnight CPI print is a hard 1d constraint but
may be digested by 5d; a slow earnings-season drift may only show up at 3d/5d.
`market_state` is the near-term anchor and should align with the 1d call.

## Macro structured snapshot
{macro_summary}

## Upcoming macro calendar
{calendar}

## Market/global news
{market_news}

## Concept-cluster signals (L2)
{concept_block}
"""


def analyze_regime(
    as_of_date: str,
    market: str = "US",
    concept_signals: list[ConceptSignal] | None = None,
    stock_signals: list[StockSignal] | None = None,
    *,
    look_back_days: int = 10,
    calendar_forward_days: int = 7,
    range_min_confidence: float = 0.6,
    llm=None,
    provider: str = "google",
    model: str = DEFAULT_REGIME_MODEL,
    base_url: str | None = None,
    tools: MarketDataTools | None = None,
    news_end: str | None = None,
) -> RegimeReport:
    """Run L3 and return a ``RegimeReport`` with **raw** L1/L2 signals preserved.

    ``concept_signals``/``stock_signals`` come from L2/L1; passing none yields the
    macro-only regime skeleton. The regime veto is NOT baked into the signals —
    ``range_min_confidence`` is recorded on the report and applied at consumption
    via ``tradable_long_whitelist``. ``news_end`` ("YYYY-MM-DD HH:MM:SS") caps
    market news at a pre-market cutoff. Macro uses ``as_of_date`` directly (the
    shifted macro_daily row is the prior close, visible pre-open). Injectable for tests.
    """
    tools = tools or get_market_tools(market)
    concept_signals = concept_signals or []
    stock_signals = stock_signals or []

    macro_summary = tools.get_macro_summary(as_of_date, look_back_days)
    market_news = tools.get_market_news(as_of_date, look_back_days, end_datetime=news_end)
    cal_end = (datetime.fromisoformat(as_of_date[:10]) + timedelta(days=calendar_forward_days)).strftime("%Y-%m-%d")
    calendar = tools.get_economic_calendar(as_of_date, cal_end, cutoff=news_end)

    if llm is None:
        from ._llm import build_cascade_llm

        llm = build_cascade_llm(provider, model, base_url)

    structured = llm.with_structured_output(_L3Verdict)
    verdict: _L3Verdict = structured.invoke(
        _build_prompt(as_of_date, macro_summary, market_news, calendar, concept_signals)
    )

    return RegimeReport(
        as_of_date=as_of_date,
        market_state=verdict.market_state,
        macro_summary=verdict.macro_summary,
        key_drivers=verdict.key_drivers,
        macro_snapshot=macro_summary,
        economic_calendar=calendar,
        range_min_confidence=range_min_confidence,
        outlook=verdict.outlook,
        concept_signals=concept_signals,
        stock_signals=stock_signals,
    )
