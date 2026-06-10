"""L3: market-regime verdict + circuit breaker (the strategic commander).

L3 reads only macro/calendar/market-news (no per-stock LLM cost), asks the deep
model for a market regime + macro synthesis, then assembles the top-level
``RegimeReport``. The circuit breaker is the "physical" override: in a
Bearish/Range regime it demotes weak/long permits to BLOCK regardless of what
L1/L2 proposed. See docs/regime-gate-design.md §5.3.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from pydantic import BaseModel, Field

from tradingagents.market_tools import MarketDataTools, get_market_tools

from .schemas import ConceptSignal, Direction, MarketRegime, RegimeReport, StockSignal

DEFAULT_REGIME_MODEL = "gemini-3.1-pro"  # deep thinking for the commander


class _L3Verdict(BaseModel):
    """What the commander LLM returns; the rest of RegimeReport is assembled locally."""

    market_state: MarketRegime
    macro_summary: str = Field(description="Macro/fundamental synthesis, micro-noise stripped")
    key_drivers: list[str] = Field(default_factory=list, description="Top regime drivers")


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

## Macro structured snapshot
{macro_summary}

## Upcoming macro calendar
{calendar}

## Market/global news
{market_news}

## Concept-cluster signals (L2)
{concept_block}
"""


def apply_circuit_breaker(report: RegimeReport, range_min_confidence: float = 0.6) -> RegimeReport:
    """Override long permits when the regime is hostile.

    - Bearish: every Long → Block.
    - Range: Long with catalyst_confidence < ``range_min_confidence`` → Block.
    - Bullish: untouched.
    """
    if report.market_state is MarketRegime.BULLISH:
        return report

    new_signals: list[StockSignal] = []
    for s in report.stock_signals:
        if report.market_state is MarketRegime.BEARISH and s.direction is Direction.LONG:
            new_signals.append(
                s.model_copy(update={"direction": Direction.BLOCK, "reason": f"[circuit-breaker:bearish] {s.reason}"})
            )
        elif (
            report.market_state is MarketRegime.RANGE
            and s.direction is Direction.LONG
            and s.catalyst_confidence < range_min_confidence
        ):
            new_signals.append(
                s.model_copy(
                    update={
                        "direction": Direction.BLOCK,
                        "reason": f"[circuit-breaker:range<{range_min_confidence}] {s.reason}",
                    }
                )
            )
        else:
            new_signals.append(s)
    return report.model_copy(update={"stock_signals": new_signals})


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
    tools: MarketDataTools | None = None,
    news_end: str | None = None,
) -> RegimeReport:
    """Run L3 and return a circuit-broken ``RegimeReport``.

    ``concept_signals``/``stock_signals`` come from L2/L1; passing none yields the
    macro-only regime skeleton. ``news_end`` ("YYYY-MM-DD HH:MM:SS") caps market
    news at a pre-market cutoff. Macro uses ``as_of_date`` directly (the shifted
    macro_daily row is the prior close, visible pre-open). Injectable for tests.
    """
    tools = tools or get_market_tools(market)
    concept_signals = concept_signals or []
    stock_signals = stock_signals or []

    macro_summary = tools.get_macro_summary(as_of_date, look_back_days)
    market_news = tools.get_market_news(as_of_date, look_back_days, end_datetime=news_end)
    cal_end = (datetime.fromisoformat(as_of_date[:10]) + timedelta(days=calendar_forward_days)).strftime("%Y-%m-%d")
    calendar = tools.get_economic_calendar(as_of_date, cal_end)

    if llm is None:
        import os

        from tradingagents.llm_clients import create_llm_client

        client = create_llm_client(provider, model, google_api_key=os.getenv("GOOGLE_API_KEY"))
        llm = client.get_llm()

    structured = llm.with_structured_output(_L3Verdict)
    verdict: _L3Verdict = structured.invoke(
        _build_prompt(as_of_date, macro_summary, market_news, calendar, concept_signals)
    )

    report = RegimeReport(
        as_of_date=as_of_date,
        market_state=verdict.market_state,
        macro_summary=verdict.macro_summary,
        concept_signals=concept_signals,
        stock_signals=stock_signals,
    )
    return apply_circuit_breaker(report, range_min_confidence=range_min_confidence)
