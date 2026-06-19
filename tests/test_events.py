"""Tests for standardized event extraction, source reliability, price-in, store."""

from __future__ import annotations

import re

import pandas as pd
import pytest

from tradingagents.regime import events as events_mod
from tradingagents.regime.events import (
    Certainty,
    EventType,
    NewsEvent,
    Polarity,
    PriceInStatus,
    SourceReliability,
    extract_events,
)
from tradingagents.regime.price_in import label_price_in, tag_price_in
from tradingagents.regime.source_reliability import classify_source, tag_source_reliability
from tradingagents.regime.store import (
    append_events,
    load_event_progress,
    load_events,
    mark_event_progress,
    save_events,
)

pytestmark = pytest.mark.unit


# --- schema ---------------------------------------------------------------

def _make_event(**overrides) -> NewsEvent:
    base = dict(
        ticker="AAPL",
        as_of_date="2026-05-11",
        event_type=EventType.EARNINGS,
        certainty=Certainty.CONFIRMED,
        polarity=Polarity.POSITIVE,
        is_primary=True,
        summary="Q2 EPS above consensus",
    )
    base.update(overrides)
    return NewsEvent(**base)


def test_news_event_defaults_to_unknown_enrichment():
    ev = _make_event()
    assert ev.source_reliability is SourceReliability.UNKNOWN
    assert ev.price_in is PriceInStatus.UNKNOWN
    assert ev.pre_return is None and ev.post_return is None


def test_news_event_json_roundtrip():
    ev = _make_event(source="Reuters", published_utc="2026-05-10T13:00:00Z", is_primary=False)
    restored = NewsEvent.model_validate_json(ev.model_dump_json())
    assert restored == ev


# --- extraction (two-stage) -----------------------------------------------

_ARTICLES = {
    "AAPL": [
        {"date": "2026-05-10", "title": "Apple beats", "publisher": "Reuters",
         "description": "EPS above consensus", "insights": [], "published_utc": "2026-05-10T13:00:00Z",
         "article_url": "http://x/1"},
        {"date": "2026-05-09", "title": "Apple sued", "publisher": "Simply Wall St",
         "description": "Patent suit filed", "insights": [], "published_utc": "2026-05-09T18:00:00Z",
         "article_url": "http://x/2"},
    ],
}


def _indices(prompt: str) -> list[int]:
    return [int(i) for i in re.findall(r"^\[(\d+)\]", prompt, flags=re.MULTILINE)]


class _FakeStage1:
    """One event per article index parsed from the stage-1 prompt."""

    def invoke(self, prompt):
        return events_mod.Stage1Extraction(events=[
            events_mod.Stage1Event(article_index=i, is_primary=True, summary=f"event {i}")
            for i in _indices(prompt)
        ])


class _FakeStage2:
    """One label per summary index parsed from the stage-2 prompt."""

    def invoke(self, prompt):
        return events_mod.Stage2Labels(labels=[
            events_mod.Stage2Label(index=i, event_type=EventType.OTHER, polarity=Polarity.NEUTRAL)
            for i in _indices(prompt)
        ])


class _FakeLLM:
    def with_structured_output(self, schema):
        return _FakeStage1() if schema is events_mod.Stage1Extraction else _FakeStage2()


def test_extract_events_attaches_provenance(monkeypatch):
    monkeypatch.setattr(
        events_mod.massive, "fetch_news_articles",
        lambda start, end, ticker=None, max_articles=50: _ARTICLES.get(ticker, []),
    )
    out = extract_events(["AAPL"], "2026-05-11", llm=_FakeLLM(), source="massive", min_source_tier=None)
    assert len(out) == 2
    first = next(e for e in out if e.summary == "event 0")
    assert first.ticker == "AAPL"
    assert first.source == "Reuters"
    assert first.published_utc == "2026-05-10T13:00:00Z"
    assert first.article_url == "http://x/1"
    assert first.event_type is EventType.OTHER
    # certainty is derived from the source: Reuters is not a primary wire -> Unconfirmed
    assert first.certainty is Certainty.UNCONFIRMED
    assert first.is_primary is True


def test_extract_events_certainty_from_wire(monkeypatch):
    wire = {"WIRE": [{"date": "2026-05-10", "title": "8-K filed", "publisher": "GlobeNewswire",
                      "description": "Board declared a dividend", "insights": [],
                      "published_utc": "2026-05-10T13:00:00Z", "article_url": "http://x/9"}]}
    monkeypatch.setattr(
        events_mod.massive, "fetch_news_articles",
        lambda start, end, ticker=None, max_articles=50: wire.get(ticker, []),
    )
    out = extract_events(["WIRE"], "2026-05-11", llm=_FakeLLM(), source="massive", min_source_tier=None)
    assert out and out[0].certainty is Certainty.CONFIRMED  # primary wire -> Confirmed


def test_extract_events_drops_out_of_range_index(monkeypatch):
    monkeypatch.setattr(
        events_mod.massive, "fetch_news_articles",
        lambda start, end, ticker=None, max_articles=50: _ARTICLES.get(ticker, []),
    )

    class _BadStage1:
        def invoke(self, prompt):
            return events_mod.Stage1Extraction(events=[events_mod.Stage1Event(
                article_index=99, is_primary=True, summary="ghost",
            )])

    class _BadLLM:
        def with_structured_output(self, schema):
            return _BadStage1() if schema is events_mod.Stage1Extraction else _FakeStage2()

    out = extract_events(["AAPL"], "2026-05-11", llm=_BadLLM(), source="massive", min_source_tier=None)
    assert out == []  # ghost index dropped, not guessed


def test_extract_events_empty_is_noop():
    assert extract_events([], "2026-05-11", llm=_FakeLLM()) == []


def test_fetch_ticker_articles_drops_low_tier(monkeypatch):
    arts = [
        {"date": "2026-05-10", "title": "a", "publisher": "Reuters", "description": "x",
         "insights": [], "published_utc": "2026-05-10T13:00:00Z", "article_url": "u1"},
        {"date": "2026-05-10", "title": "b", "publisher": "The Motley Fool", "description": "y",
         "insights": [], "published_utc": "2026-05-10T13:00:00Z", "article_url": "u2"},
    ]
    monkeypatch.setattr(
        events_mod.massive, "fetch_news_articles",
        lambda start, end, ticker=None, max_articles=50: arts,
    )
    kept = events_mod.fetch_ticker_articles("AAPL", "2026-05-11", source="massive",
                                            min_source_tier=SourceReliability.MEDIUM)
    assert [a["publisher"] for a in kept] == ["Reuters"]  # Motley Fool (LOW) dropped


# --- source reliability ---------------------------------------------------

def test_classify_source_tiers():
    assert classify_source("Reuters") is SourceReliability.HIGH
    assert classify_source("Thomson Reuters") is SourceReliability.HIGH  # substring
    assert classify_source("Zacks Investment Research") is SourceReliability.MEDIUM  # substring
    assert classify_source("Simply Wall St") is SourceReliability.LOW
    assert classify_source("The Motley Fool") is SourceReliability.LOW  # opinion mill
    assert classify_source("24/7 Wall Street") is SourceReliability.LOW
    assert classify_source("Some Random Blog") is SourceReliability.UNKNOWN
    assert classify_source("") is SourceReliability.UNKNOWN


def test_certainty_for_source_rule():
    from tradingagents.regime.source_reliability import certainty_for_source, meets_min_tier

    assert certainty_for_source("GlobeNewswire") is Certainty.CONFIRMED  # primary wire
    assert certainty_for_source("Business Wire") is Certainty.CONFIRMED
    assert certainty_for_source("Reuters") is Certainty.UNCONFIRMED  # report, not disclosure
    assert certainty_for_source("The Motley Fool") is Certainty.UNCONFIRMED
    assert certainty_for_source("") is Certainty.UNCONFIRMED

    assert meets_min_tier("Reuters", SourceReliability.MEDIUM) is True
    assert meets_min_tier("The Motley Fool", SourceReliability.MEDIUM) is False
    assert meets_min_tier("Some Random Blog", SourceReliability.MEDIUM) is False


def test_is_litigation_solicitation():
    from tradingagents.regime.source_reliability import is_litigation_solicitation

    # law-firm shareholder-alert / class-action solicitation -> drop
    assert is_litigation_solicitation(
        "ROSEN, A LEADING LAW FIRM, Encourages Camping World Investors to Secure Counsel "
        "Before Important Lead Plaintiff Deadline")
    assert is_litigation_solicitation(
        "Berger Montague: ODDITY Tech investors who purchased shares are encouraged to contact")
    assert is_litigation_solicitation("Investors are reminded of the class period and lead plaintiff deadline")
    # securities class-action genre (even when paraphrased without an explicit cue)
    assert is_litigation_solicitation(
        "A class action lawsuit has been filed against Oddity Tech for violations of the "
        "Securities Exchange Act of 1934")
    # genuine legal reporting -> keep
    assert not is_litigation_solicitation(
        "Meta returns to court in New Mexico for an ongoing child safety trial")
    assert not is_litigation_solicitation("SEC charges company executives with accounting fraud")
    assert not is_litigation_solicitation(
        "A worker injured in a Kinder Morgan pipeline explosion filed a lawsuit against the company")
    assert not is_litigation_solicitation("")


def test_fetch_ticker_articles_drops_solicitation(monkeypatch):
    arts = [
        {"date": "2026-05-04", "title": "Pomerantz Law Firm reminds investors of CWH class action deadline",
         "publisher": "GlobeNewswire", "description": "lead plaintiff deadline", "insights": [],
         "published_utc": "2026-05-04T12:00:00Z", "article_url": "u1"},
        {"date": "2026-05-04", "title": "Apple beats earnings estimates", "publisher": "Reuters",
         "description": "results topped consensus", "insights": [],
         "published_utc": "2026-05-04T12:00:00Z", "article_url": "u2"},
    ]
    monkeypatch.setattr(
        events_mod.massive, "fetch_news_articles",
        lambda start, end, ticker=None, max_articles=50: arts,
    )
    kept = events_mod.fetch_ticker_articles("AAPL", "2026-05-04", source="massive", min_source_tier=None)
    assert [a["article_url"] for a in kept] == ["u2"]  # solicitation dropped, real news kept


def test_tag_source_reliability_in_place():
    evs = [_make_event(source="Reuters"), _make_event(source="Insider Monkey")]
    tag_source_reliability(evs)
    assert evs[0].source_reliability is SourceReliability.HIGH
    assert evs[1].source_reliability is SourceReliability.LOW


# --- price-in -------------------------------------------------------------

def _frame(closes: list[float]) -> pd.DataFrame:
    dates = pd.bdate_range("2026-01-01", periods=len(closes))
    rows = []
    for i, c in enumerate(closes):
        o = closes[i - 1] if i > 0 else c
        hi = max(o, c) * 1.005
        lo = min(o, c) * 0.995
        rows.append({"ticker": "T", "trade_date": dates[i], "open": o, "high": hi, "low": lo,
                     "close": c, "volume": 1000})
    return pd.DataFrame(rows)


def _event_on(df: pd.DataFrame, idx: int, hour: int) -> NewsEvent:
    d = df["trade_date"].iloc[idx].strftime("%Y-%m-%d")
    return _make_event(ticker="T", published_utc=f"{d}T{hour:02d}:00:00Z")


def test_price_in_not_priced_in():
    # flat through reaction (idx 16), then a large post-event jump
    closes = [100.0] * 16 + [106.67, 113.33, 120.0, 120.0]
    df = _frame(closes)
    out = label_price_in(_event_on(df, 16, hour=13), df)
    assert out["price_in"] is PriceInStatus.NOT_PRICED_IN
    assert out["post_return"] > 0


def test_price_in_priced_in_intraday():
    # large pre-event ramp (idx 13-15), flat after; published intraday
    closes = [100.0] * 13 + [106.67, 113.33, 120.0] + [120.0] * 4
    df = _frame(closes)
    out = label_price_in(_event_on(df, 16, hour=13), df)
    assert out["price_in"] is PriceInStatus.PRICED_IN


def test_price_in_post_hoc_after_hours():
    # same prior move, but published after the close -> recap of a done move
    closes = [100.0] * 13 + [106.67, 113.33, 120.0] + [120.0] * 4
    df = _frame(closes)
    out = label_price_in(_event_on(df, 16, hour=21), df)
    assert out["price_in"] is PriceInStatus.POST_HOC


def test_price_in_partial():
    closes = [100.0] * 13 + [106.67, 113.33, 120.0] + [126.67, 133.33, 140.0, 140.0]
    df = _frame(closes)
    out = label_price_in(_event_on(df, 16, hour=13), df)
    assert out["price_in"] is PriceInStatus.PARTIAL


def test_price_in_unknown_without_timestamp():
    df = _frame([100.0] * 20)
    out = label_price_in(_make_event(ticker="T", published_utc=""), df)
    assert out["price_in"] is PriceInStatus.UNKNOWN


def test_price_in_unknown_insufficient_history():
    df = _frame([100.0] * 5)  # not enough for ATR window
    out = label_price_in(_event_on(df, 4, hour=13), df)
    assert out["price_in"] is PriceInStatus.UNKNOWN


class _FakeTools:
    def __init__(self, df):
        self._df = df

    def load_daily_ohlcv(self, tickers, start_date, end_date):
        return self._df


def test_tag_price_in_batch_in_place():
    closes = [100.0] * 16 + [106.67, 113.33, 120.0, 120.0]
    df = _frame(closes)
    ev = _event_on(df, 16, hour=13)
    tag_price_in([ev], tools=_FakeTools(df))
    assert ev.price_in is PriceInStatus.NOT_PRICED_IN
    assert ev.post_return is not None


# --- store ----------------------------------------------------------------

def test_save_load_events_roundtrip(tmp_path):
    evs = [
        _make_event(source="Reuters", source_reliability=SourceReliability.HIGH,
                    price_in=PriceInStatus.NOT_PRICED_IN, post_return=0.2),
        _make_event(ticker="NVDA", event_type=EventType.GUIDANCE),
    ]
    path = save_events("2026-05-11", evs, out_dir=str(tmp_path))
    assert path.endswith("2026-05-11/events.jsonl")
    loaded = load_events("2026-05-11", out_dir=str(tmp_path))
    assert loaded == evs


def test_append_events_and_progress_resume(tmp_path):
    out = str(tmp_path)
    append_events("2026-05-11", [_make_event(ticker="AAPL")], out_dir=out)
    mark_event_progress("2026-05-11", "AAPL", out_dir=out)
    append_events("2026-05-11", [_make_event(ticker="NVDA")], out_dir=out)
    mark_event_progress("2026-05-11", "NVDA", out_dir=out)

    assert load_event_progress("2026-05-11", out_dir=out) == {"AAPL", "NVDA"}
    loaded = load_events("2026-05-11", out_dir=out)
    assert [e.ticker for e in loaded] == ["AAPL", "NVDA"]  # appended in order
