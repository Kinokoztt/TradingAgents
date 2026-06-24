"""L2: aggregate L1 stock signals into dynamic concept-cluster signals.

L2 sits on the concept-graph snapshot (M1/M2). It does two things, both pure
logic (no LLM, offline-testable):

1. ``propagate_catalysts``: a strong single-stock catalyst leaks a decayed,
   capped fraction to its graph neighbours (co-mention/co-move), so a catalyst
   on one name lifts its peers before per-name analysis even sees them.
2. ``aggregate_concepts``: roll member StockSignals up to each cluster and emit
   a ``ConceptSignal`` (label, strength, members, rationale).

See docs/regime-gate-design.md §5.3 and docs/concept-graph-design.md.
"""

from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta

from pydantic import BaseModel, Field

from tradingagents.concept_graph import store
from tradingagents.concept_graph.schemas import Cluster, Membership
from tradingagents.concept_graph.sectors import normalize_sector
from tradingagents.market_tools import MarketDataTools, get_market_tools

from ._llm import clip_text
from .schemas import ConceptSignal, Direction, StockSignal, Strength
from .tickers import canonical_ticker

DEFAULT_CONCEPT_MODEL = "gemini-3.1-pro-preview"


def _days_before(date_str: str, days: int) -> str:
    return (datetime.fromisoformat(date_str[:10]) - timedelta(days=days)).strftime("%Y-%m-%d")


class _ConceptVerdict(BaseModel):
    """LLM verdict for one concept node (theme cluster or sector)."""

    direction: Direction = Field(description="Bullish=Long / Bearish=Short / no clear lean=Block")
    strength: Strength
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str

_DIR_SIGN = {Direction.LONG: 1.0, Direction.SHORT: -1.0, Direction.BLOCK: 0.0}


def propagate_catalysts(
    as_of_date: str,
    stock_signals: list[StockSignal],
    *,
    out_dir: str = store.DEFAULT_OUT_DIR,
    neighbors_fn=None,
    top_k: int = 10,
    decay: float = 0.5,
    min_source_confidence: float = 0.6,
    max_boost: float = 0.3,
) -> list[StockSignal]:
    """Return stock_signals augmented with decayed neighbour catalysts.

    A source signal (Long/Short, confidence ≥ ``min_source_confidence``) seeds
    each neighbour ``n`` with ``conf*weight*decay`` (capped at ``max_boost``).
    Only names without an existing signal are filled; existing signals win.
    ``neighbors_fn(ticker) -> [(other, weight)]`` is injectable for tests.
    """
    if neighbors_fn is None:
        from tradingagents.concept_graph.service import get_neighbors

        def neighbors_fn(t: str):  # type: ignore[misc]
            return get_neighbors(as_of_date, t, top_k=top_k, out_dir=out_dir)

    # Key everything by canonical ticker so a graph neighbour that is just a
    # share-class sibling of an already-signalled name (e.g. GOOGL vs GOOG) is
    # not resurrected as a separate propagated signal.
    existing = {canonical_ticker(s.ticker) for s in stock_signals}
    # canonical ticker -> (best_confidence, direction, source_ticker)
    seeded: dict[str, tuple[float, Direction, str]] = {}

    for s in stock_signals:
        if s.direction is Direction.BLOCK or s.catalyst_confidence < min_source_confidence:
            continue
        for other, weight in neighbors_fn(s.ticker):
            co = canonical_ticker(other)
            if co in existing:
                continue
            boost = min(s.catalyst_confidence * weight * decay, max_boost)
            if boost <= 0:
                continue
            prev = seeded.get(co)
            if prev is None or boost > prev[0]:
                seeded[co] = (boost, s.direction, s.ticker)

    propagated = list(stock_signals)
    for ticker, (conf, direction, src) in seeded.items():
        propagated.append(
            StockSignal(
                ticker=ticker,
                direction=direction,
                catalyst_confidence=round(conf, 4),
                reason=f"[propagated from {src}] concept-graph neighbour catalyst",
            )
        )
    return propagated


def aggregate_concepts(
    as_of_date: str,
    stock_signals: list[StockSignal],
    *,
    out_dir: str = store.DEFAULT_OUT_DIR,
    cluster_map: dict[str, list[Membership]] | None = None,
    clusters: dict[str, Cluster] | None = None,
    include_secondary: bool = True,
    min_members: int = 2,
    strong_threshold: float = 0.5,
    neutral_threshold: float = 0.2,
) -> list[ConceptSignal]:
    """Roll member StockSignals up to ``ConceptSignal`` per concept cluster.

    Aggregates only actionable (Long/Short) signals, weighted by cluster
    membership weight. Clusters with fewer than ``min_members`` actionable
    members are skipped. ``cluster_map``/``clusters`` are injectable (else loaded
    from the snapshot). Output is sorted by descending strength score.
    """
    if cluster_map is None:
        cluster_map = store.load_memberships(as_of_date, out_dir)
    if clusters is None:
        clusters = store.load_clusters(as_of_date, out_dir)

    sig_by_ticker = {s.ticker: s for s in stock_signals}

    # cluster_id -> list[(weight, signal)]
    contributions: dict[str, list[tuple[float, StockSignal]]] = defaultdict(list)
    for ticker, memberships in cluster_map.items():
        sig = sig_by_ticker.get(ticker)
        if sig is None or sig.direction is Direction.BLOCK:
            continue
        for m in memberships:
            if not m.is_primary and not include_secondary:
                continue
            contributions[m.cluster_id].append((m.weight, sig))

    out: list[tuple[float, ConceptSignal]] = []
    for cid, contribs in contributions.items():
        if len(contribs) < min_members:
            continue
        total_w = sum(w for w, _ in contribs)
        if total_w <= 0:
            continue
        mean_conf = sum(w * s.catalyst_confidence for w, s in contribs) / total_w
        net = sum(w * _DIR_SIGN[s.direction] * s.catalyst_confidence for w, s in contribs) / total_w
        coherence = abs(net) / mean_conf if mean_conf > 0 else 0.0
        score = mean_conf * coherence

        if score >= strong_threshold:
            strength = Strength.STRONG
        elif score >= neutral_threshold:
            strength = Strength.NEUTRAL
        else:
            strength = Strength.WEAK

        cluster = clusters.get(cid)
        concept = (cluster.label if cluster and cluster.label else cid)
        members = cluster.members if cluster else sorted(s.ticker for _, s in contribs)
        parent_sector = cluster.parent_sector if cluster else None
        direction = Direction.LONG if net > 0 else Direction.SHORT if net < 0 else Direction.BLOCK
        lean = "bullish" if net > 0 else "bearish" if net < 0 else "mixed"
        signalled = sorted(s.ticker for _, s in contribs)
        rationale = (
            f"{len(contribs)} member(s) signalled ({', '.join(signalled)}); "
            f"net {lean} (score={net:+.2f}), mean catalyst {mean_conf:.2f}, coherence {coherence:.2f}."
        )

        out.append(
            (
                score,
                ConceptSignal(
                    concept=concept,
                    cluster_id=cid,
                    level="theme",
                    parent_sector=parent_sector,
                    direction=direction,
                    strength=strength,
                    confidence=round(score, 4),
                    member_tickers=members,
                    rationale=rationale,
                ),
            )
        )

    out.sort(key=lambda x: x[0], reverse=True)
    return [cs for _, cs in out]


def _resolve_structured_llm(llm, provider: str, model: str, base_url: str | None = None):
    """Bind structured output, creating a client from env if ``llm`` is None."""
    if llm is None:
        from ._llm import build_cascade_llm

        llm = build_cascade_llm(provider, model, base_url)
    return llm.with_structured_output(_ConceptVerdict)


def _cluster_member_signals(
    stock_signals: list[StockSignal],
    cluster_map: dict[str, list[Membership]],
    include_secondary: bool,
) -> dict[str, list[StockSignal]]:
    """cluster_id -> actionable member StockSignals (Block excluded)."""
    sig_by_ticker = {s.ticker: s for s in stock_signals}
    by_cluster: dict[str, list[StockSignal]] = defaultdict(list)
    for ticker, memberships in cluster_map.items():
        sig = sig_by_ticker.get(ticker)
        if sig is None or sig.direction is Direction.BLOCK:
            continue
        for m in memberships:
            if not m.is_primary and not include_secondary:
                continue
            by_cluster[m.cluster_id].append(sig)
    return by_cluster


def judge_clusters(
    as_of_date: str,
    stock_signals: list[StockSignal],
    *,
    market: str = "US",
    tools: MarketDataTools | None = None,
    llm=None,
    provider: str = "google",
    model: str = DEFAULT_CONCEPT_MODEL,
    base_url: str | None = None,
    out_dir: str = store.DEFAULT_OUT_DIR,
    cluster_map: dict[str, list[Membership]] | None = None,
    clusters: dict[str, Cluster] | None = None,
    include_secondary: bool = True,
    min_members: int = 2,
    news_members_top_k: int = 5,
    look_back_days: int = 7,
    max_workers: int = 4,
    news_end: str | None = None,
) -> list[ConceptSignal]:
    """S2: LLM verdict per *active* theme cluster (concurrent).

    Active = passes ``aggregate_concepts`` numeric gate (≥``min_members``
    actionable members). Context = member StockSignals + cluster-level news
    refetched for the top-``news_members_top_k`` members (by confidence). ``news_end``
    (RFC3339 instant) caps that news at a pre-market cutoff (defaults to
    ``as_of_date``). Quiet clusters are skipped. Returns level="theme".
    """
    if cluster_map is None:
        cluster_map = store.load_memberships(as_of_date, out_dir)
    if clusters is None:
        clusters = store.load_clusters(as_of_date, out_dir)

    gate = aggregate_concepts(
        as_of_date, stock_signals, cluster_map=cluster_map, clusters=clusters,
        include_secondary=include_secondary, min_members=min_members,
    )
    active_cids = [cs.cluster_id for cs in gate if cs.cluster_id]
    if not active_cids:
        return []

    tools = tools or get_market_tools(market)
    structured = _resolve_structured_llm(llm, provider, model, base_url)
    member_signals = _cluster_member_signals(stock_signals, cluster_map, include_secondary)
    news_start = _days_before(as_of_date, look_back_days)
    news_end = news_end or as_of_date

    def judge(cid: str) -> ConceptSignal:
        cluster = clusters.get(cid)
        label = cluster.label if cluster and cluster.label else cid
        members = cluster.members if cluster else []
        parent_sector = cluster.parent_sector if cluster else None
        sigs = sorted(member_signals.get(cid, []), key=lambda s: s.catalyst_confidence, reverse=True)

        sig_lines = "\n".join(
            f"- {s.ticker}: {s.direction.value} conf={s.catalyst_confidence:.2f} — {s.reason}" for s in sigs
        )
        # Cap each member's news and the assembled block so a hot cluster's
        # refetched news can't overflow the model's context window.
        news_block = clip_text("\n\n".join(
            f"#### {s.ticker}\n{clip_text(tools.get_stock_news(s.ticker, news_start, news_end), 5000)}"
            for s in sigs[:news_members_top_k]
        ), 20000)
        prompt = f"""You are a thematic equity analyst. Judge concept cluster '{label}'
(members: {', '.join(members)}) as of {as_of_date}.

Give a `direction` (Long=bullish theme, Short=bearish theme, Block=no clear lean),
a `strength` (Strong/Neutral/Weak conviction), a `confidence` in [0,1], and a `rationale`.

## Member signals (from per-stock analysis)
{sig_lines}

## Member news
{news_block}
"""
        v: _ConceptVerdict = structured.invoke(prompt)
        return ConceptSignal(
            concept=label, cluster_id=cid, level="theme", parent_sector=parent_sector,
            direction=v.direction, strength=v.strength, confidence=v.confidence,
            member_tickers=members, rationale=v.rationale,
        )

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        return list(pool.map(judge, active_cids))


def judge_sectors(
    cluster_verdicts: list[ConceptSignal],
    as_of_date: str,
    *,
    llm=None,
    provider: str = "google",
    model: str = DEFAULT_CONCEPT_MODEL,
    base_url: str | None = None,
    max_workers: int = 4,
) -> list[ConceptSignal]:
    """S3: LLM verdict per sector, aggregating its theme verdicts (concurrent).

    Sectors are grouped by the theme verdicts' ``parent_sector``; sectors with no
    active theme are simply absent (Neutral downstream). Returns level="sector".
    """
    by_sector: dict[str, list[ConceptSignal]] = defaultdict(list)
    for cv in cluster_verdicts:
        sector = normalize_sector(cv.parent_sector)
        if sector:
            by_sector[sector].append(cv)
    if not by_sector:
        return []

    structured = _resolve_structured_llm(llm, provider, model, base_url)

    def judge(item: tuple[str, list[ConceptSignal]]) -> ConceptSignal:
        sector, themes = item
        theme_lines = "\n".join(
            f"- {t.concept}: {t.direction.value} / {t.strength.value} conf={t.confidence:.2f} — {t.rationale}"
            for t in themes
        )
        members = sorted({m for t in themes for m in t.member_tickers})
        prompt = f"""You are a sector strategist. Judge sector '{sector}' as of {as_of_date},
aggregating its theme-cluster verdicts below.

Give a `direction` (Long=bullish sector, Short=bearish, Block=no clear lean),
a `strength`, a `confidence` in [0,1], and a `rationale`.

## Theme verdicts
{theme_lines}
"""
        v: _ConceptVerdict = structured.invoke(prompt)
        return ConceptSignal(
            concept=sector, cluster_id=None, level="sector", parent_sector=None,
            direction=v.direction, strength=v.strength, confidence=v.confidence,
            member_tickers=members, rationale=v.rationale,
        )

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        return list(pool.map(judge, by_sector.items()))
