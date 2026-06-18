# NN Pipeline Roadmap (events -> encoding -> training)

## Why this exists

Asking an LLM to output a trading **direction** (buy/sell, Long/Short) directly
has no track record of reliability — no frontier model has been shown to do this
natively. So we are repositioning the LLM to the one job it is good at: turning
unstructured news/filings into a **standardized, typed event record**. The
direction call moves out of the LLM and into a downstream model trained on
realized returns.

This document describes the planned (not-yet-built) training stage. The data it
consumes — `events.jsonl` — is produced today by `scripts/extract_events.py`.

```mermaid
flowchart LR
  News["News (vendor summaries)"] --> Extract["extract_events (LLM, classification only)"]
  Extract --> Enrich["source_reliability + price_in"]
  Enrich --> JSONL["events.jsonl (standardized corpus)"]
  JSONL --> Encode["encoding (this roadmap)"]
  Prices["prices / fundamentals"] --> Features["feature join"]
  Encode --> Features
  Features --> Train["NN training"]
  Train --> Backtest["walk-forward backtest"]
```

## Stage 0 — standardized corpus (DONE)

`NewsEvent` (see `tradingagents/regime/events.py`) per article/ticker, with:
- LLM classification: `event_type`, `certainty`, `polarity`, `materiality`, `horizon`, `summary`.
- provenance: `source`, `article_url`, `published_utc`, `event_date`.
- enrichment: `source_reliability` (publisher tier), `price_in`
  (`NotPricedIn/Partial/PricedIn/PostHoc`), `pre/post_return`, `pre/post_volume_ratio`.

Persisted as JSONL at `regime_gate_output/{as_of}/events.jsonl`.

## Stage 1 — encoding (planned)

Per event, build a feature vector:
- Categorical one-hot: `event_type`, `certainty`, `polarity`, `materiality`,
  `horizon`, `source_reliability`.
- `summary` -> sentence embedding (local encoder served alongside the vLLM box;
  e.g. a Qwen/BGE embedding model). Keep the embedding model **frozen and
  versioned** so features are reproducible across re-runs.
- price-in numerics: `pre_return`, `post_return`, `pre/post_volume_ratio`, and
  the `price_in` label one-hot (it is a strong leakage signal — an event that is
  already `PricedIn`/`PostHoc` should carry little forward alpha).

Aggregate to a (ticker, session) example by pooling that session's events
(e.g. attention/mean over event vectors), then concatenate point-in-time
market + fundamental features already available via `market_tools`.

## Stage 2 — labels (planned)

Supervised target = forward return over a fixed horizon (1d/3d/5d), measured
from the next tradable open (consistent with `regime/evaluate.py`), optionally
market-neutralized against SPY/QQQ. Leakage guards: features must be strictly
point-in-time (news capped at the pre-market cutoff, fundamentals filtered by
SEC acceptedDate — both already enforced upstream).

## Stage 3 — model + training (planned)

- Start simple (gradient-boosted trees / shallow MLP) as a baseline before any
  sequence model; the event corpus is tabular-ish after pooling.
- Train on the self-hosted GPUs; keep the whole loop offline and cheap so the
  expected many rollback/re-train cycles do not incur API cost.
- Walk-forward validation by date; never shuffle across the time boundary.

## Stage 4 — backtest + feedback (planned)

Feed model scores into the existing whitelist/veto consumption rules and grade
against realized paths with the existing evaluator, closing the loop so source
reliability and event types can eventually be **learned** from outcomes rather
than hand-set.

## Open decisions (defer until Stage 1)

- Embedding model + dimensionality, and whether to fine-tune it.
- Event de-duplication policy across articles describing the same event.
- Pooling architecture for multi-event sessions.
