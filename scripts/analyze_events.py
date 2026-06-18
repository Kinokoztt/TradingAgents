"""Summarize extracted news events (events.jsonl) over a date range.

Pure local read — no BigQuery. Scans {out_dir}/{YYYY-MM-DD}/events.jsonl, then
reports the distributions you need to judge extraction quality before tuning the
taxonomy / prompts:
  - field distributions: event_type, certainty, polarity, price_in, source tier
  - cross-tabs: event_type x polarity, event_type x certainty
  - primary vs secondary share, duplicate ratio (events per unique article url)
  - per-session counts, top sources, top tickers

Example:
    python scripts/analyze_events.py --start 2026-05-01 --end 2026-05-31
    python scripts/analyze_events.py            # all sessions found under out-dir
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tradingagents.regime.store import DEFAULT_OUT_DIR, EVENTS_FILE, load_events

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _sessions(out_dir: str, start: str | None, end: str | None) -> list[str]:
    if not os.path.isdir(out_dir):
        raise SystemExit(f"out-dir not found: {out_dir}")
    days = []
    for name in os.listdir(out_dir):
        if not _DATE_RE.match(name):
            continue
        if not os.path.exists(os.path.join(out_dir, name, EVENTS_FILE)):
            continue
        if start and name < start:
            continue
        if end and name > end:
            continue
        days.append(name)
    return sorted(days)


def _dist(title: str, counter: Counter, total: int) -> None:
    print(f"\n{title}  (n={total})")
    print(f"  {'value':<16}{'count':>8}{'pct':>8}")
    for val, cnt in counter.most_common():
        pct = cnt / total if total else 0.0
        print(f"  {val:<16}{cnt:>8}{pct:>7.1%}")


def _crosstab(title: str, rows: dict[str, Counter], col_order: list[str]) -> None:
    print(f"\n{title}")
    header = "  " + f"{'':<16}" + "".join(f"{c:>13}" for c in col_order) + f"{'total':>9}"
    print(header)
    for rk in sorted(rows, key=lambda k: -sum(rows[k].values())):
        cells = "".join(f"{rows[rk].get(c, 0):>13}" for c in col_order)
        print(f"  {rk:<16}{cells}{sum(rows[rk].values()):>9}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--start", default=None, help="First session YYYY-MM-DD (inclusive)")
    p.add_argument("--end", default=None, help="Last session YYYY-MM-DD (inclusive)")
    p.add_argument("--top", type=int, default=15, help="Top-N sources/tickers to list")
    p.add_argument("--primary-only", action="store_true", help="Restrict stats to is_primary events")
    args = p.parse_args()

    sessions = _sessions(args.out_dir, args.start, args.end)
    if not sessions:
        raise SystemExit(f"no events.jsonl found under {args.out_dir} in the given range")

    events = []
    per_session = Counter()
    for s in sessions:
        evs = load_events(s, out_dir=args.out_dir)
        if args.primary_only:
            evs = [e for e in evs if e.is_primary]
        per_session[s] = len(evs)
        events.extend(evs)

    total = len(events)
    if total == 0:
        raise SystemExit("0 events in range")

    primary = sum(1 for e in events if e.is_primary)
    urls = [e.article_url for e in events if e.article_url]
    uniq_urls = len(set(urls))
    dup_ratio = (len(urls) / uniq_urls) if uniq_urls else 0.0

    print("=" * 64)
    print(f"{total} events across {len(sessions)} session(s)  "
          f"({sessions[0]} .. {sessions[-1]})")
    print(f"tickers: {len({e.ticker for e in events})}  "
          f"primary: {primary} ({primary/total:.1%})  secondary: {total-primary} ({(total-primary)/total:.1%})")
    print(f"articles: {len(urls)} urls, {uniq_urls} unique  -> {dup_ratio:.2f} events/url "
          f"(>1 means repeated coverage)")

    _dist("event_type", Counter(e.event_type.value for e in events), total)
    _dist("certainty", Counter(e.certainty.value for e in events), total)
    _dist("polarity", Counter(e.polarity.value for e in events), total)
    _dist("price_in", Counter(e.price_in.value for e in events), total)
    _dist("source_reliability", Counter(e.source_reliability.value for e in events), total)

    pol_cols = ["Positive", "Negative", "Neutral", "Mixed"]
    cert_cols = ["Confirmed", "Unconfirmed"]
    et_pol: dict[str, Counter] = defaultdict(Counter)
    et_cert: dict[str, Counter] = defaultdict(Counter)
    for e in events:
        et_pol[e.event_type.value][e.polarity.value] += 1
        et_cert[e.event_type.value][e.certainty.value] += 1
    _crosstab("event_type x polarity", et_pol, pol_cols)
    _crosstab("event_type x certainty", et_cert, cert_cols)

    print("\nper-session counts")
    for s in sessions:
        print(f"  {s}  {per_session[s]:>6}")

    print(f"\ntop {args.top} sources")
    for src, cnt in Counter(e.source for e in events if e.source).most_common(args.top):
        print(f"  {src:<28}{cnt:>6}")

    print(f"\ntop {args.top} tickers")
    for tk, cnt in Counter(e.ticker for e in events).most_common(args.top):
        print(f"  {tk:<10}{cnt:>6}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
