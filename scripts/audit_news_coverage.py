"""Audit news-source coverage: are we actually pulling all the articles?

Answers the practical questions before trusting the news pipeline:
  - How many articles does Massive return for a window, and did we hit a cap
    (i.e. silently truncate older articles)?
  - How many articles does each ticker get? Which tickers exceed the
    per-stock cap used by get_stock_news (so their feed is truncated)?
  - Which publishers dominate the feed (input to source-reliability work)?

This is read-only reconnaissance over tradingagents.dataflows.massive; it does
not write reports. MASSIVE_API_KEY comes from the env / Secret Manager.

Examples:
    python scripts/audit_news_coverage.py --as-of 2026-05-11 --look-back 3
    python scripts/audit_news_coverage.py --start 2026-05-01 --end 2026-05-11 --ticker AAPL
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tradingagents.dataflows import massive

# Mirror the per-stock cap in massive.get_news so we flag the same truncation
# the regime gate / analysts would actually experience.
PER_TICKER_CAP = 200


def _days_before(date_str: str, days: int) -> str:
    return (datetime.fromisoformat(date_str[:10]) - timedelta(days=days)).strftime("%Y-%m-%d")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--as-of", default=None, help="End date YYYY-MM-DD (with --look-back); alternative to --start/--end")
    p.add_argument("--look-back", type=int, default=3, help="Days before --as-of to start (default 3)")
    p.add_argument("--start", default=None, help="Window start YYYY-MM-DD")
    p.add_argument("--end", default=None, help="Window end YYYY-MM-DD")
    p.add_argument("--ticker", default=None, help="Audit a single ticker instead of market-wide")
    p.add_argument("--max-articles", type=int, default=50000, help="Market-wide fetch cap (cap-hit = likely truncated)")
    p.add_argument("--per-ticker-cap", type=int, default=PER_TICKER_CAP, help="Flag tickers above this many articles")
    p.add_argument("--top", type=int, default=25, help="How many top tickers/publishers to print")
    args = p.parse_args()

    if args.start and args.end:
        start, end = args.start, args.end
    elif args.as_of:
        start, end = _days_before(args.as_of, args.look_back), args.as_of
    else:
        p.error("provide either --start and --end, or --as-of")

    # No-op when MASSIVE_API_KEY is already in the env (local .env / shell);
    # otherwise pulls from Secret Manager and fails loudly if unavailable.
    from tradingagents.dataflows.secrets import load_secrets_to_env

    load_secrets_to_env()

    print(f"Auditing Massive news: {start} .. {end}" + (f" ticker={args.ticker}" if args.ticker else " (market-wide)"))
    articles = massive.fetch_news_articles(start, end, ticker=args.ticker, max_articles=args.max_articles)

    total = len(articles)
    cap_hit = total >= args.max_articles
    print(f"\ntotal articles fetched: {total}" + ("  *** CAP HIT — feed likely truncated, raise --max-articles ***" if cap_hit else ""))

    if not articles:
        return 0

    published = sorted(a["published_utc"] for a in articles if a["published_utc"])
    if published:
        print(f"published_utc range: {published[0]} .. {published[-1]}")

    with_tickers = sum(1 for a in articles if a["tickers"])
    print(f"articles with >=1 ticker: {with_tickers}  ({with_tickers / total:.0%})  "
          f"(ticker-less articles are dropped by the co-mention builder)")

    ticker_counts: Counter[str] = Counter()
    for a in articles:
        for t in a["tickers"]:
            ticker_counts[t] += 1

    publisher_counts: Counter[str] = Counter(a["publisher"] or "(unknown)" for a in articles)
    has_desc = sum(1 for a in articles if a["description"])
    has_insights = sum(1 for a in articles if a["insights"])
    print(f"\nfield availability:  description={has_desc}/{total} ({has_desc / total:.0%})  "
          f"vendor insights={has_insights}/{total} ({has_insights / total:.0%})")
    print("note: Massive returns title + description (a summary), NOT full article body.")

    print(f"\ntop {args.top} publishers:")
    for name, n in publisher_counts.most_common(args.top):
        print(f"  {n:6d}  {name}")

    print(f"\ntop {args.top} tickers by article count:")
    for t, n in ticker_counts.most_common(args.top):
        flag = "  <-- exceeds per-ticker cap, get_stock_news would truncate" if n > args.per_ticker_cap else ""
        print(f"  {n:6d}  {t}{flag}")

    truncated = [t for t, n in ticker_counts.items() if n > args.per_ticker_cap]
    print(f"\ntickers exceeding per-ticker cap ({args.per_ticker_cap}): {len(truncated)}")
    if truncated:
        print("  " + ", ".join(sorted(truncated)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
