"""Extract line-1 structured catalysts over a date range -> catalysts.jsonl.

Pure structured pipeline (no LLM, no vLLM): pull FMP's earnings, analyst grade
actions, price-target changes, dividends, and M&A for the candidate universe,
derive event_type/polarity/certainty by rule, keep full numeric payloads, and
write one ``{out_dir}/<date>/catalysts.jsonl`` per effective date (mirroring the
news ``events.jsonl`` layout). It joins ``events.jsonl`` on (ticker, session) at
NN feature-assembly time.

Examples:
    python scripts/extract_catalysts.py --start 2026-05-01 --end 2026-05-29
    python scripts/extract_catalysts.py --start 2026-05-01 --end 2026-05-29 --tickers AAPL,NVDA
"""

from __future__ import annotations

import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

os.environ.setdefault("GRPC_VERBOSITY", "ERROR")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
from collections import Counter, defaultdict

from tradingagents.dataflows import fmp
from tradingagents.dataflows.secrets import load_secrets_to_env
from tradingagents.market_tools import get_market_tools
from tradingagents.regime.catalysts import build_ticker_catalysts, mergers_to_catalysts
from tradingagents.regime.store import CATALYSTS_FILE, DEFAULT_OUT_DIR


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--start", required=True, help="First effective date YYYY-MM-DD (inclusive)")
    p.add_argument("--end", required=True, help="Last effective date YYYY-MM-DD (inclusive)")
    p.add_argument("--market", default="US")
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--tickers", default=None, help="Comma-separated override; default = candidate universe")
    p.add_argument("--max-workers", type=int, default=8)
    p.add_argument("--no-mna", action="store_true", help="Skip the market-wide M&A feed")
    p.add_argument("--gcs-bucket", default=None, help="If set, upload each date's catalysts.jsonl to this bucket")
    p.add_argument("--gcs-prefix", default="event_corpus")
    args = p.parse_args()

    load_secrets_to_env()
    tools = get_market_tools(args.market)

    if args.tickers:
        universe = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    else:
        universe = tools.load_candidate_universe()
    print(f"catalysts {args.start} .. {args.end}: {len(universe)} ticker(s)")

    all_cats = []
    processed = 0
    total = len(universe)
    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futures = {pool.submit(build_ticker_catalysts, t, args.start, args.end): t for t in universe}
        for fut in as_completed(futures):
            ticker = futures[fut]
            cats = fut.result()
            all_cats.extend(cats)
            processed += 1
            if cats or processed % 25 == 0:
                print(f"[{processed}/{total}] {ticker:<8} -> {len(cats)} catalyst(s)", flush=True)

    if not args.no_mna:
        print("fetching market-wide M&A ...", flush=True)
        mna_rows = fmp.fetch_mergers(stop_before=args.start)
        mna = mergers_to_catalysts(mna_rows, set(universe), args.start, args.end)
        all_cats.extend(mna)
        print(f"M&A matched to universe: {len(mna)}")

    # Partition by effective_date so the layout mirrors events.jsonl:
    # {out_dir}/{date}/catalysts.jsonl (one dir per date, overwritten each run).
    by_date: dict[str, list] = defaultdict(list)
    for c in all_cats:
        by_date[c.effective_date].append(c)
    for date, cats in sorted(by_date.items()):
        day_dir = os.path.join(args.out_dir, date)
        os.makedirs(day_dir, exist_ok=True)
        with open(os.path.join(day_dir, CATALYSTS_FILE), "w") as f:
            for c in cats:
                f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")

    by_type = Counter(c.catalyst_type.value for c in all_cats)
    by_pol = Counter(c.polarity.value for c in all_cats)
    print(f"\nwrote {len(all_cats)} catalyst(s) across {len(by_date)} date(s) -> {args.out_dir}/<date>/{CATALYSTS_FILE}")
    print("by type:    " + ", ".join(f"{k}={v}" for k, v in by_type.most_common()))
    print("by polarity:" + ", ".join(f" {k}={v}" for k, v in by_pol.most_common()))

    if args.gcs_bucket:
        from tradingagents.regime.gcs import upload_catalysts

        for date in sorted(by_date):
            upload_catalysts(date, args.gcs_bucket, args.gcs_prefix, out_dir=args.out_dir)
        print(f"uploaded {len(by_date)} date(s) -> gs://{args.gcs_bucket}/{args.gcs_prefix}/<date>/{CATALYSTS_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
