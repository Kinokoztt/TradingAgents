"""Recompute price-in labels on existing events.jsonl (no LLM, no news re-fetch).

price_in depends only on prices + the event's polarity, both already stored in
events.jsonl, so changing the price-in logic does NOT require re-running the LLM
extraction. This script loads each session's events, re-runs ``tag_price_in``
(reading daily bars from BigQuery), and OVERWRITES the file in place (save_events
truncates, so no duplication — unlike a plain extract_events re-run, which
appends). Optionally re-uploads the rewritten file to GCS.

Examples:
    python scripts/relabel_price_in.py --out-dir model_data/event_corpus
    python scripts/relabel_price_in.py --start 2026-05-01 --end 2026-05-31
    python scripts/relabel_price_in.py --gcs-bucket trading_agent   # + re-upload
"""

from __future__ import annotations

import os
import sys
from collections import Counter

os.environ.setdefault("GRPC_VERBOSITY", "ERROR")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import re

from tradingagents.dataflows.secrets import load_secrets_to_env
from tradingagents.regime.price_in import tag_price_in
from tradingagents.regime.store import DEFAULT_OUT_DIR, EVENTS_FILE, load_events, save_events

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _sessions(out_dir: str, start: str | None, end: str | None) -> list[str]:
    if not os.path.isdir(out_dir):
        raise SystemExit(f"out-dir not found: {out_dir}")
    days = []
    for name in sorted(os.listdir(out_dir)):
        if not _DATE_RE.match(name):
            continue
        if not os.path.exists(os.path.join(out_dir, name, EVENTS_FILE)):
            continue
        if start and name < start:
            continue
        if end and name > end:
            continue
        days.append(name)
    return days


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--start", default=None, help="First session YYYY-MM-DD (inclusive)")
    p.add_argument("--end", default=None, help="Last session YYYY-MM-DD (inclusive)")
    p.add_argument("--market", default="US")
    p.add_argument("--pre-days", type=int, default=3)
    p.add_argument("--post-days", type=int, default=2)
    p.add_argument("--atr-window", type=int, default=14)
    p.add_argument("--sig-atr", type=float, default=1.0, help="Aligned pre-move (ATR units) for PricedIn")
    p.add_argument("--partial-atr", type=float, default=0.5, help="Aligned pre-move (ATR units) for Partial")
    p.add_argument("--gcs-bucket", default=None, help="If set, re-upload each rewritten events.jsonl")
    p.add_argument("--gcs-prefix", default="event_corpus")
    p.add_argument("--dry-run", action="store_true", help="Recompute + print distribution, do NOT write")
    args = p.parse_args()

    load_secrets_to_env()
    sessions = _sessions(args.out_dir, args.start, args.end)
    if not sessions:
        raise SystemExit(f"no sessions with {EVENTS_FILE} under {args.out_dir}")
    print(f"{len(sessions)} session(s): {sessions[0]} .. {sessions[-1]}")

    before, after = Counter(), Counter()
    total = 0
    for s in sessions:
        events = load_events(s, out_dir=args.out_dir)
        if not events:
            continue
        before.update(e.price_in.value for e in events)
        tag_price_in(
            events, market=args.market, pre_days=args.pre_days, post_days=args.post_days,
            atr_window=args.atr_window, sig_atr=args.sig_atr, partial_atr=args.partial_atr,
        )
        after.update(e.price_in.value for e in events)
        total += len(events)
        if not args.dry_run:
            save_events(s, events, out_dir=args.out_dir)
            if args.gcs_bucket:
                from tradingagents.regime.gcs import upload_events

                upload_events(s, args.gcs_bucket, args.gcs_prefix, out_dir=args.out_dir)
        print(f"  {s}: {len(events)} event(s){' (dry-run)' if args.dry_run else ' rewritten'}", flush=True)

    def _show(title: str, c: Counter) -> None:
        print(f"\n[{title}] (n={total})")
        for k, v in c.most_common():
            print(f"  {k:14} {v:6} {v / total * 100:5.1f}%")

    _show("price_in BEFORE", before)
    _show("price_in AFTER", after)
    print(f"\n{'dry-run, nothing written' if args.dry_run else 'rewritten in place'}"
          f"{' + uploaded to GCS' if (args.gcs_bucket and not args.dry_run) else ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
