"""Backfill standardized news-event extraction over a date range.

Loops trading sessions in [--start, --end] (derived from the proxy's SPY bars,
so weekends/holidays are skipped) and shells out to the tested
scripts/extract_events.py once per session. Each session is independently
resumable (extract_events skips tickers already in its progress file), so a
re-run of the whole range is a near no-op for sessions already finished.

The vLLM service is auto-started by the first session and reused by the rest
(extract_events' --auto-serve detects the running model). Pass --stop-after-task
to shut it down once the whole range completes (not per session).

Examples:
    # dry-run: list the sessions that would be processed
    python scripts/backfill_events.py --start 2026-05-01 --end 2026-05-31 --dry-run

    # backfill May with qwen3-32b, keep going on a bad session, stop vLLM at the end
    python scripts/backfill_events.py --start 2026-05-01 --end 2026-05-31 \
        --model qwen3-32b --continue-on-error --stop-after-task
"""

from __future__ import annotations

import os

os.environ.setdefault("GRPC_VERBOSITY", "ERROR")
os.environ.setdefault("GLOG_minloglevel", "2")

import argparse
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tradingagents.dataflows.secrets import load_secrets_to_env
from tradingagents.llm_clients import config
from tradingagents.market_tools import get_market_tools
from tradingagents.regime.store import DEFAULT_OUT_DIR

_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "extract_events.py")


def _trading_days(tools, proxy: str, start: str, end: str) -> list[str]:
    """Sessions in [start, end] where the proxy printed a bar (the trading calendar)."""
    df = tools.load_daily_ohlc([proxy], start, end)
    if df.empty:
        raise ValueError(f"no {proxy} bars in {start}..{end}; check the date range / proxy / BQ data")
    return [d.strftime("%Y-%m-%d") for d in sorted(df["trade_date"].unique())]


def _extract_cmd(session: str, args) -> list[str]:
    cmd = [
        sys.executable, _SCRIPT, "--as-of", session,
        "--market", args.market, "--out-dir", args.out_dir,
        "--provider", args.provider, "--model", args.model, "--port", str(args.port),
        "--news-look-back", str(args.news_look_back),
        "--max-articles-per-ticker", str(args.max_articles_per_ticker),
        "--max-workers", str(args.max_workers),
    ]
    if args.backend_url:
        cmd += ["--backend-url", args.backend_url]
    if args.max_news_tickers is not None:
        cmd += ["--max-news-tickers", str(args.max_news_tickers)]
    if args.no_price_in:
        cmd.append("--no-price-in")
    if args.no_resume:
        cmd.append("--no-resume")
    # vLLM lifecycle is managed once for the whole range, not per session.
    cmd.append("--no-auto-serve" if args.backend_url else "--auto-serve")
    return cmd


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--start", required=True, help="First session YYYY-MM-DD (inclusive)")
    p.add_argument("--end", required=True, help="Last session YYYY-MM-DD (inclusive)")
    p.add_argument("--market", default="US")
    p.add_argument("--proxy", default="SPY", help="Ticker whose bars define the trading calendar")
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR)

    p.add_argument("--provider", default="vllm")
    p.add_argument("--model", default="qwen3-32b")
    p.add_argument("--backend-url", default=None, help="Use a hosted API instead of auto-serving vLLM")
    p.add_argument("--port", type=int, default=config.default_port())
    p.add_argument("--stop-after-task", action="store_true",
                   help="Stop the auto-started vLLM service after the whole range finishes")

    p.add_argument("--news-look-back", type=int, default=7)
    p.add_argument("--max-news-tickers", type=int, default=None)
    p.add_argument("--max-articles-per-ticker", type=int, default=50)
    p.add_argument("--max-workers", type=int, default=4)
    p.add_argument("--no-price-in", action="store_true")
    p.add_argument("--no-resume", action="store_true", help="Re-extract every ticker (ignore per-session progress)")

    p.add_argument("--continue-on-error", action="store_true", help="Log and continue instead of aborting on a failed session")
    p.add_argument("--dry-run", action="store_true", help="Only list the sessions that would be processed")
    args = p.parse_args()

    load_secrets_to_env()
    tools = get_market_tools(args.market)
    days = _trading_days(tools, args.proxy, args.start, args.end)
    print(f"{len(days)} trading session(s) in {args.start}..{args.end}: {days}")
    if args.dry_run:
        return 0

    done: list[str] = []
    failed: list[tuple[str, str]] = []
    try:
        for session in days:
            print(f"\n========== {session} ==========")
            try:
                subprocess.run(_extract_cmd(session, args), check=True)
                done.append(session)
            except subprocess.CalledProcessError as e:
                if not args.continue_on_error:
                    raise
                print(f"[{session}] FAILED: exit {e.returncode}")
                failed.append((session, f"exit {e.returncode}"))
    finally:
        if args.stop_after_task and args.provider == "vllm" and not args.backend_url:
            from tradingagents.llm_clients import vllm_service

            vllm_service.stop()
            print("stopped the auto-started vLLM service")

    print(f"\n=== events backfill summary === done={len(done)} failed={len(failed)}")
    if failed:
        for s, msg in failed:
            print(f"  FAILED {s}: {msg}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
