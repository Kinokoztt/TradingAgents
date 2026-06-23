"""Backfill the regime gate over a date range: run S0->S4 per trading session,
persist locally and (optionally) upload to GCS — one report per session.

By default L1 reads the clean event corpus (events.jsonl/catalysts.jsonl pulled
from GCS per session) and the whole cascade runs on a self-hosted vLLM (Qwen).
The vLLM service is auto-started before the first session and reused for the
rest; pass --stop-after-task to shut it down once the whole range completes.

Robustness (mirrors backfill_events.py): a session that fails on a transient
BQ/network blip or a wedged vLLM is re-run up to --retries times with exponential
backoff; each retry re-ensures the vLLM service (the generation probe restarts a
hung engine). --continue-on-error logs and moves on instead of aborting.

Overwrite semantics: uploading a report overwrites the same GCS blob. So a plain
re-run (WITHOUT --skip-existing) regenerates and overwrites the existing GCS
reports for the range; add --skip-existing only to resume and leave done ones be.

The concept-graph snapshot lives on GCS (never local): each session is pulled
from gs://{cg_bucket}/{cg_prefix}/{session}/ before the gate runs (or rebuilt per
session with --rebuild-graph). Trading days come from the proxy's (SPY) bars in
[--start, --end], so holidays/weekends are skipped.

Examples:
    # dry-run: list the sessions that would be processed
    python scripts/backfill_regime.py --start 2026-05-01 --end 2026-05-31 --dry-run

    # regenerate May on local vLLM + clean events, overwrite GCS, self-heal blips
    python scripts/backfill_regime.py --start 2026-05-01 --end 2026-05-31 \
        --gcs-bucket trading_agent --continue-on-error --stop-after-task

    # legacy path: hosted Gemini + raw vendor news
    python scripts/backfill_regime.py --start 2026-05-01 --end 2026-05-31 \
        --provider google --events-source none --l1-model gemini-3-flash-preview \
        --concept-model gemini-3.1-pro-preview --regime-model gemini-3.1-pro-preview \
        --gcs-bucket trading_agent
"""

from __future__ import annotations

import os

os.environ.setdefault("GRPC_VERBOSITY", "ERROR")
os.environ.setdefault("GLOG_minloglevel", "2")

import argparse
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tradingagents.concept_graph.store import DEFAULT_OUT_DIR as CG_OUT_DIR
from tradingagents.dataflows.secrets import load_secrets_to_env
from tradingagents.llm_clients import config
from tradingagents.market_tools import get_market_tools
from tradingagents.regime import run_regime_gate
from tradingagents.regime.store import DEFAULT_OUT_DIR, REPORT_FILE, save_report


def _trading_days(tools, proxy: str, start: str, end: str) -> list[str]:
    """Sessions in [start, end] where the proxy printed a bar (the trading calendar)."""
    df = tools.load_daily_ohlc([proxy], start, end)
    if df.empty:
        raise ValueError(f"no {proxy} bars in {start}..{end}; check the date range / proxy / BQ data")
    return [d.strftime("%Y-%m-%d") for d in sorted(df["trade_date"].unique())]


def _rebuild_graph(session: str, args) -> None:
    """Shell out to the tested concept-graph CLI for one session (optional path)."""
    cmd = [
        sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "rebuild_concept_graph.py"),
        "--as-of", session, "--all", "--name", "--out-dir", args.cg_out_dir,
    ]
    if args.gcs_bucket:
        cmd += ["--gcs-bucket", args.gcs_bucket, "--gcs-prefix", "concept_graph"]
    subprocess.run(cmd, check=True)


def _ensure_vllm(args) -> str | None:
    """Ensure the vLLM service is up (auto-serve) and return its base URL.

    Re-ensuring is idempotent and self-heals: ``ensure`` probes generation, so a
    wedged engine (answers /models but can't generate) is restarted here.
    """
    if args.backend_url or args.provider != "vllm" or not args.auto_serve:
        return args.backend_url
    from tradingagents.llm_clients import vllm_service

    model = args.model or args.l1_model
    state = vllm_service.ensure(model, port=args.port)
    print(f"vLLM ready at {state.base_url} (pid {state.pid}, log {state.log_file})")
    return state.base_url


def _run_session(session: str, args, base_url: str | None, cg_bucket: str | None):
    """One session's work: pull/rebuild the concept graph, run the cascade,
    persist locally, and upload (overwrite) to GCS if a bucket is set."""
    if args.rebuild_graph:
        _rebuild_graph(session, args)
    else:
        from tradingagents.concept_graph.gcs import download_snapshot

        download_snapshot(session, cg_bucket, args.cg_gcs_prefix, out_dir=args.cg_out_dir)
        print(f"[{session}] concept graph pulled from gs://{cg_bucket}/{args.cg_gcs_prefix}/{session}/")

    events_source = None if args.events_source == "none" else args.events_source
    report = run_regime_gate(
        session, market=args.market, out_dir=args.cg_out_dir,
        provider=args.provider, base_url=base_url,
        model=args.model, l1_model=args.l1_model,
        concept_model=args.concept_model, regime_model=args.regime_model,
        news_look_back_days=args.news_look_back, max_news_tickers=args.max_news_tickers,
        batch_size=args.batch_size, max_workers=args.max_workers,
        with_fundamentals=not args.no_fundamentals, propagate=not args.no_propagate,
        use_llm_concepts=not args.no_llm_concepts,
        events_source=events_source, gcs_bucket=args.gcs_bucket,
        events_prefix=args.events_prefix, events_out_dir=args.out_dir,
        catalyst_look_back_days=args.catalyst_look_back, proxy=args.proxy,
    )
    save_report(session, report, out_dir=args.out_dir)
    if args.gcs_bucket:
        from tradingagents.regime.gcs import upload_report

        uri = upload_report(session, args.gcs_bucket, args.gcs_prefix, out_dir=args.out_dir)
        print(f"[{session}] uploaded (overwrote) {uri}")
    return report


def _is_deterministic(e: BaseException) -> bool:
    """Client/schema errors that fail identically on retry (4xx, context
    overflow, validation). Retrying these only wastes time and hides a real bug,
    so surface them immediately (the design rule: never retry 4xx/schema)."""
    if getattr(e, "status_code", None) == 400:
        return True
    if type(e).__name__ in {"BadRequestError", "UnprocessableEntityError", "ValidationError"}:
        return True
    return "maximum context length" in str(e).lower()


def _run_session_with_retries(session: str, args, base_url: str | None, cg_bucket: str | None):
    """Run one session, re-running on failure up to ``--retries`` times.

    A regime session isn't incrementally resumable, so a retry re-runs it whole
    (and overwrites) — fine for healing transient BQ/network/vLLM blips. Each
    retry first re-ensures vLLM (restarts a wedged engine) and returns the fresh
    base URL. Deterministic 4xx/schema errors are NOT retried (they'd fail the
    same every time); they propagate to the session loop immediately.
    """
    last_url = base_url
    for attempt in range(args.retries + 1):
        try:
            return _run_session(session, args, last_url, cg_bucket)
        except Exception as e:  # noqa: BLE001 — retry loop heals transient failures
            if _is_deterministic(e) or attempt >= args.retries:
                raise
            wait = min(args.retry_wait * (2 ** attempt), 300.0)
            print(f"[{session}] attempt {attempt + 1}/{args.retries + 1} failed "
                  f"({type(e).__name__}: {e}); retrying in {wait:.0f}s ...", flush=True)
            time.sleep(wait)
            last_url = _ensure_vllm(args)  # heal a wedged/crashed engine before retry


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--start", required=True, help="First session YYYY-MM-DD (inclusive)")
    p.add_argument("--end", required=True, help="Last session YYYY-MM-DD (inclusive)")
    p.add_argument("--market", default="US")
    p.add_argument("--proxy", default="SPY", help="Ticker whose bars define the trading calendar")
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--cg-out-dir", default=CG_OUT_DIR, help="Concept graph snapshot dir to read (and write if --rebuild-graph)")

    # LLM provider/endpoint + per-layer models (default: self-hosted vLLM Qwen).
    p.add_argument("--provider", default="vllm", help="LLM provider (vllm, google, openai, ...)")
    p.add_argument("--backend-url", default=None, help="Use a hosted/remote endpoint instead of auto-serving vLLM")
    p.add_argument("--port", type=int, default=config.default_port(), help="vLLM port for --auto-serve")
    p.add_argument("--auto-serve", action=argparse.BooleanOptionalAction, default=True,
                   help="For provider=vllm with no --backend-url: ensure the model's vLLM service is running. Default on.")
    p.add_argument("--stop-after-task", action="store_true",
                   help="Stop the auto-started vLLM service after the WHOLE range finishes (not per session)")
    p.add_argument("--model", default=None, help="If set, use this model for ALL layers")
    p.add_argument("--l1-model", default="qwen3-32b")
    p.add_argument("--concept-model", default="qwen3-32b")
    p.add_argument("--regime-model", default="qwen3-32b")

    # L1 input source: standardized event corpus (clean) vs raw vendor news.
    p.add_argument("--events-source", choices=["gcs", "local", "none"], default="gcs",
                   help="L1 input: 'gcs' pulls events/catalysts from --gcs-bucket; 'local' reads --out-dir; "
                        "'none' uses the legacy raw-news S0 scan")
    p.add_argument("--events-prefix", default="event_corpus", help="GCS prefix for the event corpus")
    p.add_argument("--catalyst-look-back", type=int, default=5,
                   help="Trading-day window for structured catalysts (strictly before the session, age-tagged)")

    # cascade knobs
    p.add_argument("--news-look-back", type=int, default=3)
    p.add_argument("--max-news-tickers", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=20)
    p.add_argument("--max-workers", type=int, default=6)
    p.add_argument("--no-fundamentals", action="store_true")
    p.add_argument("--no-propagate", action="store_true")
    p.add_argument("--no-llm-concepts", action="store_true")

    # storage + backfill control
    p.add_argument("--gcs-bucket", default=None, help="If set, pull events from / upload (overwrite) each report to this bucket")
    p.add_argument("--gcs-prefix", default="regime_gate")
    p.add_argument("--cg-gcs-bucket", default=None, help="Bucket holding concept-graph snapshots (defaults to --gcs-bucket)")
    p.add_argument("--cg-gcs-prefix", default="concept_graph")
    p.add_argument("--rebuild-graph", action="store_true", help="Rebuild the concept graph per session instead of pulling from GCS")
    p.add_argument("--skip-existing", action="store_true",
                   help="Skip (don't overwrite) sessions whose local report already exists — use to resume a range")
    p.add_argument("--continue-on-error", action="store_true", help="Log and continue instead of aborting on a failed session")
    p.add_argument("--retries", type=int, default=5,
                   help="Re-run a failed session up to N more times before giving up (exponential backoff). "
                        "Each retry re-ensures vLLM to heal a wedged engine. 0 disables.")
    p.add_argument("--retry-wait", type=float, default=30.0,
                   help="Base seconds between retries (exponential backoff, capped at 300s)")
    p.add_argument("--dry-run", action="store_true", help="Only list the sessions that would be processed")

    args = p.parse_args()
    cg_bucket = args.cg_gcs_bucket or args.gcs_bucket
    if not args.rebuild_graph and not cg_bucket:
        p.error("concept graph lives on GCS: pass --gcs-bucket (or --cg-gcs-bucket), or use --rebuild-graph")
    if args.events_source == "gcs" and not args.gcs_bucket:
        p.error("--events-source gcs needs --gcs-bucket to pull the event corpus (or use --events-source local/none)")
    load_secrets_to_env()

    tools = get_market_tools(args.market)
    days = _trading_days(tools, args.proxy, args.start, args.end)
    print(f"{len(days)} trading session(s) in {args.start}..{args.end}: {days}")
    if args.dry_run:
        return 0

    base_url = _ensure_vllm(args)  # auto-start vLLM once; reused across the range

    done: list[str] = []
    skipped: list[str] = []
    failed: list[tuple[str, str]] = []
    try:
        for session in days:
            if args.skip_existing and (Path(args.out_dir) / session / REPORT_FILE).exists():
                print(f"[{session}] skip (report exists)")
                skipped.append(session)
                continue

            print(f"\n========== {session} ==========")
            try:
                report = _run_session_with_retries(session, args, base_url, cg_bucket)
                print(f"[{session}] regime={report.market_state.value}  stocks={len(report.stock_signals)}  "
                      f"raw_long={len(report.long_whitelist)}  tradable_long={len(report.tradable_long_whitelist)}")
                done.append(session)
            except Exception as e:  # noqa: BLE001 — backfill loop: optionally survive one bad session
                if not args.continue_on_error:
                    raise
                print(f"[{session}] FAILED after retries: {type(e).__name__}: {e}")
                failed.append((session, f"{type(e).__name__}: {e}"))
    finally:
        if args.stop_after_task and args.provider == "vllm" and args.auto_serve and not args.backend_url:
            from tradingagents.llm_clients import vllm_service

            vllm_service.stop()
            print("stopped the auto-started vLLM service")

    print(f"\n=== backfill summary === done={len(done)} skipped={len(skipped)} failed={len(failed)}")
    if failed:
        for s, msg in failed:
            print(f"  FAILED {s}: {msg}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
