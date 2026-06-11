"""Backfill (roll back) the concept graph over a historical date range.

Drives the same build -> detect -> (name) -> persist pipeline as
``rebuild_concept_graph.py``, but loops over a trading-day calendar so you can
materialise a whole history of snapshots (e.g. 2024-01-01 -> today) in one run.

Leak-free by construction
--------------------------
For each trading session ``T`` we build the graph from the **prior** session's
close (``data_date = T-1``) and store the snapshot under label ``T`` — exactly
the convention ``run_regime_gate.py`` reads, so a regime-gate replay of session
``T`` sees a concept graph that only knows data up to ``T-1``'s close. Prices,
splits and news are all right-bounded by ``data_date`` inside the service.

The ONE caveat: ``--all`` uses ``load_candidate_universe()``, which is a CURRENT
liquidity snapshot, not point-in-time. That introduces survivorship / liquidity
bias on *which nodes appear* (not on edge values). For a strict point-in-time
universe, pass your own per-period ``--tickers`` instead of ``--all``.

Resumable: progress is appended (fsync'd) to ``<out-dir>/rollback_progress.jsonl``
after every built session, and snapshots that already exist on disk are skipped.
So if the run is interrupted, just launch the same command again — it picks up
exactly where it stopped. ``--overwrite`` forces a full rebuild.

Examples:
    # weekly snapshots, full universe, 2024-01-01 -> today, upload to GCS
    python scripts/rollback_concept_graph.py --start 2024-01-01 --all \
        --stride 5 --gcs-bucket trading_agent

    # daily snapshots over a tech subset (cheap test)
    python scripts/rollback_concept_graph.py --start 2024-01-01 --end 2024-03-01 \
        --stride 1 --tickers AAPL,MSFT,NVDA,AMD,GOOGL,META,AMZN,JPM,XOM,LLY
"""

from __future__ import annotations

# Silence gRPC / absl INFO+WARNING noise from google clients on macOS. Must be
# set before any google.cloud import (which happens lazily on first call).
import os

os.environ.setdefault("GRPC_VERBOSITY", "ERROR")
os.environ.setdefault("GLOG_minloglevel", "2")

import argparse
import datetime as _dt
import json
import sys
import time
import zoneinfo
from datetime import datetime, timedelta
from pathlib import Path

# Allow running as a plain script (python scripts/rollback_concept_graph.py).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tradingagents.concept_graph import CommunityConfig, GraphConfig
from tradingagents.concept_graph.service import build_detect_save, name_and_save_clusters
from tradingagents.concept_graph.store import DEFAULT_OUT_DIR
from tradingagents.dataflows.secrets import load_secrets_to_env

_SNAPSHOT_FILES = ("edges.json", "memberships.json", "clusters.json")
_PROGRESS_FILE = "rollback_progress.jsonl"


def _days_before(date_str: str, days: int) -> str:
    return (datetime.fromisoformat(date_str[:10]) - timedelta(days=days)).strftime("%Y-%m-%d")


def _snapshot_exists(out_dir: str, session: str) -> bool:
    d = Path(out_dir) / session
    return all((d / name).exists() for name in _SNAPSHOT_FILES)


def _load_progress(path: Path) -> set[str]:
    """Sessions already recorded as built (one JSON object per line)."""
    if not path.exists():
        return set()
    done: set[str] = set()
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            done.add(json.loads(line)["session"])
    return done


def _append_progress(path: Path, record: dict) -> None:
    """Append one progress record and fsync so a kill mid-run keeps it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _fmt_eta(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}h{m:02d}m" if h else f"{m:d}m{s:02d}s"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--start", default="2024-01-01", help="First session (YYYY-MM-DD), inclusive")
    p.add_argument("--end", default="latest", help="Last session (YYYY-MM-DD) or 'latest' (= today ET), inclusive")
    p.add_argument("--stride", type=int, default=5,
                   help="Take every Nth trading day: 1=daily, 5=~weekly, 21=~monthly (default 5)")
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--tickers", help="Comma-separated universe subset (point-in-time safe)")
    grp.add_argument("--all", action="store_true",
                     help="Full candidate universe (WARNING: current snapshot, not point-in-time)")
    p.add_argument("--market", default="US")
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--progress-file", default=None,
                   help=f"Progress log (JSONL). Default: <out-dir>/{_PROGRESS_FILE}")
    p.add_argument("--overwrite", action="store_true", help="Rebuild snapshots that already exist")

    # graph (M1) knobs
    p.add_argument("--comovement-window", type=int, default=120)
    p.add_argument("--prune-top-k", type=int, default=20)

    # community (M2) knobs
    p.add_argument("--min-cluster-size", type=int, default=4)
    p.add_argument("--theme-min", type=int, default=40)
    p.add_argument("--theme-max", type=int, default=80)
    p.add_argument("--tau", type=float, default=0.15, help="multi-membership affinity threshold")
    p.add_argument("--k", type=int, default=3, help="max memberships per ticker")

    # naming (G5)
    p.add_argument("--name", action="store_true", help="Run LLM cluster naming after each detection")
    p.add_argument("--naming-model", default=None)

    # durable storage (GCS)
    p.add_argument("--gcs-bucket", default=None, help="If set, upload each snapshot to this bucket")
    p.add_argument("--gcs-prefix", default="concept_graph")

    args = p.parse_args()

    load_secrets_to_env()

    from tradingagents.market_tools import get_market_tools

    tools = get_market_tools(args.market)

    start = args.start
    end = args.end
    if end == "latest":
        end = _dt.datetime.now(zoneinfo.ZoneInfo("America/New_York")).strftime("%Y-%m-%d")

    gcfg = GraphConfig(comovement_window=args.comovement_window, prune_top_k=args.prune_top_k)
    ccfg = CommunityConfig(
        min_cluster_size=args.min_cluster_size,
        theme_target_min=args.theme_min,
        theme_target_max=args.theme_max,
        multi_membership_tau=args.tau,
        multi_membership_k=args.k,
    )
    universe = None if args.all else [t.strip().upper() for t in args.tickers.split(",") if t.strip()]

    if args.all:
        print("WARNING: --all uses the CURRENT candidate universe (not point-in-time): "
              "survivorship/liquidity bias on node membership. Edge values stay leak-free.")

    # Trading-day calendar from the market proxy (SPY trades every session). Pad
    # the left edge so the first in-range session has a prior trading day for its
    # leak-free data_date (= the session before it).
    cal_df = tools.load_daily_close([gcfg.market_ticker], _days_before(start, 12), end)
    cal = sorted(cal_df["trade_date"].dt.strftime("%Y-%m-%d").unique().tolist())
    if not cal:
        raise ValueError(f"no trading days for {gcfg.market_ticker} in {start}..{end}")
    prev_of = {cal[i]: cal[i - 1] for i in range(1, len(cal))}

    sessions = [d for d in cal if start <= d <= end]
    selected = sessions[:: max(args.stride, 1)]
    if not selected:
        raise ValueError(f"no sessions selected in {start}..{end} (stride={args.stride})")

    progress_path = Path(args.progress_file) if args.progress_file else Path(args.out_dir) / _PROGRESS_FILE
    # A session counts as done if its snapshot exists OR it's in the progress log.
    logged = _load_progress(progress_path)
    done = {s for s in selected if s in logged or _snapshot_exists(args.out_dir, s)}
    pending = [s for s in selected if args.overwrite or s not in done]

    print(f"rollback: {len(selected)} sessions (stride={args.stride}) "
          f"from {selected[0]} to {selected[-1]}; out_dir={args.out_dir}")
    print(f"progress: {len(done)} already done, {len(pending)} to build "
          f"(log: {progress_path})")

    built = skipped = 0
    durations: list[float] = []
    for i, session in enumerate(pending, 1):
        if session not in prev_of:
            # first calendar element has no predecessor; skip rather than leak same-day data
            print(f"[{i}/{len(pending)}] {session}: no prior trading day in window, skip")
            continue
        if not args.overwrite and _snapshot_exists(args.out_dir, session):
            # built by a concurrent/earlier run that didn't log; record and skip.
            skipped += 1
            print(f"[{i}/{len(pending)}] {session}: snapshot exists, skip")
            continue

        data_date = prev_of[session]
        t0 = time.monotonic()
        edges, g, memberships, clusters = build_detect_save(
            data_date, universe=universe, config=gcfg, community_config=ccfg,
            market=args.market, out_dir=args.out_dir, label_date=session,
        )

        if args.name:
            name_and_save_clusters(
                session, out_dir=args.out_dir,
                **({"model": args.naming_model} if args.naming_model else {}),
            )

        if args.gcs_bucket:
            from tradingagents.concept_graph.gcs import upload_snapshot

            upload_snapshot(session, args.gcs_bucket, args.gcs_prefix, out_dir=args.out_dir)

        elapsed = time.monotonic() - t0
        durations.append(elapsed)
        built += 1
        _append_progress(progress_path, {
            "session": session,
            "data_date": data_date,
            "nodes": g.number_of_nodes(),
            "edges": g.number_of_edges(),
            "clusters": len(clusters),
            "elapsed_s": round(elapsed, 1),
            "ts": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        })

        avg = sum(durations) / len(durations)
        eta = avg * (len(pending) - i)
        print(f"[{i}/{len(pending)}] {session} (data={data_date}): "
              f"nodes={g.number_of_nodes()} edges={g.number_of_edges()} clusters={len(clusters)} "
              f"| {elapsed:.0f}s  ETA {_fmt_eta(eta)}")

    print(f"\ndone: built={built} skipped={skipped} pending_was={len(pending)} total={len(selected)}")
    print(f"progress log: {progress_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
