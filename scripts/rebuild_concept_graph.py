"""Rebuild the concept graph for a date: build -> detect -> (name) -> persist.

A thin, schedulable CLI over the concept_graph service. API keys come from
Secret Manager (dataflows.secrets); BigQuery auth is ADC.

Examples:
    # 22-ticker tech/finance subset, fine clusters, with LLM naming
    python scripts/rebuild_concept_graph.py --as-of 2026-06-05 \
        --tickers AAPL,MSFT,NVDA,AMD,AVGO,MU,INTC,GOOGL,META,AMZN,NFLX,JPM,BAC,GS,XOM,CVX,PFE,LLY,KO,PEP,WMT,COST \
        --min-cluster-size 2 --theme-min 5 --theme-max 10 --name

    # full candidate universe, market-scale targets
    python scripts/rebuild_concept_graph.py --as-of 2026-06-05 --all --name
"""

from __future__ import annotations

# Silence gRPC / absl INFO+WARNING noise from google clients on macOS. Must be
# set before any google.cloud import (which happens lazily on first call).
import os

os.environ.setdefault("GRPC_VERBOSITY", "ERROR")
os.environ.setdefault("GLOG_minloglevel", "2")

import argparse
import sys

# Allow running as a plain script (python scripts/rebuild_concept_graph.py).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tradingagents.concept_graph import CommunityConfig, GraphConfig
from tradingagents.concept_graph.service import build_detect_save, name_and_save_clusters
from tradingagents.concept_graph.store import DEFAULT_OUT_DIR
from tradingagents.dataflows.secrets import load_secrets_to_env


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--as-of", required=True, help="Trading session YYYY-MM-DD (named output), or 'latest' (= today ET). Data uses the prior close.")
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--tickers", help="Comma-separated universe subset")
    grp.add_argument("--all", action="store_true", help="Use the full candidate universe")
    p.add_argument("--market", default="US")
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR)

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
    p.add_argument("--name", action="store_true", help="Run LLM cluster naming after detection")
    p.add_argument("--naming-model", default=None)

    # durable storage (GCS)
    p.add_argument("--gcs-bucket", default=None, help="If set, upload the snapshot to this bucket")
    p.add_argument("--gcs-prefix", default="concept_graph")

    args = p.parse_args()

    load_secrets_to_env()

    from tradingagents.market_tools import get_market_tools

    tools = get_market_tools(args.market)

    # session = the trading day we name the snapshot by; data_date = its prior close.
    session = args.as_of
    if session == "latest":
        import datetime as _dt
        import zoneinfo

        session = _dt.datetime.now(zoneinfo.ZoneInfo("America/New_York")).strftime("%Y-%m-%d")
        print(f"resolved --as-of latest -> session {session} (today ET)")
    data_date = tools.previous_trading_day(session)
    print(f"session={session}  data_date={data_date} (prior close)")

    universe = None if args.all else [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    gcfg = GraphConfig(comovement_window=args.comovement_window, prune_top_k=args.prune_top_k)
    ccfg = CommunityConfig(
        min_cluster_size=args.min_cluster_size,
        theme_target_min=args.theme_min,
        theme_target_max=args.theme_max,
        multi_membership_tau=args.tau,
        multi_membership_k=args.k,
    )

    edges, g, memberships, clusters = build_detect_save(
        data_date, universe=universe, config=gcfg, community_config=ccfg,
        market=args.market, out_dir=args.out_dir, label_date=session,
    )
    print(f"nodes={g.number_of_nodes()} edges={g.number_of_edges()} clusters={len(clusters)}")

    if args.name:
        clusters = name_and_save_clusters(
            session, out_dir=args.out_dir,
            **({"model": args.naming_model} if args.naming_model else {}),
        )
        print("named clusters:")
        for cid, c in clusters.items():
            print(f"  {cid}: {c.label}  (sector={c.parent_sector})  members={c.members}")
    else:
        for cid, c in clusters.items():
            print(f"  {cid} (sector={c.parent_sector}): {c.members}")

    print(f"\nsnapshot written to {args.out_dir}/{session}/")

    if args.gcs_bucket:
        from tradingagents.concept_graph.gcs import upload_snapshot

        uris = upload_snapshot(session, args.gcs_bucket, args.gcs_prefix, out_dir=args.out_dir)
        print("uploaded to GCS:")
        for u in uris:
            print(f"  {u}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
