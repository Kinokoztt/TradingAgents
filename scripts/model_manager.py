"""Manage self-hosted model files: list, download, and serve.

The catalog of 2x3090-deployable models lives in
``tradingagents/llm_clients/local_models.py``. This CLI downloads the weights
to a local store and launches vLLM for a chosen model. The served name is the
local directory name and the ``--served-model-name`` vLLM exposes, so the app
references the same name via ``--model`` / ``TRADINGAGENTS_*_LLM``.

Download store: ``$TRADINGAGENTS_MODELS_DIR`` (default ``~/models``).

Examples:
    python scripts/model_manager.py list
    python scripts/model_manager.py download qwen3-32b
    python scripts/model_manager.py downloaded
    python scripts/model_manager.py serve qwen3-32b            # prints the command
    python scripts/model_manager.py serve qwen3-32b --exec      # downloads if needed + runs vLLM
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tradingagents.llm_clients import hf_resolver
from tradingagents.llm_clients.local_models import LocalModelSpec, get_local_model, list_local_models

_SERVE_SCRIPT = Path(__file__).resolve().parent / "serve_vllm.sh"


def _repo_for(spec: LocalModelSpec, latest: bool) -> str:
    """Repo id to fetch: live-resolved newest match when ``latest``, else pinned."""
    if not latest:
        return spec.hf_repo
    found = hf_resolver.resolve_latest(spec)
    if found is None:
        print(f"warning: live search found no match for '{spec.served_name}'; using pinned {spec.hf_repo}")
        return spec.hf_repo
    print(f"resolved latest for '{spec.served_name}': {found.repo_id} "
          f"(downloads={found.downloads}, likes={found.likes}, modified={found.last_modified})")
    return found.repo_id


def models_dir() -> Path:
    return Path(os.environ.get("TRADINGAGENTS_MODELS_DIR", str(Path.home() / "models"))).expanduser()


def local_path(spec: LocalModelSpec) -> Path:
    return models_dir() / spec.served_name


def is_downloaded(spec: LocalModelSpec) -> bool:
    p = local_path(spec)
    return p.is_dir() and any(p.iterdir())


def cmd_list(_args) -> int:
    print(f"download store: {models_dir()}\n")
    header = f"{'served-name':<24} {'params':<26} {'quant':<28} {'~VRAM':>6} {'ctx':>8} {'dl':>3}  repo"
    print(header)
    print("-" * len(header))
    for spec in list_local_models():
        mark = "yes" if is_downloaded(spec) else "-"
        print(f"{spec.served_name:<24} {spec.params:<26} {spec.quant:<28} "
              f"{spec.approx_vram_gb:>5.0f}G {spec.max_model_len:>8} {mark:>3}  {spec.hf_repo}")
    print("\nnote: ~VRAM > ~44G means you must use an INT4/AWQ build of that repo to fit 2x3090.")
    return 0


def cmd_downloaded(_args) -> int:
    present = [s for s in list_local_models() if is_downloaded(s)]
    if not present:
        print(f"no models downloaded under {models_dir()}")
        return 0
    for spec in present:
        print(f"{spec.served_name:<24} {local_path(spec)}")
    return 0


def cmd_path(args) -> int:
    spec = get_local_model(args.name)
    print(local_path(spec) if is_downloaded(spec) else f"(not downloaded) {spec.hf_repo}")
    return 0


def _download(spec: LocalModelSpec, repo_id: str | None = None) -> Path:
    from huggingface_hub import snapshot_download

    repo_id = repo_id or spec.hf_repo
    dest = local_path(spec)
    dest.mkdir(parents=True, exist_ok=True)
    print(f"downloading {repo_id} -> {dest}")
    snapshot_download(repo_id=repo_id, local_dir=str(dest))
    return dest


def cmd_download(args) -> int:
    spec = get_local_model(args.name)
    dest = _download(spec, _repo_for(spec, args.latest))
    print(f"done: {dest}")
    return 0


def cmd_resolve(args) -> int:
    spec = get_local_model(args.name)
    ranked = hf_resolver.resolve_candidates(spec, limit=args.limit)
    print(f"query='{hf_resolver.effective_query(spec)}'  "
          f"match={hf_resolver.effective_match_terms(spec)}  prefer={hf_resolver.effective_prefer_terms(spec)}")
    if not ranked:
        print("no matching repos found on Hugging Face")
        return 0
    print(f"\ntop {min(args.top, len(ranked))} candidates (best first):")
    for c in ranked[: args.top]:
        print(f"  {c.repo_id:<55} downloads={c.downloads:<10} likes={c.likes:<6} modified={c.last_modified}")
    print(f"\nchosen: {ranked[0].repo_id}  (pinned fallback: {spec.hf_repo})")
    return 0


def cmd_discover(args) -> int:
    cands = hf_resolver.discover(sort=args.sort, task=args.task, limit=args.limit, query=args.query)
    print(f"top {len(cands)} '{args.task}' models by {args.sort}"
          + (f" matching '{args.query}'" if args.query else "") + ":")
    for c in cands:
        print(f"  {c.repo_id:<55} downloads={c.downloads:<10} likes={c.likes:<6} modified={c.last_modified}")
    return 0


def cmd_serve(args) -> int:
    spec = get_local_model(args.name)
    model_ref = str(local_path(spec)) if is_downloaded(spec) else _repo_for(spec, args.latest)

    env = dict(os.environ)
    env.update(
        MODEL=model_ref,
        SERVED_MODEL_NAME=spec.served_name,
        TP_SIZE=str(spec.tp_size),
        MAX_MODEL_LEN=str(args.max_model_len or spec.max_model_len),
        PORT=str(args.port),
    )

    pretty = (f"MODEL={model_ref} SERVED_MODEL_NAME={spec.served_name} "
              f"TP_SIZE={spec.tp_size} MAX_MODEL_LEN={env['MAX_MODEL_LEN']} PORT={args.port} "
              f"{_SERVE_SCRIPT}")
    if not args.exec:
        print("serve command (run with --exec to launch):\n  " + pretty)
        if not is_downloaded(spec):
            print(f"\nnote: '{spec.served_name}' is not downloaded; --exec will fetch it first.")
        return 0

    if not is_downloaded(spec):
        _download(spec, _repo_for(spec, args.latest))
        env["MODEL"] = str(local_path(spec))
    print("launching: " + pretty)
    return subprocess.call(["bash", str(_SERVE_SCRIPT)], env=env)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="show the catalog + which are downloaded").set_defaults(func=cmd_list)
    sub.add_parser("downloaded", help="list locally downloaded models").set_defaults(func=cmd_downloaded)

    pp = sub.add_parser("path", help="print local path (or repo if not downloaded)")
    pp.add_argument("name")
    pp.set_defaults(func=cmd_path)

    pd = sub.add_parser("download", help="download a model's weights to the store")
    pd.add_argument("name")
    pd.add_argument("--latest", action="store_true", help="live-resolve the newest matching HF repo first")
    pd.set_defaults(func=cmd_download)

    pr = sub.add_parser("resolve", help="live-search HF for the newest matching repo (no download)")
    pr.add_argument("name")
    pr.add_argument("--limit", type=int, default=50, help="how many HF results to scan")
    pr.add_argument("--top", type=int, default=10, help="how many candidates to print")
    pr.set_defaults(func=cmd_resolve)

    pdisc = sub.add_parser("discover", help="browse trending/popular HF models")
    pdisc.add_argument("--sort", default="downloads", choices=["downloads", "trending", "likes", "modified", "created"])
    pdisc.add_argument("--task", default="text-generation")
    pdisc.add_argument("--query", default="", help="optional search filter")
    pdisc.add_argument("--limit", type=int, default=20)
    pdisc.set_defaults(func=cmd_discover)

    ps = sub.add_parser("serve", help="print/launch the vLLM serve command for a model")
    ps.add_argument("name")
    ps.add_argument("--port", type=int, default=8000)
    ps.add_argument("--max-model-len", type=int, default=None, help="override the catalog default")
    ps.add_argument("--latest", action="store_true", help="live-resolve the newest matching HF repo if not downloaded")
    ps.add_argument("--exec", action="store_true", help="actually download (if needed) and launch vLLM")
    ps.set_defaults(func=cmd_serve)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
