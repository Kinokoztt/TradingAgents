#!/usr/bin/env bash
# Daily pre-market concept-graph rebuild for the FULL candidate universe,
# then upload the snapshot to GCS for durable storage. Schedule this via
# launchd or cron (see docs). Idempotent: re-running for the same session
# overwrites that day's snapshot.
#
# Configure via env vars (or edit the defaults):
#   CG_GCS_BUCKET   (required) GCS bucket name, e.g. my-trading-bucket
#   CG_GCS_PREFIX   object prefix (default: concept_graph)
#   CG_REPO         repo path
#   CG_PY           python interpreter (the py10 env)
#   CG_LOG_DIR      log directory
set -euo pipefail

REPO="${CG_REPO:-/Users/wangzetian/Projects/TradingAgents}"
PY="${CG_PY:-/opt/anaconda3/envs/py10/bin/python}"
BUCKET="${CG_GCS_BUCKET:?set CG_GCS_BUCKET to your GCS bucket name}"
PREFIX="${CG_GCS_PREFIX:-concept_graph}"
LOG_DIR="${CG_LOG_DIR:-$REPO/logs}"

# ADC is used for BigQuery/GCS; Secret Manager supplies the API keys.
export GOOGLE_CLOUD_PROJECT="${GOOGLE_CLOUD_PROJECT:-mystockproject-431701}"

# Run only on US (Eastern) weekdays. DST is handled by zoneinfo, so the cron
# trigger can stay at a fixed local time year-round. US market holidays are
# absorbed by --as-of latest (it resolves to the last session; a holiday
# re-run just overwrites that snapshot, which is harmless).
ET_DOW="$("$PY" -c "import datetime,zoneinfo;print(datetime.datetime.now(zoneinfo.ZoneInfo('America/New_York')).weekday())")"
if [ "$ET_DOW" -ge 5 ]; then
  echo "[$(date)] weekend in US/Eastern (dow=$ET_DOW); skipping."
  exit 0
fi

mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/concept_graph_$(date +%Y%m%d-%H%M%S).log"

cd "$REPO"
echo "[$(date)] start concept-graph daily rebuild" >> "$LOG"
"$PY" scripts/rebuild_concept_graph.py \
  --as-of latest --all --name \
  --gcs-bucket "$BUCKET" --gcs-prefix "$PREFIX" \
  >> "$LOG" 2>&1
echo "[$(date)] done" >> "$LOG"
