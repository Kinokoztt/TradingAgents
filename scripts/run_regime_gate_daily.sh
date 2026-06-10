#!/usr/bin/env bash
# Daily pre-market regime gate. Rebuilds the concept graph for the last completed
# session, then runs the regime gate on the same snapshot, uploading both to GCS
# (concept_graph/ and regime_gate/ prefixes under the same bucket). Schedule via
# launchd or cron at a pre-market time (e.g. Beijing 20:00 = US pre-open).
#
# Configure via env vars (or edit defaults):
#   RG_GCS_BUCKET   GCS bucket (default: trading_agent)
#   RG_REPO         repo path
#   RG_PY           python interpreter (the py10 env)
#   RG_LOG_DIR      log directory
set -euo pipefail

REPO="${RG_REPO:-/Users/wangzetian/Projects/TradingAgents}"
PY="${RG_PY:-/opt/anaconda3/envs/py10/bin/python}"
BUCKET="${RG_GCS_BUCKET:-trading_agent}"
LOG_DIR="${RG_LOG_DIR:-$REPO/logs}"

export GOOGLE_CLOUD_PROJECT="${GOOGLE_CLOUD_PROJECT:-mystockproject-431701}"

# Run only on US (Eastern) weekdays; DST handled by zoneinfo. At a pre-market
# trigger 'latest' resolves to the prior completed session (today's bars don't
# exist yet pre-open), so there is no look-ahead.
ET_DOW="$("$PY" -c "import datetime,zoneinfo;print(datetime.datetime.now(zoneinfo.ZoneInfo('America/New_York')).weekday())")"
if [ "$ET_DOW" -ge 5 ]; then
  echo "[$(date)] weekend in US/Eastern (dow=$ET_DOW); skipping."
  exit 0
fi

mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/regime_gate_$(date +%Y%m%d-%H%M%S).log"

cd "$REPO"
echo "[$(date)] start daily concept-graph + regime-gate" >> "$LOG"

# 1) Concept graph snapshot for the last completed session.
"$PY" scripts/rebuild_concept_graph.py \
  --as-of latest --all --name \
  --gcs-bucket "$BUCKET" --gcs-prefix concept_graph \
  >> "$LOG" 2>&1

# 2) Regime gate on that snapshot.
"$PY" scripts/run_regime_gate.py \
  --as-of latest \
  --gcs-bucket "$BUCKET" --gcs-prefix regime_gate \
  >> "$LOG" 2>&1

echo "[$(date)] done" >> "$LOG"
