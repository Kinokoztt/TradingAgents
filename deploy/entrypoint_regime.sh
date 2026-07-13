#!/usr/bin/env bash
# Cloud Run Job entrypoint: run ONLY the regime gate for the current pre-market
# session. The concept graph is now built by a separate job (entrypoint_concept_graph.sh)
# and lives on GCS, so this job downloads that session's snapshot from GCS first
# (--cg-from-gcs) instead of rebuilding it. Schedule this AFTER the concept-graph
# job for the same session.
#
# Cloud Run's filesystem is read-only except /tmp, so local artifacts go to /tmp
# (ephemeral; the durable copy is the GCS upload). stdout/stderr stream to Cloud
# Logging. Cloud Scheduler restricts execution to US weekdays, so no weekend
# guard here. `--as-of latest` resolves to today ET (the session being traded
# pre-open); data still uses the prior close, so there is no look-ahead.
set -euo pipefail

BUCKET="${RG_GCS_BUCKET:-trading_agent}"
CG_DIR="/tmp/concept_graph_output"
RG_DIR="/tmp/regime_gate_output"

echo "[entrypoint:regime] $(date -u) regime gate (session=latest, bucket=$BUCKET)"
python scripts/run_regime_gate.py \
  --as-of latest \
  --cg-from-gcs --cg-out-dir "$CG_DIR" --cg-gcs-prefix concept_graph \
  --out-dir "$RG_DIR" \
  --gcs-bucket "$BUCKET" --gcs-prefix regime_gate

echo "[entrypoint:regime] $(date -u) done"
