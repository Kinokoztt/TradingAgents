#!/usr/bin/env bash
# Cloud Run Job entrypoint: rebuild the concept graph for the current pre-market
# session, then run the regime gate on that snapshot, uploading both to GCS.
#
# Cloud Run's filesystem is read-only except /tmp, so local artifacts go to /tmp
# (they are ephemeral; the durable copy is the GCS upload). stdout/stderr stream
# to Cloud Logging. Cloud Scheduler restricts execution to US weekdays, so no
# weekend guard here. `--as-of latest` resolves to today ET (the session being
# traded pre-open); data still uses the prior close, so there is no look-ahead.
set -euo pipefail

BUCKET="${RG_GCS_BUCKET:-trading_agent}"
CG_DIR="/tmp/concept_graph_output"
RG_DIR="/tmp/regime_gate_output"

echo "[entrypoint] $(date -u) concept-graph rebuild (session=latest, bucket=$BUCKET)"
python scripts/rebuild_concept_graph.py \
  --as-of latest --all --name \
  --out-dir "$CG_DIR" \
  --gcs-bucket "$BUCKET" --gcs-prefix concept_graph

echo "[entrypoint] $(date -u) regime gate"
python scripts/run_regime_gate.py \
  --as-of latest \
  --cg-out-dir "$CG_DIR" --out-dir "$RG_DIR" \
  --gcs-bucket "$BUCKET" --gcs-prefix regime_gate

echo "[entrypoint] $(date -u) done"
