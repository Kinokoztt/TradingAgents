# Cloud Run Job image for the daily pre-market batch. One image, two entrypoints:
#   deploy/entrypoint_concept_graph.sh  -> concept-graph-daily job (builds the graph)
#   deploy/entrypoint_regime.sh         -> regime-daily job (pulls graph from GCS, runs gate)
# Each Cloud Run Job picks its script via --args (see docs/regime-gate-deploy.md).
# Distinct from the repo-root Dockerfile (which packages the interactive
# `tradingagents` CLI): this installs the optional `regime` extra (GCP + graph
# deps) and runs a batch entrypoint instead.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app
COPY . /app

# `.[regime]` pulls google-cloud-{bigquery,secret-manager,storage} + networkx/
# leidenalg/python-igraph (manylinux wheels, no compiler needed on slim).
RUN pip install --no-cache-dir ".[regime]"

# Default to the regime entrypoint; the concept-graph job overrides via --args.
ENTRYPOINT ["bash"]
CMD ["/app/deploy/entrypoint_regime.sh"]
