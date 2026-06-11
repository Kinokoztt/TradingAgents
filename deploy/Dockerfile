# Cloud Run Job image for the daily pre-market batch (concept graph + regime gate).
# Distinct from the repo-root Dockerfile (which packages the interactive
# `tradingagents` CLI): this installs the optional `regime` extra (GCP + graph
# deps) and runs the batch entrypoint instead.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app
COPY . /app

# `.[regime]` pulls google-cloud-{bigquery,secret-manager,storage} + networkx/
# leidenalg/python-igraph (manylinux wheels, no compiler needed on slim).
RUN pip install --no-cache-dir ".[regime]"

ENTRYPOINT ["bash", "/app/deploy/entrypoint.sh"]
