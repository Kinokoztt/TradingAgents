"""Shared BigQuery access for US market tools.

google-cloud-bigquery is imported lazily so importing the package (and unit
testing pure logic) does not require BQ credentials. Auth uses ADC, matching
the Secret Manager setup (see dataflows/secrets.py).
"""

from __future__ import annotations

import os
from typing import Any

import pandas as pd

PROJECT = "mystockproject-431701"
DATASET = "stock_dataset"

DAY_TABLE = "day_aggs_di"
MINUTE_TABLE = "minute_aggs_di"
MACRO_TABLE = "macro_daily"
UNIVERSE_TABLE = "valid_ticker_v3_pure_cs"


def fq(table: str, project: str = PROJECT, dataset: str = DATASET) -> str:
    """Fully-qualified backtick-quoted table reference."""
    return f"`{project}.{dataset}.{table}`"


def run_query(
    sql: str,
    params: list[Any] | None = None,
    project: str = PROJECT,
) -> pd.DataFrame:
    """Run a parameterized query and return a DataFrame.

    The BigQuery Storage Read API (gRPC) speeds up downloads on Cloud Run, but it
    does not honor a system HTTP proxy, so behind a local TUN/proxy that maps
    domains to fake IPs (e.g. 198.18.x.x) it intermittently drops the gRPC socket
    (``ServiceUnavailable``) or hangs. Set ``BQ_USE_STORAGE_API=0`` to force the
    plain REST path; otherwise we try Storage once and fall back to REST on a
    transient Storage failure (REST goes through the HTTP proxy, so it works).
    """
    from google.api_core.exceptions import ServiceUnavailable
    from google.cloud import bigquery

    use_storage = os.environ.get("BQ_USE_STORAGE_API", "1").lower() not in ("0", "false", "no")
    client = bigquery.Client(project=project)
    job_config = bigquery.QueryJobConfig(query_parameters=params or [])
    result = client.query(sql, job_config=job_config)
    if not use_storage:
        return result.to_dataframe(create_bqstorage_client=False)
    try:
        return result.to_dataframe(create_bqstorage_client=True)
    except ServiceUnavailable:
        # Storage gRPC unreachable (e.g. fake-IP TUN proxy); REST still works.
        return result.to_dataframe(create_bqstorage_client=False)
