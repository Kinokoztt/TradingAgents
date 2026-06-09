"""Shared BigQuery access for US market tools.

google-cloud-bigquery is imported lazily so importing the package (and unit
testing pure logic) does not require BQ credentials. Auth uses ADC, matching
the Secret Manager setup (see dataflows/secrets.py).
"""

from __future__ import annotations

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
    """Run a parameterized query and return a DataFrame."""
    from google.cloud import bigquery

    client = bigquery.Client(project=project)
    job_config = bigquery.QueryJobConfig(query_parameters=params or [])
    return client.query(sql, job_config=job_config).to_dataframe()
