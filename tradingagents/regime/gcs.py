"""Upload regime-gate reports to GCS for durable storage.

Mirrors the local layout to ``gs://{bucket}/{prefix}/{as_of_date}/regime_report.json``.
google-cloud-storage is imported lazily; auth is ADC (same as BigQuery).
"""

from __future__ import annotations

from pathlib import Path

from .store import DEFAULT_OUT_DIR, REPORT_FILE


def upload_report(
    as_of_date: str,
    bucket: str,
    prefix: str = "regime_gate",
    out_dir: str = DEFAULT_OUT_DIR,
    project: str | None = None,
) -> str:
    """Upload the local report for ``as_of_date`` to GCS. Returns the gs:// URI."""
    from google.cloud import storage

    local = Path(out_dir) / as_of_date / REPORT_FILE
    client = storage.Client(project=project)
    blob_path = f"{prefix}/{as_of_date}/{REPORT_FILE}" if prefix else f"{as_of_date}/{REPORT_FILE}"
    blob = client.bucket(bucket).blob(blob_path)
    blob.upload_from_filename(str(local))
    return f"gs://{bucket}/{blob_path}"
