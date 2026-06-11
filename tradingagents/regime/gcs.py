"""Upload regime-gate reports to GCS for durable storage.

Mirrors the local layout to ``gs://{bucket}/{prefix}/{as_of_date}/regime_report.json``.
google-cloud-storage is imported lazily; auth is ADC (same as BigQuery).
"""

from __future__ import annotations

from pathlib import Path

from .store import DEFAULT_OUT_DIR, REPORT_FILE, SCORECARD_FILE


def _upload(session: str, filename: str, bucket: str, prefix: str, out_dir: str, project: str | None) -> str:
    from google.cloud import storage

    local = Path(out_dir) / session / filename
    client = storage.Client(project=project)
    blob_path = f"{prefix}/{session}/{filename}" if prefix else f"{session}/{filename}"
    client.bucket(bucket).blob(blob_path).upload_from_filename(str(local))
    return f"gs://{bucket}/{blob_path}"


def upload_report(
    as_of_date: str,
    bucket: str,
    prefix: str = "regime_gate",
    out_dir: str = DEFAULT_OUT_DIR,
    project: str | None = None,
) -> str:
    """Upload the local report for ``as_of_date`` to GCS. Returns the gs:// URI."""
    return _upload(as_of_date, REPORT_FILE, bucket, prefix, out_dir, project)


def upload_scorecard(
    session: str,
    bucket: str,
    prefix: str = "regime_gate",
    out_dir: str = DEFAULT_OUT_DIR,
    project: str | None = None,
) -> str:
    """Upload the local scorecard for ``session`` to GCS. Returns the gs:// URI."""
    return _upload(session, SCORECARD_FILE, bucket, prefix, out_dir, project)
