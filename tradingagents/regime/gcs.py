"""Upload regime-gate reports to GCS for durable storage.

Mirrors the local layout to ``gs://{bucket}/{prefix}/{as_of_date}/regime_report.json``.
google-cloud-storage is imported lazily; auth is ADC (same as BigQuery).
"""

from __future__ import annotations

from pathlib import Path

from .store import CATALYSTS_FILE, DEFAULT_OUT_DIR, EVENTS_FILE, REPORT_FILE, SCORECARD_FILE


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


def upload_events(
    session: str,
    bucket: str,
    prefix: str = "event_corpus",
    out_dir: str = DEFAULT_OUT_DIR,
    project: str | None = None,
) -> str:
    """Upload the news ``events.jsonl`` for ``session`` to GCS."""
    return _upload(session, EVENTS_FILE, bucket, prefix, out_dir, project)


def upload_catalysts(
    session: str,
    bucket: str,
    prefix: str = "event_corpus",
    out_dir: str = DEFAULT_OUT_DIR,
    project: str | None = None,
) -> str:
    """Upload the structured ``catalysts.jsonl`` for ``session`` (a date) to GCS."""
    return _upload(session, CATALYSTS_FILE, bucket, prefix, out_dir, project)


def download_report(
    session: str,
    bucket: str,
    prefix: str = "regime_gate",
    out_dir: str = DEFAULT_OUT_DIR,
    project: str | None = None,
) -> Path:
    """Download a report from GCS into the local layout. Returns the local path.

    Reports live on GCS; this materializes one locally so the evaluator can score
    it. Fails loudly if the report is absent on GCS.
    """
    from google.cloud import storage

    local = Path(out_dir) / session / REPORT_FILE
    local.parent.mkdir(parents=True, exist_ok=True)
    blob_path = f"{prefix}/{session}/{REPORT_FILE}" if prefix else f"{session}/{REPORT_FILE}"
    blob = storage.Client(project=project).bucket(bucket).blob(blob_path)
    if not blob.exists():
        raise FileNotFoundError(f"regime report missing on GCS: gs://{bucket}/{blob_path}")
    blob.download_to_filename(str(local))
    return local
