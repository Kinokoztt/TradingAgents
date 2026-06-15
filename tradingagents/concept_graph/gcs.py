"""Upload concept-graph snapshots to Google Cloud Storage for durable storage.

Mirrors the local layout to ``gs://{bucket}/{prefix}/{as_of_date}/*.json``.
google-cloud-storage is imported lazily; auth is ADC (same as BigQuery).
"""

from __future__ import annotations

from pathlib import Path

from .store import DEFAULT_OUT_DIR

_SNAPSHOT_FILES = ("edges.json", "memberships.json", "clusters.json")


def upload_snapshot(
    as_of_date: str,
    bucket: str,
    prefix: str = "concept_graph",
    out_dir: str = DEFAULT_OUT_DIR,
    project: str | None = None,
) -> list[str]:
    """Upload the local snapshot for ``as_of_date`` to GCS. Returns gs:// URIs."""
    from google.cloud import storage

    local = Path(out_dir) / as_of_date
    client = storage.Client(project=project)
    bkt = client.bucket(bucket)

    uris: list[str] = []
    for name in _SNAPSHOT_FILES:
        blob_path = f"{prefix}/{as_of_date}/{name}" if prefix else f"{as_of_date}/{name}"
        blob = bkt.blob(blob_path)
        blob.upload_from_filename(str(local / name))
        uris.append(f"gs://{bucket}/{blob_path}")
    return uris


def download_snapshot(
    as_of_date: str,
    bucket: str,
    prefix: str = "concept_graph",
    out_dir: str = DEFAULT_OUT_DIR,
    project: str | None = None,
) -> Path:
    """Download the snapshot for ``as_of_date`` from GCS into the local layout.

    The concept graph lives on GCS; this is the only sanctioned way to materialize
    it locally for the regime gate. Fails loudly if a snapshot file is missing.
    Returns the local snapshot directory.
    """
    from google.cloud import storage

    local = Path(out_dir) / as_of_date
    local.mkdir(parents=True, exist_ok=True)
    client = storage.Client(project=project)
    bkt = client.bucket(bucket)

    for name in _SNAPSHOT_FILES:
        blob_path = f"{prefix}/{as_of_date}/{name}" if prefix else f"{as_of_date}/{name}"
        blob = bkt.blob(blob_path)
        if not blob.exists():
            raise FileNotFoundError(f"concept-graph snapshot missing on GCS: gs://{bucket}/{blob_path}")
        blob.download_to_filename(str(local / name))
    return local
