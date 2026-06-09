"""GCP Secret Manager access for all external API keys.

Keys are never stored locally: at startup we read them from Secret Manager
and inject them into ``os.environ`` under the env-var names the rest of the
codebase already expects (MASSIVE_API_KEY, FMP_API_KEY, GOOGLE_API_KEY, ...).
This keeps the vendor modules (massive.py, fmp.py) unchanged — they still
read a plain env var — while the actual secret only ever lives in memory.

Auth uses Application Default Credentials (ADC), so ``gcloud auth
application-default login`` (locally) or the runtime service account (Cloud
Run) is all that's required.
"""

from __future__ import annotations

import os

DEFAULT_PROJECT_ID = "mystockproject-431701"

# Secret Manager secret id -> environment variable consumed by the code.
DEFAULT_SECRET_ENV_MAP: dict[str, str] = {
    "massive_key": "MASSIVE_API_KEY",
    "massive_key_md5": "MASSIVE_KEY_MD5",
    "fmp_api_key": "FMP_API_KEY",
    "fred_api_key": "FRED_API_KEY",
    "gemini_api_key": "GOOGLE_API_KEY",
}


def get_secret(
    secret_id: str,
    project_id: str = DEFAULT_PROJECT_ID,
    version: str = "latest",
) -> str:
    """Return the decoded secret payload (trailing whitespace stripped)."""
    from google.cloud import secretmanager

    client = secretmanager.SecretManagerServiceClient()
    name = f"projects/{project_id}/secrets/{secret_id}/versions/{version}"
    response = client.access_secret_version(name=name)
    return response.payload.data.decode("utf-8").strip()


def load_secrets_to_env(
    mapping: dict[str, str] | None = None,
    project_id: str = DEFAULT_PROJECT_ID,
    override: bool = False,
) -> list[str]:
    """Fetch secrets and inject them into ``os.environ``.

    Returns the list of env vars that were set. Existing env vars are left
    untouched unless ``override`` is True, so a caller can pre-set a value
    (e.g. in tests) without hitting Secret Manager.
    """
    mapping = mapping or DEFAULT_SECRET_ENV_MAP
    loaded: list[str] = []
    for secret_id, env_var in mapping.items():
        if not override and os.environ.get(env_var):
            continue
        os.environ[env_var] = get_secret(secret_id, project_id)
        loaded.append(env_var)
    return loaded
