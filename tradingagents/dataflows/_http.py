"""Shared HTTP GET with bounded retry + exponential backoff for data vendors.

Only *transient* failures are retried — connection/TLS drops (e.g. Massive's
occasional ``SSLEOFError`` / "server disconnected" under load), timeouts, broken
chunked reads, and 429/5xx — with exponential backoff (capped) plus random
jitter that honors ``Retry-After``. Retries exhausting re-raises, and
non-retryable 4xx fail immediately (``raise_for_status``). So this hardens flaky
networks / rate limits without masking real errors.

The caller passes its own ``getter`` (its module-level ``requests.get``) so auth
headers/params and test monkeypatches keep working unchanged.
"""

from __future__ import annotations

import random
import time
from typing import Callable, Optional

from requests.exceptions import ChunkedEncodingError
from requests.exceptions import ConnectionError as ReqConnectionError
from requests.exceptions import SSLError, Timeout

_RETRY_STATUS = (429, 500, 502, 503, 504)
_TRANSIENT_ERRORS = (SSLError, ReqConnectionError, Timeout, ChunkedEncodingError)


def _backoff_sleep(attempt: int, backoff: float, max_backoff: float) -> None:
    """Exponential backoff capped at ``max_backoff``, with up to 25% jitter."""
    delay = min(backoff * (2 ** attempt), max_backoff)
    time.sleep(delay + random.uniform(0, delay * 0.25))


def get_with_retry(
    getter: Callable,
    url: str,
    *,
    params: Optional[dict] = None,
    headers: Optional[dict] = None,
    timeout: int = 30,
    retries: int = 5,
    backoff: float = 1.0,
    max_backoff: float = 30.0,
):
    """GET ``url`` via ``getter``, retrying transient errors. Returns the response."""
    for attempt in range(retries + 1):
        try:
            resp = getter(url, params=params, headers=headers, timeout=timeout)
        except _TRANSIENT_ERRORS:
            if attempt >= retries:
                raise
            _backoff_sleep(attempt, backoff, max_backoff)
            continue

        if getattr(resp, "status_code", None) in _RETRY_STATUS and attempt < retries:
            retry_after = resp.headers.get("Retry-After") if hasattr(resp, "headers") else None
            if retry_after and str(retry_after).isdigit():
                time.sleep(float(retry_after))
            else:
                _backoff_sleep(attempt, backoff, max_backoff)
            continue

        resp.raise_for_status()
        return resp
    raise RuntimeError("unreachable")  # loop always returns or raises
