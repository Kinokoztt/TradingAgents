"""Shared LLM builder for the regime cascade (L1/L2/L3).

Centralizes client construction so the whole cascade can run on a self-hosted
vLLM model as easily as on a hosted API. For Qwen served by vLLM we disable the
thinking trace (the layer outputs structured JSON, not reasoning) and pin
``max_tokens``/``timeout`` so a single structured call can't hang or run away —
mirroring ``build_event_llms`` in events.py.
"""

from __future__ import annotations

import os


def build_cascade_llm(
    provider: str,
    model: str,
    base_url: str | None = None,
    *,
    max_tokens: int = 8192,
    timeout: float = 300.0,
):
    """Build the structured-output base LLM for one cascade layer.

    Qwen (self-hosted vLLM) gets thinking disabled + bounded generation; hosted
    providers (e.g. Google Gemini) keep their defaults. ``google_api_key`` is
    forwarded harmlessly — OpenAI-compatible clients ignore the unknown kwarg.
    """
    from tradingagents.llm_clients import create_llm_client

    kwargs: dict = {"google_api_key": os.getenv("GOOGLE_API_KEY")}
    if "qwen" in model.lower():
        kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
        kwargs["max_tokens"] = max_tokens
        kwargs["timeout"] = timeout
    return create_llm_client(provider, model, base_url=base_url, **kwargs).get_llm()
