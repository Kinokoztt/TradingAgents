"""Shared LLM builder for the regime cascade (L1/L2/L3).

Centralizes client construction so the whole cascade can run on a self-hosted
vLLM model as easily as on a hosted API. For Qwen served by vLLM we disable the
thinking trace (the layer outputs structured JSON, not reasoning) and pin
``max_tokens``/``timeout`` so a single structured call can't hang or run away —
mirroring ``build_event_llms`` in events.py.
"""

from __future__ import annotations

import os


def clip_text(text: str, max_chars: int) -> str:
    """Cap a variable-length prompt block so a layer can't overflow a self-hosted
    model's context window. Keeps the head (most feeds lead with the newest items)
    and marks the cut so the model knows it's truncated."""
    if text and len(text) > max_chars:
        return text[:max_chars] + "\n…(truncated)"
    return text


def build_cascade_llm(
    provider: str,
    model: str,
    base_url: str | None = None,
    *,
    max_tokens: int = 2048,
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
