import random
import time
from typing import Any, Optional

import httpx
from google.genai import errors as genai_errors
from langchain_google_genai import ChatGoogleGenerativeAI

from .base_client import BaseLLMClient, normalize_content
from .validators import validate_model

# Transient errors worth retrying: the Gemini endpoint occasionally drops the
# connection mid-request ("Server disconnected without sending a response") or
# returns 5xx under load. google.genai's built-in retry does not cover these.
_TRANSIENT_LLM_ERRORS = (
    httpx.RemoteProtocolError,
    httpx.ConnectError,
    httpx.ConnectTimeout,
    httpx.ReadError,
    httpx.ReadTimeout,
    httpx.WriteError,
    httpx.PoolTimeout,
    genai_errors.ServerError,
)
_LLM_MAX_RETRIES = 5
_LLM_BACKOFF = 1.0
_LLM_MAX_BACKOFF = 30.0


class NormalizedChatGoogleGenerativeAI(ChatGoogleGenerativeAI):
    """ChatGoogleGenerativeAI with normalized content output.

    Gemini 3 models return content as list of typed blocks.
    This normalizes to string for consistent downstream handling.

    Also retries transient connection/5xx failures with exponential backoff;
    non-transient errors (bad request, schema) propagate immediately.
    """

    def invoke(self, input, config=None, **kwargs):
        for attempt in range(_LLM_MAX_RETRIES + 1):
            try:
                result = super().invoke(input, config, **kwargs)
                break
            except _TRANSIENT_LLM_ERRORS:
                if attempt >= _LLM_MAX_RETRIES:
                    raise
                delay = min(_LLM_BACKOFF * (2 ** attempt), _LLM_MAX_BACKOFF)
                time.sleep(delay + random.uniform(0, delay * 0.25))
        return normalize_content(result)


class GoogleClient(BaseLLMClient):
    """Client for Google Gemini models."""

    def __init__(self, model: str, base_url: Optional[str] = None, **kwargs):
        super().__init__(model, base_url, **kwargs)

    def get_llm(self) -> Any:
        """Return configured ChatGoogleGenerativeAI instance."""
        self.warn_if_unknown_model()
        llm_kwargs = {"model": self.model}

        if self.base_url:
            llm_kwargs["base_url"] = self.base_url

        for key in ("timeout", "max_retries", "callbacks", "http_client", "http_async_client"):
            if key in self.kwargs:
                llm_kwargs[key] = self.kwargs[key]

        # Unified api_key maps to provider-specific google_api_key
        google_api_key = self.kwargs.get("api_key") or self.kwargs.get("google_api_key")
        if google_api_key:
            llm_kwargs["google_api_key"] = google_api_key

        # Map thinking_level to appropriate API param based on model
        # Gemini 3 Pro: low, high
        # Gemini 3 Flash: minimal, low, medium, high
        # Gemini 2.5: thinking_budget (0=disable, -1=dynamic)
        thinking_level = self.kwargs.get("thinking_level")
        if thinking_level:
            model_lower = self.model.lower()
            if "gemini-3" in model_lower:
                # Gemini 3 Pro doesn't support "minimal", use "low" instead
                if "pro" in model_lower and thinking_level == "minimal":
                    thinking_level = "low"
                llm_kwargs["thinking_level"] = thinking_level
            else:
                # Gemini 2.5: map to thinking_budget
                llm_kwargs["thinking_budget"] = -1 if thinking_level == "high" else 0

        return NormalizedChatGoogleGenerativeAI(**llm_kwargs)

    def validate_model(self) -> bool:
        """Validate model for Google."""
        return validate_model("google", self.model)
