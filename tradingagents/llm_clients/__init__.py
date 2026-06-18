from .base_client import BaseLLMClient
from .factory import create_llm_client

# Importing the catalog registers each self-hosted model's capabilities
# (json_schema structured output) so create_llm_client('vllm', <served_name>)
# resolves correctly regardless of who imported what first.
from . import config  # noqa: E402,F401
from . import local_models  # noqa: E402,F401
from .local_models import LOCAL_MODELS, LocalModelSpec, get_local_model, list_local_models

__all__ = [
    "BaseLLMClient",
    "create_llm_client",
    "config",
    "LOCAL_MODELS",
    "LocalModelSpec",
    "get_local_model",
    "list_local_models",
]
