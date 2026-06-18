"""Per-machine local-LLM config: model store, runtime/log dirs, default port.

Every setting resolves in this order: **explicit env var > config file > built-in
default**. So the config file is just a convenient place to pin per-machine paths,
and a one-off env var still overrides it.

The config file is per-machine and gitignored. Copy the template and edit:

    cp tradingagents/llm_clients/llm_config.example.json \\
       tradingagents/llm_clients/llm_config.json

Point elsewhere with ``$TRADINGAGENTS_LLM_CONFIG`` if you prefer a path outside
the repo (e.g. ``~/.config/tradingagents/llm_config.json``).
"""

from __future__ import annotations

import functools
import json
import os
from pathlib import Path

_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = _DIR / "llm_config.json"


def config_path() -> Path:
    return Path(os.environ.get("TRADINGAGENTS_LLM_CONFIG", str(DEFAULT_CONFIG_PATH))).expanduser()


@functools.lru_cache(maxsize=1)
def _file_config() -> dict:
    """Load the JSON config file once. Empty dict if it doesn't exist."""
    path = config_path()
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def reload() -> None:
    """Drop the cached config file (call after editing it in a long-lived process)."""
    _file_config.cache_clear()


def _resolve(env_var: str, key: str, default):
    """env var (if non-empty) wins, then the config file, then the default."""
    env_val = os.environ.get(env_var)
    if env_val:
        return env_val
    file_val = _file_config().get(key)
    if file_val not in (None, ""):
        return file_val
    return default


def models_dir() -> Path:
    """Where model weights are stored/downloaded (default ``~/models``)."""
    return Path(_resolve("TRADINGAGENTS_MODELS_DIR", "models_dir",
                         str(Path.home() / "models"))).expanduser()


def runtime_dir() -> Path:
    """vLLM service state + default log root (default ``~/.cache/tradingagents/vllm``)."""
    d = Path(_resolve("TRADINGAGENTS_VLLM_RUNTIME_DIR", "runtime_dir",
                      str(Path.home() / ".cache" / "tradingagents" / "vllm"))).expanduser()
    d.mkdir(parents=True, exist_ok=True)
    return d


def log_dir() -> Path:
    """Where background vLLM logs go (default ``<runtime_dir>/logs``)."""
    explicit = _resolve("TRADINGAGENTS_VLLM_LOG_DIR", "log_dir", None)
    d = Path(explicit).expanduser() if explicit else runtime_dir() / "logs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def default_port() -> int:
    """Default vLLM port for serve / auto-serve (default 8000)."""
    return int(_resolve("TRADINGAGENTS_VLLM_PORT", "default_port", 8000))


# Quant/checkpoint formats that won't run (well) per GPU generation, matched as
# case-insensitive substrings of a HF repo id. Used to pick a hardware-adapted
# checkpoint at download time instead of failing at serve time.
_BASE_BLOCKED_QUANT = ("gguf",)  # llama.cpp format, not what we serve via vLLM
_ARCH_BLOCKED_QUANT = {
    # 3090/A100 etc. (sm_80/86): no hardware fp8; NVFP4 is Blackwell-only.
    "ampere": ("fp8", "nvfp4"),
    # 4090/L40 (sm_89): fp8 ok; NVFP4 still Blackwell-only.
    "ada": ("nvfp4",),
    "hopper": ("nvfp4",),
    "blackwell": (),
}


def gpu_arch() -> str:
    """Target GPU generation for checkpoint selection (default ``ampere`` = 3090)."""
    return str(_resolve("TRADINGAGENTS_GPU_ARCH", "gpu_arch", "ampere")).lower()


def blocked_quant_terms() -> tuple[str, ...]:
    """Repo-id substrings to exclude when resolving a checkpoint for this box.

    Override directly with the ``blocked_quant_terms`` config key (or the
    ``TRADINGAGENTS_BLOCKED_QUANT_TERMS`` env, comma-separated); otherwise derived
    from ``gpu_arch``. Empty/absent override falls back to the arch defaults.
    """
    override = _file_config().get("blocked_quant_terms")
    env = os.environ.get("TRADINGAGENTS_BLOCKED_QUANT_TERMS")
    if env:
        terms: tuple[str, ...] = tuple(t.strip() for t in env.split(",") if t.strip())
    elif override:
        terms = tuple(override)
    else:
        terms = _BASE_BLOCKED_QUANT + _ARCH_BLOCKED_QUANT.get(gpu_arch(), ())
    # de-dup, lowercase, preserve order
    return tuple(dict.fromkeys(t.lower() for t in terms))
