"""Curated catalog of self-hostable open-weight models for a 2x RTX 3090 box.

This is the single source of truth for "which models can I download and serve
on 48 GB of VRAM, and how". ``scripts/model_manager.py`` reads it to list /
download / serve; the served name doubles as the local download directory name
and the ``--served-model-name`` exposed by vLLM (so the app's
``TRADINGAGENTS_*_LLM`` / ``--model`` just references the served name).

Sizing reality for 2x RTX 3090 (24 GB x2, PCIe, no NVLink), per the 2026
dual-3090 serving guides:
  - 48 GB does NOT pool: each card holds half the weights (vLLM TP=2). Usable
    budget after activations/KV is ~44 GB.
  - Dense 27-35B at INT4/AWQ, 30B-class MoE, and 70B-dense at INT4 (tight, short
    context) are the sweet spots. fp8 KV cache is what makes long context fit.

Repo ids are starting points — the quant ecosystem moves fast, so verify the
exact repo on Hugging Face (or swap in any compatible AWQ/GPTQ/FP8/MXFP4 build)
before downloading. The downloader fails loudly on a bad id rather than guessing.

All entries use json_schema structured output (vLLM guided decoding is
model-agnostic), registered into the capability table at import.
"""

from __future__ import annotations

from dataclasses import dataclass

from .capabilities import ModelCapabilities, register


@dataclass(frozen=True)
class LocalModelSpec:
    """One self-hostable model: how to fetch it and how to serve it on 2x3090.

    ``hf_repo`` is a pinned fallback. The live resolver (hf_resolver.py) can
    instead search Hugging Face for the newest matching build using
    ``hf_query`` / ``match_terms`` / ``prefer_terms`` (all optional — sensible
    defaults are derived from ``served_name`` + ``quant`` when blank).
    """

    served_name: str          # API model id + local dir name + --served-model-name
    hf_repo: str              # Hugging Face repo id to download (pinned fallback)
    params: str               # human-readable size, e.g. "32B" or "30B-A3B (MoE)"
    quant: str                # "AWQ-INT4" | "GPTQ-INT4" | "MXFP4" | "FP16" | ...
    approx_vram_gb: float      # rough weights footprint across both cards
    tp_size: int = 2          # tensor-parallel shards (both 3090s)
    max_model_len: int = 32768
    license: str = ""
    notes: str = ""
    # Live-resolution hints (optional; defaults derived in hf_resolver.py):
    hf_query: str = ""              # HF search string, e.g. "Qwen3 32B AWQ"
    match_terms: tuple[str, ...] = ()   # repo id must contain ALL of these (case-insensitive)
    prefer_terms: tuple[str, ...] = ()  # rank repos containing these higher (quant prefs)


# Ordered roughly small -> large / by recommendation. Keep this list curated;
# it is meant to be edited as new open-weights land.
LOCAL_MODELS: dict[str, LocalModelSpec] = {
    spec.served_name: spec
    for spec in [
        LocalModelSpec(
            served_name="gpt-oss-20b",
            hf_repo="openai/gpt-oss-20b",
            params="20B (MoE, 3.6B active)",
            quant="MXFP4 (native)",
            approx_vram_gb=13,
            tp_size=2,
            max_model_len=131072,
            license="Apache-2.0",
            notes="OpenAI open-weights reasoning model; ships MXFP4, fits easily, long context.",
        ),
        LocalModelSpec(
            served_name="phi-4",
            hf_repo="microsoft/phi-4",
            params="14B (dense)",
            quant="FP16",
            approx_vram_gb=28,
            tp_size=2,
            max_model_len=16384,
            license="MIT",
            notes="Small, predictable, strong reasoning/code; FP16 across both cards.",
        ),
        LocalModelSpec(
            served_name="mistral-small-3",
            hf_repo="mistralai/Mistral-Small-3.2-24B-Instruct-2506",
            params="24B (dense)",
            quant="FP16 (use an -AWQ build to leave KV room)",
            approx_vram_gb=48,
            tp_size=2,
            max_model_len=32768,
            license="Apache-2.0",
            notes="Strong enterprise-friendly general model; prefer an AWQ/FP8 quant for headroom.",
        ),
        LocalModelSpec(
            served_name="gemma-4-26b-a4b",
            hf_repo="cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit",
            params="26B-A4B (MoE, 4B active)",
            quant="AWQ-INT4",
            approx_vram_gb=14,
            tp_size=2,
            max_model_len=32768,
            license="Gemma",
            hf_query="gemma-4 26B A4B AWQ",
            match_terms=("gemma-4", "26b"),
            prefer_terms=("awq", "4bit", "int4"),
            notes="Gemma 4 MoE (2026); AWQ-INT4 fits easily and runs fast. "
            "Needs a recent vLLM with Gemma 4 support.",
        ),
        LocalModelSpec(
            served_name="gemma-4-31b",
            hf_repo="cyankiwi/gemma-4-31B-it-AWQ-4bit",
            params="31B (dense)",
            quant="AWQ-INT4",
            approx_vram_gb=18,
            tp_size=2,
            max_model_len=32768,
            license="Gemma",
            hf_query="gemma-4 31B AWQ",
            match_terms=("gemma-4", "31b"),
            prefer_terms=("awq", "4bit", "int4"),
            notes="Gemma 4 dense (2026); use the AWQ-INT4 build to fit 48 GB "
            "(FP16 ~62 GB won't; NVFP4 builds are Blackwell-only, not 3090). "
            "Heavier but stronger single-stream than the MoE sibling.",
        ),
        LocalModelSpec(
            served_name="qwen3-30b-a3b",
            hf_repo="Qwen/Qwen3-30B-A3B-Instruct-2507",
            params="30B-A3B (MoE, 3B active)",
            quant="FP16 (use a GPTQ/AWQ Int4 build to fit)",
            approx_vram_gb=60,
            tp_size=2,
            max_model_len=131072,
            license="Apache-2.0",
            notes="MoE throughput sweet spot for prosumer dual-3090; download an Int4 quant to fit VRAM.",
        ),
        LocalModelSpec(
            served_name="qwen3-32b",
            hf_repo="Qwen/Qwen3-32B-AWQ",
            params="32B (dense)",
            quant="AWQ-INT4",
            approx_vram_gb=20,
            tp_size=2,
            max_model_len=32768,
            license="Apache-2.0",
            notes="Default. Dense 32B at AWQ-INT4 fits comfortably with room for KV cache.",
        ),
        LocalModelSpec(
            served_name="deepseek-r1-distill-70b",
            hf_repo="deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
            params="70B (dense, distilled reasoning)",
            quant="FP16 (use an -AWQ build to fit)",
            approx_vram_gb=140,
            tp_size=2,
            max_model_len=16384,
            license="MIT",
            notes="Reasoning distill; only fits 48 GB via an AWQ/GPTQ INT4 build (tight, short context).",
        ),
        LocalModelSpec(
            served_name="qwen2.5-72b",
            hf_repo="Qwen/Qwen2.5-72B-Instruct-AWQ",
            params="72B (dense)",
            quant="AWQ-INT4",
            approx_vram_gb=40,
            tp_size=2,
            max_model_len=16384,
            license="Qwen",
            notes="Largest dense that fits at INT4; ~40 GB weights leave only short-context KV room.",
        ),
        LocalModelSpec(
            served_name="llama-3.3-70b",
            hf_repo="casperhansen/llama-3.3-70b-instruct-awq",
            params="70B (dense)",
            quant="AWQ-INT4",
            approx_vram_gb=40,
            tp_size=2,
            max_model_len=8192,
            license="Llama-3.3",
            notes="Reference dual-3090 70B build; AWQ-INT4, short context (8K) to leave KV room.",
        ),
    ]
}


def list_local_models() -> list[LocalModelSpec]:
    """All catalog specs, in catalog order."""
    return list(LOCAL_MODELS.values())


def get_local_model(served_name: str) -> LocalModelSpec:
    """Look up a spec by served name; fail loudly if unknown."""
    if served_name not in LOCAL_MODELS:
        known = ", ".join(LOCAL_MODELS)
        raise KeyError(f"unknown local model '{served_name}'. Known: {known}")
    return LOCAL_MODELS[served_name]


# Self-hosted models use vLLM guided decoding -> json_schema structured output.
_VLLM_CAPS = ModelCapabilities(
    supports_tool_choice=True,
    supports_json_mode=True,
    supports_json_schema=True,
    preferred_structured_method="json_schema",
)

for _spec in LOCAL_MODELS.values():
    register(_spec.served_name, _VLLM_CAPS)
