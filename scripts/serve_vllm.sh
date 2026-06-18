#!/usr/bin/env bash
#
# Launch a self-hosted vLLM OpenAI-compatible server for TradingAgents.
#
# Target hardware: 2x RTX 3090 (24GB each, 48GB total). A 32B model in FP16
# needs ~64GB and will NOT fit, so we default to an AWQ-quantized 32B
# (~20GB weights) sharded across both GPUs with tensor parallelism. For a
# non-quantized option use the MoE Qwen3-30B-A3B-Instruct instead.
#
# Once running, point TradingAgents at it:
#   export TRADINGAGENTS_LLM_PROVIDER=vllm
#   export TRADINGAGENTS_DEEP_THINK_LLM=qwen3-32b
#   export TRADINGAGENTS_QUICK_THINK_LLM=qwen3-32b
#   export VLLM_BASE_URL=http://<this-host>:8000/v1   # if calling remotely
#
# Structured output: TradingAgents asks Qwen for json_schema responses
# (see capabilities.py), which vLLM serves via its guided-decoding backend
# with no extra flags required.
#
# Usage:
#   scripts/serve_vllm.sh                       # defaults below
#   MODEL=Qwen/Qwen3-30B-A3B-Instruct scripts/serve_vllm.sh
#   PORT=8001 MAX_MODEL_LEN=16384 scripts/serve_vllm.sh

set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-32B-AWQ}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3-32b}"
PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"
TP_SIZE="${TP_SIZE:-2}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.92}"
# fp8 KV cache is what makes long context fit on 48 GB; set to "auto" to disable.
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"
# PCIe-connected 3090s have no NVLink, so the custom all-reduce kernels must be
# disabled (forces NCCL). Set to 0 if your cards are NVLink-bridged.
DISABLE_CUSTOM_ALL_REDUCE="${DISABLE_CUSTOM_ALL_REDUCE:-1}"
# Anything extra (e.g. --tool-call-parser, --reasoning-parser, --trust-remote-code).
EXTRA_ARGS="${EXTRA_ARGS:-}"

# Restrict to the two 3090s unless the caller overrides it.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

args=(
  --served-model-name "${SERVED_MODEL_NAME}"
  --host "${HOST}"
  --port "${PORT}"
  --tensor-parallel-size "${TP_SIZE}"
  --max-model-len "${MAX_MODEL_LEN}"
  --gpu-memory-utilization "${GPU_MEM_UTIL}"
  --kv-cache-dtype "${KV_CACHE_DTYPE}"
  --enable-prefix-caching
)
if [[ "${DISABLE_CUSTOM_ALL_REDUCE}" == "1" ]]; then
  args+=(--disable-custom-all-reduce)
fi
if [[ -n "${EXTRA_ARGS}" ]]; then
  # shellcheck disable=SC2206
  args+=(${EXTRA_ARGS})
fi

echo "Serving ${MODEL} as '${SERVED_MODEL_NAME}' on ${HOST}:${PORT}"
echo "  tp=${TP_SIZE}  max-model-len=${MAX_MODEL_LEN}  kv=${KV_CACHE_DTYPE}  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

exec vllm serve "${MODEL}" "${args[@]}"
