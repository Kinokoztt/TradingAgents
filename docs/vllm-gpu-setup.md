# vLLM GPU Setup & CUDA Sanity Guide (2x RTX 3090)

A checklist to make sure your CUDA / driver stack is healthy before serving a
model with `scripts/serve_vllm.sh`. Target box: 2x RTX 3090 (Ampere, sm_86),
PCIe (no NVLink), Linux.

## TL;DR — the one thing that matters

The pip `vllm` / `torch` wheels **bundle their own CUDA runtime**. You do NOT
need a matching system CUDA toolkit (`nvcc`). You only need an NVIDIA **driver**
new enough for the CUDA version the wheel was built against.

- vLLM `0.23.0` ships `torch==2.11.0` built against **CUDA 13** → needs driver
  **>= 580** (Linux).
- If your driver is older and you can't update it, install a vLLM/torch build
  matching your existing CUDA (see "Driver too old" below) — don't install a
  system toolkit to "fix" it; that won't change which runtime the wheel uses.

## Prerequisites

| Item | Requirement |
| --- | --- |
| GPUs | 2x RTX 3090 (or any 2 Ampere+ cards). sm_86 is supported by CUDA 12 and 13. |
| OS | Linux (Ubuntu 22.04+ tested by the community recipes). |
| NVIDIA driver | >= 580 for the CUDA-13 vLLM 0.23.0 wheel (`nvidia-smi` shows it). |
| Python | 3.10 – 3.14 (vLLM 0.23 requires `<3.15,>=3.10`). |
| Disk | ~20-80 GB per model under `$TRADINGAGENTS_MODELS_DIR`. |
| Interconnect | PCIe is fine; no NVLink needed (we pass `--disable-custom-all-reduce`). |

## Step 1 — see both GPUs and read the driver

```bash
nvidia-smi
```

Check:
- Both RTX 3090s are listed (GPU 0 and GPU 1).
- Top-right **"CUDA Version: 13.x"** — this is the **maximum** CUDA the driver
  supports, not what's installed. It must be >= the wheel's CUDA (13 for vLLM
  0.23). If it says 12.x, your driver is too old for the default wheel.
- **Driver Version** >= 580 for CUDA 13.

```bash
nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv
```

If `nvidia-smi` itself fails: the driver isn't installed/loaded — fix that first
(`sudo ubuntu-drivers install` or NVIDIA's `.run`), nothing below will work.

## Step 2 — install the serving deps

```bash
# on the GPU box, in your venv/conda env
pip install -e ".[serve]"      # vllm>=0.23.0 + huggingface_hub
```

Then confirm PyTorch sees CUDA and **both** GPUs:

```bash
python -c "import torch; print('torch', torch.__version__); \
print('cuda build', torch.version.cuda); \
print('available', torch.cuda.is_available()); \
print('device count', torch.cuda.device_count()); \
print([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])"
```

Expected: `available True`, `device count 2`, two `NVIDIA GeForce RTX 3090`.
- `available False` → driver/runtime mismatch (see Step 1 / "Driver too old").
- `device count 1` → check `CUDA_VISIBLE_DEVICES` isn't pinned to one card.

Verify the compute capability is recognized (3090 = 8.6):

```bash
python -c "import torch; print(torch.cuda.get_device_capability(0))"   # (8, 6)
```

## Step 3 — verify multi-GPU collectives (NCCL)

Tensor parallelism needs the two cards to talk via NCCL. Quick check:

```bash
python -c "
import torch, torch.distributed as dist, os
print('nccl available:', torch.distributed.is_nccl_available())
print('peer 0->1 access:', torch.cuda.can_device_access_peer(0,1))
"
```

`nccl available: True` is what matters. Peer access may be False on PCIe boards
without NVLink — that's expected and fine; NCCL falls back to PCIe/host staging.
Because of this, `serve_vllm.sh` already passes `--disable-custom-all-reduce`
(the custom all-reduce kernels require NVLink). Leave it on for 3090s.

## Step 4 — launch and smoke-test

```bash
# downloads if needed, then serves on :8000
python scripts/model_manager.py serve qwen3-32b --exec
```

In another shell, confirm the OpenAI-compatible API is up:

```bash
curl -s http://localhost:8000/v1/models | python -m json.tool

curl -s http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3-32b","messages":[{"role":"user","content":"reply with OK"}],"max_tokens":8}'
```

Then point the app at it and run a real extraction:

```bash
export TRADINGAGENTS_LLM_PROVIDER=vllm
export VLLM_BASE_URL=http://localhost:8000/v1   # only if calling from another host
python scripts/extract_events.py --as-of 2026-05-11 --news-tickers AAPL --model qwen3-32b
```

## Troubleshooting

| Symptom | Cause | Fix |
| --- | --- | --- |
| `torch.cuda.is_available()` is False; `nvidia-smi` works | Driver older than the wheel's CUDA (e.g. driver supports CUDA 12, wheel is CUDA 13) | Update the driver to >= 580, OR install a matching build (below). |
| `CUDA error: forward compatibility was attempted on non supported HW` / version mismatch | Driver/runtime mismatch | Same as above — align driver with wheel CUDA. |
| Startup hangs at "initializing NCCL" / all-reduce | NVLink-only kernels on a PCIe box | Ensure `--disable-custom-all-reduce` (default in serve_vllm.sh). Optionally `export NCCL_P2P_DISABLE=1`. |
| `CUDA out of memory` at load | Model too big for 48 GB at this quant/context | Use an INT4/AWQ build; lower `MAX_MODEL_LEN`; keep `KV_CACHE_DTYPE=fp8`; reduce `GPU_MEM_UTIL`. |
| `no kernel image is available for execution` (sm mismatch) | Wheel built without sm_86, or a quant kernel needs Blackwell | Use a standard CUDA wheel; avoid NVFP4 builds (Blackwell-only) on 3090s. |
| flashinfer / attention backend errors | flashinfer build mismatch | `export VLLM_ATTENTION_BACKEND=FLASH_ATTN` (or `XFORMERS`) to bypass flashinfer. |
| Only 1 GPU used | `CUDA_VISIBLE_DEVICES` pinned | `export CUDA_VISIBLE_DEVICES=0,1` and use `TP_SIZE=2`. |

### Driver too old and you can't update

Install a vLLM/torch build matching the CUDA your driver supports instead of the
default CUDA-13 wheel. For example, for a CUDA 12.x driver, pick a torch cu12x
index and a vLLM version compatible with it (consult the vLLM install matrix for
your CUDA). The goal: the torch wheel's `torch.version.cuda` must be <= the
"CUDA Version" shown by `nvidia-smi`.

## Final checklist

- [ ] `nvidia-smi` lists both 3090s; Driver >= 580; CUDA Version >= 13.
- [ ] `torch.cuda.is_available()` True and `device_count() == 2`.
- [ ] `get_device_capability(0) == (8, 6)`.
- [ ] `torch.distributed.is_nccl_available()` True.
- [ ] `curl /v1/models` returns your served model name.
- [ ] A tiny chat completion returns text.
