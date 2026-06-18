# vLLM GPU Setup & CUDA Sanity Guide (2x RTX 3090)

A checklist to make sure your CUDA / driver stack is healthy before serving a
model with `scripts/serve_vllm.sh`. Target box: 2x RTX 3090 (Ampere, sm_86),
PCIe (no NVLink), Linux.

## TL;DR — the one thing that matters

The pip `vllm` / `torch` wheels **bundle their own CUDA runtime**, so you mostly
just need an NVIDIA **driver** new enough for the CUDA version the wheel was built
against. BUT vLLM also JIT-compiles some kernels at startup, which needs two extra
build tools on PATH: a **C compiler** (`gcc`) and the **CUDA compiler** (`nvcc`,
from the CUDA toolkit). Without them startup dies with `Failed to find C compiler`
or `Could not find nvcc`.

- vLLM `0.23.0` ships `torch==2.11.0` built against **CUDA 13** → needs driver
  **>= 580** (Linux), and an `nvcc` whose version matches `torch.version.cuda`.
- If your driver is older and you can't update it, install a vLLM/torch build
  matching your existing CUDA (see "Driver too old" below) — don't install a
  system toolkit to "fix" the *runtime*; that won't change which runtime the wheel
  uses. (The toolkit is still needed separately for `nvcc` JIT, see Step 2.)

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

vLLM JIT-compiles GPU kernels at startup, so it needs two build tools on PATH:

1. **C compiler** (`gcc`) for Triton — else `Failed to find C compiler`.
2. **CUDA compiler** (`nvcc`, from the CUDA toolkit) for quant/MoE kernels — else
   `Could not find nvcc and default cuda_home='/usr/local/cuda' doesn't exist`.

```bash
# 1. C compiler
sudo apt-get update && sudo apt-get install -y build-essential   # or conda: gcc_linux-64 gxx_linux-64
which gcc && export CC=$(which gcc)

# 2. nvcc — version MUST match torch.version.cuda
python -c "import torch; print(torch.version.cuda)"              # e.g. 13.0
```

You need `nvcc` **and** the CUDA dev headers/libs — flashinfer JIT-compiles
sampling/attention kernels that `#include <curand.h>`, `<cublas.h>`, etc., so the
bare compiler is not enough. Install the full toolkit. Avoid conda for this — its
`cuda-toolkit` / `cuda-nvcc` packages fail to solve when the env already has a
stale `cudatoolkit` from `defaults`.

**(a) NVIDIA apt repo — recommended, needs sudo:**

```bash
. /etc/os-release && echo "$ID$VERSION_ID"                      # -> ubuntu2204 / ubuntu2404
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb && sudo apt-get update
sudo apt-get install -y cuda-toolkit-13-0                       # nvcc + all dev headers; match torch.version.cuda
export CUDA_HOME=/usr/local/cuda-13.0
export PATH=$CUDA_HOME/bin:$PATH
ls $CUDA_HOME/include/curand.h                                  # sanity: dev headers present
```

Do **not** `apt install nvidia-cuda-toolkit` — that's Ubuntu's old bundled nvcc
(e.g. CUDA 11.x) and will mismatch `torch.version.cuda`.

**(b) pip wheels — no sudo (more fiddly: needs each lib wheel):**

```bash
pip install nvidia-cuda-nvcc-cu13 nvidia-curand-cu13 nvidia-cublas-cu13 nvidia-cuda-runtime-cu13
export CUDA_HOME=$(python -c "import nvidia.cuda_nvcc,os; print(os.path.dirname(nvidia.cuda_nvcc.__file__))")
export PATH=$CUDA_HOME/bin:$PATH
```

Whichever you pick, verify and persist it:

```bash
nvcc --version                                                  # version == torch.version.cuda
```

Persist `CC`, `CUDA_HOME`, and the `PATH` edit in `~/.bashrc` so every serve picks
them up. Then confirm PyTorch sees CUDA and **both** GPUs:

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

## Step 4 — pick a directory and batch-download the model files

Choose where weights live (a big, fast disk) and pre-fetch the whole catalog
there in one go, so the later serve step has nothing to download.

```bash
# 1. point the store at your chosen directory (persists for this shell)
export TRADINGAGENTS_MODELS_DIR=/mnt/data/models

# 2. see the catalog + rough sizes (and what's already present)
python scripts/model_manager.py list

# 3. batch-download EVERY catalog model into that directory.
#    --latest live-resolves the newest matching HF build per model;
#    --skip-existing avoids re-downloading ones already present.
python scripts/model_manager.py download --all --latest --skip-existing
```

Each model lands in `$TRADINGAGENTS_MODELS_DIR/<served-name>/` (the served name
is reused as the dir name and the vLLM `--served-model-name`). You can also set
the directory inline without exporting:

```bash
python scripts/model_manager.py --models-dir /mnt/data/models download --all --latest
```

Notes:
- Downloading the full catalog is large (the 70B AWQ builds are ~40 GB each).
  To grab just a few, download them by name instead:
  `python scripts/model_manager.py download qwen3-32b --latest`.
- Verify what landed: `python scripts/model_manager.py downloaded`.

## Step 5 — launch and smoke-test

Two ways to launch. Foreground holds the terminal and streams logs:

```bash
python scripts/model_manager.py serve qwen3-32b --exec
```

Background runs it as a detached service, writes logs to a file, and returns once
the model is ready (recommended for normal use):

```bash
python scripts/model_manager.py serve qwen3-32b -d     # blocks until healthy, then returns
python scripts/model_manager.py status                  # model, pid, url, health, log path
python scripts/model_manager.py stop                    # shut it down
```

The background service is tracked in `$TRADINGAGENTS_VLLM_RUNTIME_DIR`
(default `~/.cache/tradingagents/vllm`); logs default to
`<runtime>/logs/<model>-<port>.log`. Only one model runs at a time — starting a
different one stops the previous service first.

Confirm the OpenAI-compatible API is up:

```bash
curl -s http://localhost:8000/v1/models | python -m json.tool

curl -s http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3-32b","messages":[{"role":"user","content":"reply with OK"}],"max_tokens":8}'
```

Then run a real extraction. With provider `vllm` the task **auto-starts** the
right model's service (reusing it if already running), so you can skip the manual
serve step entirely:

```bash
# auto-serves qwen3-32b if not already up, then extracts
python scripts/extract_events.py --as-of 2026-05-11 --news-tickers AAPL --model qwen3-32b

# auto-serve, then shut the service down when the task finishes
python scripts/extract_events.py --as-of 2026-05-11 --news-tickers AAPL --model qwen3-32b --stop-after-task

# disable auto-serve and point at an already-running / remote server instead
python scripts/extract_events.py --as-of 2026-05-11 --news-tickers AAPL --model qwen3-32b \
  --no-auto-serve --backend-url http://localhost:8000/v1
```

## Troubleshooting

| Symptom | Cause | Fix |
| --- | --- | --- |
| `torch.cuda.is_available()` is False; `nvidia-smi` works | Driver older than the wheel's CUDA (e.g. driver supports CUDA 12, wheel is CUDA 13) | Update the driver to >= 580, OR install a matching build (below). |
| `CUDA error: forward compatibility was attempted on non supported HW` / version mismatch | Driver/runtime mismatch | Same as above — align driver with wheel CUDA. |
| `RuntimeError: Failed to find C compiler` (in EngineCore at `determine_available_memory`) | No `gcc` on PATH; Triton can't JIT GPU kernels | `sudo apt-get install -y build-essential` (or conda `gcc_linux-64 gxx_linux-64`), then `export CC=$(which gcc)`. |
| `Could not find nvcc and default cuda_home='/usr/local/cuda' doesn't exist` | No `nvcc`; a quant/MoE kernel needs it to JIT | Install the CUDA toolkit matching `torch.version.cuda` via Step 2 (apt `cuda-toolkit-13-0`), then set `CUDA_HOME` + PATH. |
| flashinfer JIT: `fatal error: curand.h: No such file or directory` (also `cublas.h`, `cuda_runtime.h`) | Only `nvcc` installed, missing CUDA dev headers/libs | Install the FULL toolkit, not just the compiler: `sudo apt-get install -y cuda-toolkit-13-0` (Step 2a). |
| `LibMambaUnsatisfiableError: ... cuda-nvcc ... cudatoolkit ... conflicts` | Conda can't install CUDA-13 packages over a stale `cudatoolkit` | Skip conda for nvcc — use the apt repo or `pip install nvidia-cuda-nvcc-cu13` (Step 2a/2b). |
| Startup hangs at "initializing NCCL" / all-reduce | NVLink-only kernels on a PCIe box | Ensure `--disable-custom-all-reduce` (default in serve_vllm.sh). Optionally `export NCCL_P2P_DISABLE=1`. |
| `CUDA out of memory` at load | Model too big for 48 GB at this quant/context | Use an INT4/AWQ build; lower `MAX_MODEL_LEN`; for AWQ/fp16 set `KV_CACHE_DTYPE=fp8_e5m2`; reduce `GPU_MEM_UTIL`. |
| `ValueError: type fp8e4nv not supported in this architecture ... ('fp8e4b15', 'fp8e5')` (Triton autotuning at startup) | `--kv-cache-dtype fp8` means e4m3, which is Hopper/Ada-only; 3090s (Ampere) only do e5m2 | Use `KV_CACHE_DTYPE=auto` (default), or `fp8_e5m2` for AWQ/fp16 models. Never plain `fp8` on a 3090. |
| `ValueError: fp8_e5m2 kv-cache is not supported with fp8 checkpoints` | The model itself is an fp8 checkpoint; it rejects an fp8 KV cache | `KV_CACHE_DTYPE=auto` (default). `fp8_e5m2` is only for AWQ/fp16 checkpoints. |
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
