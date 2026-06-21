"""Manage a background vLLM server: start, stop, status, ensure-correct-model.

One active service per box, tracked by a single state file under
``$TRADINGAGENTS_VLLM_RUNTIME_DIR`` (default ``~/.cache/tradingagents/vllm``).
Logs go to ``<runtime>/logs/<served-name>-<port>.log`` unless a path is given.

Used by ``scripts/model_manager.py`` (``serve -d`` / ``stop`` / ``status``) and
auto-started by LLM task scripts (e.g. ``scripts/extract_events.py``) so a task
spins up the right model, reuses it if already running, and replaces a different
running model.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path

from . import config
from .local_models import get_local_model, is_downloaded, local_path

_SERVE_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "serve_vllm.sh"


def runtime_dir() -> Path:
    return config.runtime_dir()


def _state_file() -> Path:
    return runtime_dir() / "service.json"


def default_log_dir() -> Path:
    return config.log_dir()


@dataclass
class ServiceState:
    pid: int
    served_name: str
    model_ref: str
    host: str
    port: int
    base_url: str
    log_file: str
    started_at: float


def _read_raw() -> ServiceState | None:
    f = _state_file()
    if not f.exists():
        return None
    return ServiceState(**json.loads(f.read_text()))


def _write(state: ServiceState) -> None:
    _state_file().write_text(json.dumps(asdict(state), indent=2))


def _clear() -> None:
    _state_file().unlink(missing_ok=True)


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _health(base_url: str, served_name: str | None = None, timeout: float = 2.0) -> bool:
    """GET <base_url>/models; True if up (and, if given, serving served_name).

    Connection errors are expected while the server boots, so they map to False
    rather than propagating — this is polling, not error masking.
    """
    url = base_url.rstrip("/") + "/models"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            if resp.status != 200:
                return False
            body = json.loads(resp.read().decode())
    except (urllib.error.URLError, OSError, ValueError):
        return False
    if served_name is None:
        return True
    return served_name in {m.get("id") for m in body.get("data", [])}


def is_healthy(base_url: str, served_name: str | None = None) -> bool:
    """Public health check: is an OpenAI-compatible server up at ``base_url``?"""
    return _health(base_url, served_name)


def _can_generate(base_url: str, served_name: str, timeout: float = 30.0) -> bool:
    """True if the engine actually completes a 1-token generation.

    ``/models`` only proves the API front-end is up; a wedged EngineCore (stuck
    on a prior request, the common "service hangs" failure) keeps answering
    ``/models`` while never finishing a generation. This probe issues a trivial
    completion so a hung engine reads as unhealthy and gets restarted, instead
    of every retry reusing the same stuck service.
    """
    url = base_url.rstrip("/") + "/chat/completions"
    payload = json.dumps({
        "model": served_name,
        "messages": [{"role": "user", "content": "ok"}],
        "max_tokens": 1,
        "temperature": 0,
    }).encode()
    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status == 200
    except (urllib.error.URLError, OSError, ValueError):
        return False


def _base_url(host: str, port: int) -> str:
    h = "localhost" if host in ("0.0.0.0", "") else host
    return f"http://{h}:{port}/v1"


def status() -> ServiceState | None:
    """The running service, or None. Clears stale state if the process is gone."""
    st = _read_raw()
    if st is None:
        return None
    if not _pid_alive(st.pid):
        _clear()
        return None
    return st


def _stop_pid(pid: int, timeout: float = 20.0) -> None:
    try:
        pgid = os.getpgid(pid)
    except ProcessLookupError:
        return
    os.killpg(pgid, signal.SIGTERM)
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not _pid_alive(pid):
            return
        time.sleep(0.5)
    os.killpg(pgid, signal.SIGKILL)


def stop() -> bool:
    """Stop the running service. Returns True if one was running."""
    st = _read_raw()
    if st is None:
        return False
    if _pid_alive(st.pid):
        _stop_pid(st.pid)
    _clear()
    return True


def start(
    model: str,
    *,
    port: int = 8000,
    host: str = "0.0.0.0",
    max_model_len: int | None = None,
    log_file: str | None = None,
    env_overrides: dict | None = None,
    wait_timeout: float = 600.0,
) -> ServiceState:
    """Launch vLLM detached, wait until it serves ``model``, record state.

    Fails loudly if the model isn't downloaded or the server exits / times out.
    """
    spec = get_local_model(model)
    if not is_downloaded(spec):
        raise FileNotFoundError(
            f"model '{model}' is not downloaded; run "
            f"`python scripts/model_manager.py download {model}` first"
        )

    model_ref = str(local_path(spec))
    env = dict(os.environ)
    if env_overrides:
        env.update(env_overrides)
    env.update(
        MODEL=model_ref,
        SERVED_MODEL_NAME=spec.served_name,
        TP_SIZE=str(spec.tp_size),
        MAX_MODEL_LEN=str(max_model_len or spec.max_model_len),
        PORT=str(port),
        HOST=host,
    )

    base_url = _base_url(host, port)
    log_path = Path(log_file) if log_file else default_log_dir() / f"{spec.served_name}-{port}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logf = open(log_path, "ab")
    try:
        logf.write(f"\n=== starting {spec.served_name} on {host}:{port} at "
                   f"{time.strftime('%Y-%m-%d %H:%M:%S')} ===\n".encode())
        logf.flush()
        proc = subprocess.Popen(
            ["bash", str(_SERVE_SCRIPT)],
            env=env,
            stdout=logf,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,  # own process group -> killpg stops all TP workers
        )
    finally:
        logf.close()

    state = ServiceState(
        pid=proc.pid,
        served_name=spec.served_name,
        model_ref=model_ref,
        host=host,
        port=port,
        base_url=base_url,
        log_file=str(log_path),
        started_at=time.time(),
    )
    _write(state)

    deadline = time.time() + wait_timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            _clear()
            raise RuntimeError(
                f"vLLM exited with code {proc.returncode} during startup; see {log_path}"
            )
        if _health(base_url, spec.served_name):
            return state
        time.sleep(2.0)

    _stop_pid(proc.pid)
    _clear()
    raise TimeoutError(f"vLLM '{model}' not ready after {wait_timeout:.0f}s; see {log_path}")


def ensure(
    model: str,
    *,
    port: int = 8000,
    host: str = "0.0.0.0",
    max_model_len: int | None = None,
    log_file: str | None = None,
    env_overrides: dict | None = None,
    wait_timeout: float = 600.0,
    probe_generation: bool = True,
) -> ServiceState:
    """Guarantee ``model`` is the running service and return its state.

    - already serving ``model`` and healthy -> reuse it (no restart);
    - a different model is running, it's unhealthy, or (when
      ``probe_generation``) its engine can't complete a 1-token generation
      (a wedged/hung EngineCore) -> stop it, then start fresh;
    - nothing running -> start.
    """
    spec = get_local_model(model)
    st = status()
    if st is not None:
        healthy = st.served_name == spec.served_name and _health(st.base_url, spec.served_name)
        if healthy and probe_generation and not _can_generate(st.base_url, spec.served_name):
            print(f"vLLM '{spec.served_name}' answers /models but a generation probe "
                  f"timed out (wedged engine); restarting ...")
            healthy = False
        if healthy:
            return st
        stop()
    return start(
        model, port=port, host=host, max_model_len=max_model_len,
        log_file=log_file, env_overrides=env_overrides, wait_timeout=wait_timeout,
    )
