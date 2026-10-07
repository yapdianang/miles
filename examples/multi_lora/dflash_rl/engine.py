"""Standalone MiMo-V2.6 SGLang servers with the engine flags of serve_mimo_v26_flash_tinker.py."""

import os
import shlex
import signal
import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import httpx

from examples.multi_lora import serve_mimo_v26_flash_tinker as launcher

MILES_ROOT = Path(__file__).resolve().parents[3]
LORA_NAME = "policy"
ENGINE_GPUS = launcher._ENGINE_GPUS
# Miles' --target-modules attn adapters on MiMo-V2 (the engine's lora_target_modules).
_LORA_TARGETS = [f"model.layers.*.self_attn.{proj}_proj" for proj in "qkvo"]


def server_argv(
    *,
    hf_checkpoint: str,
    drafter: str,
    block_size: int,
    port: int,
    draft_quantization: str | None = None,
    lora_path: str | None = None,
    extra: tuple[str, ...] = (),
) -> list[str]:
    """``sglang.launch_server`` argv: the Miles engine's --sglang-* flags with this drafter and block size."""
    serve_args = launcher.ScriptArgs(hf_checkpoint=hf_checkpoint, dflash_drafter=drafter, dflash_block_size=block_size)
    argv = [sys.executable, "-m", "sglang.launch_server", "--model-path", hf_checkpoint, "--port", str(port)]
    argv += ["--trust-remote-code", "--skip-server-warmup"]
    for token in shlex.split(launcher._sglang_args(serve_args)):
        name = token.removeprefix("--")
        if name == "rollout-num-gpus-per-engine":
            argv.append("--tp-size")
        elif name.startswith("sglang-"):
            argv.append("--" + name.removeprefix("sglang-"))
        elif name != token:
            raise ValueError(f"no SGLang server flag for engine argument {token}")
        else:
            argv.append(token)
    if draft_quantization:
        argv += ["--speculative-draft-model-quantization", draft_quantization]
    if lora_path:
        argv += ["--enable-lora", "--max-lora-rank", "32", "--max-loras-per-batch", "1", "--lora-target-modules"]
        argv += [*_LORA_TARGETS, "--lora-paths", f"{LORA_NAME}={lora_path}"]
    return argv + list(extra)


def engine_gpus(index: int) -> str:
    return ",".join(str(gpu) for gpu in range(index * ENGINE_GPUS, (index + 1) * ENGINE_GPUS))


@contextmanager
def running_servers(
    argvs: list[list[str]], *, ports: list[int], log_dir: Path, timeout: float = 3600
) -> Iterator[list[str]]:
    """Start server i on GPUs engine_gpus(i), yield their URLs once /health_generate answers, stop them on exit."""
    env = os.environ | {"PYTHONPATH": f"{MILES_ROOT}:{os.environ.get('PYTHONPATH', '')}"}
    processes = []
    try:
        for index, argv in enumerate(argvs):
            with open(log_dir / f"engine{index}.log", "w") as log:
                processes.append(
                    subprocess.Popen(
                        argv,
                        env=env | {"CUDA_VISIBLE_DEVICES": engine_gpus(index)},
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                )
        urls = [f"http://127.0.0.1:{port}" for port in ports]
        for index, (process, url) in enumerate(zip(processes, urls, strict=True)):
            _wait_healthy(url, process, timeout, log_dir / f"engine{index}.log")
        yield urls
    finally:
        for process in processes:
            _signal_group(process, signal.SIGTERM)
        for process in processes:
            try:
                process.wait(timeout=120)
            except subprocess.TimeoutExpired:
                _signal_group(process, signal.SIGKILL)


def _signal_group(process: subprocess.Popen, signum: int) -> None:
    # The server's scheduler and detokenizer processes share its session.
    try:
        os.killpg(process.pid, signum)
    except ProcessLookupError:
        pass


def _wait_healthy(url: str, process: subprocess.Popen, timeout: float, log_path: Path) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"SGLang server exited with code {process.returncode}; see {log_path}")
        try:
            if httpx.get(f"{url}/health_generate", timeout=60).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(10)
    raise TimeoutError(f"SGLang server at {url} not healthy after {timeout:.0f}s; see {log_path}")
