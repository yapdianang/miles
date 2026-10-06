"""Dev loop for serve_mimo_v26_flash_tinker.py on one Kubernetes pod (kubectl uses KUBECONFIG).

  up POD --node NODE --image IMAGE  create the pod from mimo_dev_pod.yaml and wait until it is Ready
  sync POD                          copy this checkout's tracked and new files over the pod's checkout
  restart POD [SERVE ARGS]          stop the service, start the launcher's `serve` with SERVE ARGS, wait for the gateway
  stop POD                          stop the service
  gate POD GATE [GATE ARGS]         run mimo_gates.py GATE in the pod against its gateway

restart, stop and gate send this checkout's copy of the tool to the pod, so the image needs no copy. The
service runs in a session of its own: stop sends its launcher SIGTERM (the launcher then stops its Ray job),
SIGKILLs what is left of that session, and runs `ray stop --force`; it does not signal other processes.
The service log is /tmp/mimo-dev/service.log in the pod.

python examples/multi_lora/tools/mimo_dev.py up mimo-dev --node <node> --image <miles-mimo-tinker image>
python examples/multi_lora/tools/mimo_dev.py restart mimo-dev --sglang-mem-fraction-static 0.85
python examples/multi_lora/tools/mimo_dev.py gate mimo-dev parity
python examples/multi_lora/tools/mimo_dev.py gate mimo-dev engine-stats /tmp/mimo-dev/service.log
"""

from __future__ import annotations

import argparse
import os
import signal
import string
import subprocess
import sys
import tarfile
import time
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
LAUNCHER = "examples/multi_lora/serve_mimo_v26_flash_tinker.py"
SERVE_ARGS = ("--save-dir", "/mnt/shared-volume/checkpoints", "--run-id", "dev", "--skip-upgrade-check")
GATEWAY_URL = "http://localhost:10613"
READY_LINE = "tinker gateway serving"
STATE_DIR = Path("/tmp/mimo-dev")
PID_FILE = "service.pid"
LOG_FILE = "service.log"
RAY_STOP = ("ray", "stop", "--force")
POLL_SECONDS = 10.0


def render_pod(pod: str, node: str, image: str) -> str:
    # safe_substitute keeps Kubernetes' own $(NODE_NAME) references.
    template = string.Template((TOOLS_DIR / "mimo_dev_pod.yaml").read_text())
    return template.safe_substitute(pod=pod, node=node, image=image)


def write_archive(fileobj, root: Path, names: list[str]) -> None:
    """Stream a gzipped tar of the existing files among `names`, relative to `root`."""
    with tarfile.open(fileobj=fileobj, mode="w|gz") as archive:
        for name in names:
            if (root / name).is_file():
                archive.add(root / name, arcname=name, recursive=False)


def launcher_argv(serve_args: list[str]) -> list[str]:
    return ["serve", *SERVE_ARGS, *serve_args]


def process_state(pid: int) -> tuple[str, int] | None:
    """(state, start time in clock ticks) of a process, or None if there is none with this PID."""
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    except OSError:
        return None
    return fields[0], int(fields[19])


def _running(pid: int) -> bool:
    state = process_state(pid)
    return state is not None and state[0] not in "ZX"


def start_service(state_dir: Path, command: list[str]) -> subprocess.Popen:
    """Start `command` as the leader of a new session; the PID file names that session."""
    state_dir.mkdir(parents=True, exist_ok=True)
    with (state_dir / LOG_FILE).open("wb") as log:
        process = subprocess.Popen(
            command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
    (state_dir / PID_FILE).write_text(f"{process.pid} {process_state(process.pid)[1]}\n")
    return process


def stop_service(state_dir: Path, timeout: float) -> None:
    """SIGTERM the session leader, wait for it, then SIGKILL the rest of its session."""
    pid_file = state_dir / PID_FILE
    if not pid_file.exists():
        return
    session, start_time = map(int, pid_file.read_text().split())
    state = process_state(session)
    # Another process with the leader's PID means the session ended and the PID was reused.
    if state is None or state[1] == start_time:
        if _running(session):
            os.kill(session, signal.SIGTERM)
            deadline = time.monotonic() + timeout
            while _running(session) and time.monotonic() < deadline:
                time.sleep(1)
        subprocess.run(["pkill", "-KILL", "-s", str(session)], check=False)
        subprocess.run(RAY_STOP, capture_output=True, check=False)
    pid_file.unlink()


def wait_ready(process: subprocess.Popen, log: Path, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while process.poll() is None and time.monotonic() < deadline:
        if READY_LINE in log.read_text(errors="replace"):
            return True
        time.sleep(POLL_SECONDS)
    return READY_LINE in log.read_text(errors="replace")


def _run_script(pod: str, script: Path, argv: list[str]) -> int:
    with script.open("rb") as source:
        return subprocess.run(["kubectl", "exec", "-i", pod, "--", "python3", "-", *argv], stdin=source).returncode


def up(args) -> int:
    pod = render_pod(args.pod, args.node, args.image)
    subprocess.run(["kubectl", "apply", "-f", "-"], input=pod, text=True, check=True)
    wait = ["kubectl", "wait", "--for=condition=Ready", f"pod/{args.pod}", f"--timeout={args.timeout:.0f}s"]
    return subprocess.run(wait).returncode


def sync(args) -> int:
    listing = ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"]
    repo = TOOLS_DIR.parents[2]
    names = subprocess.run(listing, cwd=repo, capture_output=True, check=True).stdout.decode().split("\0")
    command = ["kubectl", "exec", "-i", args.pod, "--", "tar", "-xzf", "-", "--no-same-owner"]
    with subprocess.Popen(command, stdin=subprocess.PIPE) as untar:
        write_archive(untar.stdin, repo, [name for name in names if name])
        untar.stdin.close()
    return untar.returncode


def restart(args) -> int:
    argv = ["pod-restart", "--ready-timeout", str(args.ready_timeout), *launcher_argv(args.serve_args)]
    return _run_script(args.pod, Path(__file__), argv)


def stop(args) -> int:
    return _run_script(args.pod, Path(__file__), ["pod-stop"])


def gate(args) -> int:
    url = [] if args.gate == "engine-stats" else ["--base-url", GATEWAY_URL]
    return _run_script(args.pod, TOOLS_DIR / "mimo_gates.py", [args.gate, *url, *args.gate_args])


def pod_restart(args) -> int:
    stop_service(args.state_dir, args.stop_timeout)
    start = time.monotonic()
    process = start_service(args.state_dir, [sys.executable, LAUNCHER, *args.launcher_argv])
    log = args.state_dir / LOG_FILE
    if wait_ready(process, log, args.ready_timeout):
        print(f"gateway ready on {GATEWAY_URL} after {time.monotonic() - start:.0f}s; log {log}")
        return 0
    status = f"exited with {process.returncode}" if process.poll() is not None else "is still starting"
    print("\n".join(log.read_text(errors="replace").splitlines()[-40:]))
    print(f"no gateway after {time.monotonic() - start:.0f}s; the service {status}; log {log}")
    return 1


def pod_stop(args) -> int:
    stop_service(args.state_dir, args.stop_timeout)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    def command(name: str, run, help_text: str) -> argparse.ArgumentParser:
        sub = commands.add_parser(name, help=help_text)
        sub.set_defaults(run=run)
        return sub

    create = command("up", up, "create the dev pod and wait until it is Ready")
    create.add_argument("pod")
    create.add_argument("--node", required=True)
    create.add_argument("--image", required=True)
    create.add_argument("--timeout", type=float, default=3600.0)

    command("sync", sync, "copy this checkout over the pod's checkout").add_argument("pod")

    remote_restart = command("restart", restart, "restart the service in the pod")
    remote_restart.add_argument("--ready-timeout", type=float, default=3600.0)
    remote_restart.add_argument("pod")
    remote_restart.add_argument("serve_args", nargs=argparse.REMAINDER, help="extra launcher `serve` args")

    command("stop", stop, "stop the service in the pod").add_argument("pod")

    remote_gate = command("gate", gate, "run a mimo_gates.py gate in the pod")
    remote_gate.add_argument("pod")
    remote_gate.add_argument("gate")
    remote_gate.add_argument("gate_args", nargs=argparse.REMAINDER)

    # The commands below run inside the pod.
    for name, run in (("pod-restart", pod_restart), ("pod-stop", pod_stop)):
        local = command(name, run, "(in the pod) " + name.removeprefix("pod-") + " the service")
        local.add_argument("--state-dir", type=Path, default=STATE_DIR)
        local.add_argument("--stop-timeout", type=float, default=120.0)
        if name == "pod-restart":
            local.add_argument("--ready-timeout", type=float, default=3600.0)
            local.add_argument("launcher_argv", nargs=argparse.REMAINDER, help="launcher command and its args")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    sys.exit(args.run(args))


if __name__ == "__main__":
    main()
