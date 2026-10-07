"""The MiMo dev loop stops only its own service session and sends its tools to the pod."""

import io
import signal
import subprocess
import sys
import tarfile
import time

import pytest
import yaml
from examples.multi_lora.tools import mimo_dev as dev
from examples.multi_lora.tools import mimo_gates as gates


@pytest.fixture(autouse=True)
def fast_polls(monkeypatch):
    monkeypatch.setattr(dev, "POLL_SECONDS", 0.05)
    monkeypatch.setattr(dev, "RAY_STOP", ("true",))


@pytest.fixture
def bystander():
    process = subprocess.Popen(["sleep", "600"], start_new_session=True)
    yield process
    process.kill()
    process.wait()


def _wait_for(path, timeout=10.0):
    deadline = time.monotonic() + timeout
    while not path.exists() or not path.read_text().strip():
        assert time.monotonic() < deadline, f"{path} was not written"
        time.sleep(0.05)
    return int(path.read_text())


def test_pod_template_takes_pod_node_and_image_and_keeps_kubernetes_references():
    text = dev.render_pod("mimo-dev", "vmnode-1", "registry/miles-mimo-tinker:abc")
    pod = yaml.safe_load(text)
    container = pod["spec"]["containers"][0]
    assert pod["metadata"]["name"] == "mimo-dev"
    assert pod["spec"]["nodeSelector"]["kubernetes.io/hostname"] == "vmnode-1"
    assert container["image"] == "registry/miles-mimo-tinker:abc"
    assert "$(NODE_NAME)" in text and "${" not in text


def test_stop_kills_the_service_session_and_spares_other_processes(tmp_path, bystander):
    child_file = tmp_path / "child"
    leader = dev.start_service(tmp_path, ["bash", "-c", f"sleep 600 & echo $! > {child_file}; wait"])
    child = _wait_for(child_file)

    dev.stop_service(tmp_path, timeout=10)

    leader.wait(timeout=10)
    assert leader.returncode == -signal.SIGTERM
    assert not dev._running(child)
    assert dev._running(bystander.pid)
    assert not (tmp_path / dev.PID_FILE).exists()


def test_stop_leaves_a_reused_pid_alone(tmp_path, bystander):
    _, start_time = dev.process_state(bystander.pid)
    (tmp_path / dev.PID_FILE).write_text(f"{bystander.pid} {start_time + 1}\n")

    dev.stop_service(tmp_path, timeout=1)

    assert dev._running(bystander.pid)
    assert not (tmp_path / dev.PID_FILE).exists()


def test_wait_ready_sees_the_gateway_line_or_the_launcher_exit(tmp_path):
    ready = dev.start_service(tmp_path, ["bash", "-c", f"echo '{dev.READY_LINE} m on :10613'; sleep 600"])
    assert dev.wait_ready(ready, tmp_path / dev.LOG_FILE, timeout=10)
    dev.stop_service(tmp_path, timeout=10)
    ready.wait(timeout=10)

    failed = dev.start_service(tmp_path, ["bash", "-c", "echo boom; exit 3"])
    assert not dev.wait_ready(failed, tmp_path / dev.LOG_FILE, timeout=10)
    assert failed.returncode == 3


def test_pod_restart_starts_the_launcher_with_its_argv_and_replaces_the_previous_one(tmp_path, monkeypatch, capsys):
    launcher = tmp_path / "launcher.py"
    launcher.write_text(f"import sys, time\nprint({dev.READY_LINE!r}, sys.argv[1:], flush=True)\ntime.sleep(600)\n")
    monkeypatch.setattr(dev, "LAUNCHER", str(launcher))
    state = ["--state-dir", str(tmp_path / "state")]
    pids = []
    for run_id in ("a", "b"):
        with pytest.raises(SystemExit) as exit:
            dev.main(["pod-restart", *state, "--ready-timeout", "10", "serve", "--run-id", run_id])
        assert exit.value.code == 0
        pids.append(int((tmp_path / "state" / dev.PID_FILE).read_text().split()[0]))
        assert f"['serve', '--run-id', '{run_id}']" in (tmp_path / "state" / dev.LOG_FILE).read_text()
    assert not dev._running(pids[0]) and dev._running(pids[1])
    with pytest.raises(SystemExit):
        dev.main(["pod-stop", *state])
    assert not dev._running(pids[1])
    assert "gateway ready" in capsys.readouterr().out


def test_remote_commands_run_this_checkouts_tools_in_the_pod(monkeypatch):
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs["stdin"].name))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(dev.subprocess, "run", fake_run)
    for argv in (
        ["restart", "mimo-dev", "--hf-checkpoint", "/ckpt"],
        ["gate", "mimo-dev", "replay", "--workload", "/root/replay.json.gz"],
        ["gate", "mimo-dev", "engine-stats", "/tmp/mimo-dev/service.log"],
        ["run", "mimo-dev", gates.__file__, "--help"],
    ):
        with pytest.raises(SystemExit):
            dev.main(argv)

    exec_prefix = ["kubectl", "exec", "-i", "mimo-dev", "--", "python3", "-"]
    (restart, restart_script), (replay, replay_script), (stats, _), (run, run_script) = calls
    assert restart[:7] == exec_prefix and restart_script.endswith("mimo_dev.py")
    in_pod = dev.build_parser().parse_args(restart[7:])
    assert in_pod.launcher_argv == ["serve", *dev.SERVE_ARGS, "--hf-checkpoint", "/ckpt"]
    assert replay[7:] == ["replay", "--base-url", dev.GATEWAY_URL, "--workload", "/root/replay.json.gz"]
    assert replay_script.endswith("mimo_gates.py")
    assert gates.build_parser().parse_args(replay[7:]).workload.name == "replay.json.gz"
    assert stats[7:] == ["engine-stats", "/tmp/mimo-dev/service.log"]
    assert run == [*exec_prefix, "--help"] and run_script == gates.__file__


def test_sync_archive_holds_the_listed_files_that_exist(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "a.txt").write_text("a")
    (tmp_path / "sub" / "b.py").write_text("b")
    buffer = io.BytesIO()
    dev.write_archive(buffer, tmp_path, ["a.txt", "sub/b.py", "deleted.py", "sub"])
    buffer.seek(0)
    with tarfile.open(fileobj=buffer, mode="r:gz") as archive:
        assert archive.getnames() == ["a.txt", "sub/b.py"]
        assert archive.extractfile("sub/b.py").read() == b"b"


def test_tools_run_from_stdin_as_in_the_pod():
    for tool in (dev, gates):
        with open(tool.__file__) as source:
            result = subprocess.run(
                [sys.executable, "-", "--help"], stdin=source, capture_output=True, text=True, cwd="/"
            )
        assert result.returncode == 0, result.stderr
