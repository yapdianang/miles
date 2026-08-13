from __future__ import annotations

import ast
import asyncio
from argparse import Namespace
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from types import SimpleNamespace

import pytest
from tests.fast.charts.utils import REPO_ROOT

from miles.utils.function_registry import function_registry
from miles.utils.workers import deployment_entrypoint
from miles.utils.workers.deployment_entrypoint import DeploymentWiring

_WIRING_PATH = "tests.fake_deployment:wiring"
_SPECS_PATH = "tests.fake_deployment:specs"


class _FakeWiring:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.wiring_argv: list[str] | None = None
        self.specs_argv: list[str] | None = None

    def compute_wiring(self, argv: list[str]) -> DeploymentWiring:
        self.wiring_argv = argv
        return DeploymentWiring(
            launch_worker_manager=self._launch_worker_manager,
            describe_reachability=self._describe_reachability,
        )

    def compute_specs(self, argv: list[str]) -> list[SimpleNamespace]:
        self.specs_argv = argv
        return [SimpleNamespace(name="trainer-controller")]

    def _launch_worker_manager(self) -> str:
        self.calls.append("launch_worker_manager")
        return "worker-manager"

    async def _describe_reachability(self) -> str:
        self.calls.append("describe_reachability")
        return "Reach it with --trainer-controller-addrs actor=trainer-host:8000"


@contextmanager
def _injected(monkeypatch, fake: _FakeWiring) -> Iterator[Namespace]:
    monkeypatch.setattr(deployment_entrypoint, "configure_logger", lambda *_args, **_kwargs: None)
    with ExitStack() as stack:
        stack.enter_context(function_registry.temporary(_WIRING_PATH, fake.compute_wiring))
        stack.enter_context(function_registry.temporary(_SPECS_PATH, fake.compute_specs))
        yield deployment_entrypoint._parse_own_args(["--specs", _SPECS_PATH, "--wiring", _WIRING_PATH])


async def _serve_until_it_stays_up(monkeypatch, *, args: Namespace, worker_argv: list[str], fake: _FakeWiring) -> None:
    with _injected(monkeypatch, fake) as own_args:
        task = asyncio.ensure_future(
            deployment_entrypoint._serve_deployed_workers(args, own_args=own_args, worker_argv=worker_argv)
        )
        for _ in range(100):
            await asyncio.sleep(0)
            if len(fake.calls) == 2:
                break

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


class TestServeDeployedWorkers:
    async def test_it_runs_the_injected_wiring_and_then_stays_up(self, monkeypatch):
        """The entrypoint knows nothing about ray: it launches and describes whatever the wiring flag hands it."""
        fake = _FakeWiring()

        await _serve_until_it_stays_up(
            monkeypatch,
            args=Namespace(deploy_component="trainer"),
            worker_argv=["--model-name", "whatever"],
            fake=fake,
        )

        assert fake.calls == ["launch_worker_manager", "describe_reachability"]

    async def test_it_hands_the_worker_argv_to_both_injected_functions(self, monkeypatch):
        """Both flags name functions of the run's own argv, so the entrypoint has to pass it through unchanged."""
        fake = _FakeWiring()

        await _serve_until_it_stays_up(
            monkeypatch,
            args=Namespace(deploy_component="trainer"),
            worker_argv=["--model-name", "whatever"],
            fake=fake,
        )

        assert fake.wiring_argv == ["--model-name", "whatever"]
        assert fake.specs_argv == ["--model-name", "whatever"]

    async def test_a_deployment_that_carries_the_orchestration_script_is_refused(self, monkeypatch):
        """Such a deployment is started by its driver script, and serving it here would run it twice."""
        fake = _FakeWiring()

        with _injected(monkeypatch, fake) as own_args:
            with pytest.raises(AssertionError, match="carries no orchestration script"):
                await deployment_entrypoint._serve_deployed_workers(
                    Namespace(deploy_component="all"), own_args=own_args, worker_argv=[]
                )

        assert fake.calls == []


class TestParseOwnArgs:
    @pytest.mark.parametrize("own_argv", [[], ["--specs", _SPECS_PATH], ["--wiring", _WIRING_PATH]])
    def test_a_launch_that_names_only_one_of_the_two_injection_points_is_refused(self, own_argv):
        """Without both dotted paths the entrypoint has no run to deploy, so it must fail at launch, not later."""
        with pytest.raises(SystemExit):
            deployment_entrypoint._parse_own_args(own_argv)

    def test_it_takes_both_dotted_paths(self):
        """The two flags are the entrypoint's whole configuration surface."""
        own_args = deployment_entrypoint._parse_own_args(["--specs", _SPECS_PATH, "--wiring", _WIRING_PATH])

        assert own_args.specs == _SPECS_PATH
        assert own_args.wiring == _WIRING_PATH


_DRIVER_SCRIPTS = [
    "train.py",
    "train_async.py",
    "train_multi_lora_async.py",
    "train_multi_policy.py",
]


class TestEveryDriverRunsItsOwnOrchestrationScript:
    @pytest.mark.parametrize("script", _DRIVER_SCRIPTS)
    def test_a_driver_starts_its_training_without_asking_what_this_launch_deploys(self, script):
        """A deployment carrying no orchestration script is served by its own entrypoint, never by a driver."""
        assert "asyncio.run" in _functions_called_in_main(script)
        assert "deploy_component" not in (REPO_ROOT / script).read_text()


def _functions_called_in_main(script: str) -> set[str]:
    tree = ast.parse((REPO_ROOT / script).read_text())
    return {
        ast.unparse(node.func)
        for statement in tree.body
        if isinstance(statement, ast.If)
        for node in ast.walk(statement)
        if isinstance(node, ast.Call)
    }
