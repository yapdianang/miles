from __future__ import annotations

import itertools
from argparse import Namespace

from tests.fast.fixtures.capability_fixtures import FakeBackendCapability
from tests.fast.ray.rollout.conftest import make_args_with_sglang_config

from miles.ray import wiring
from miles.ray.specs import addressing
from miles.ray.specs.static_addrs import trainer_controller_url
from miles.utils.workers.worker_info import WorkerInfo
from miles.utils.workers.worker_provider.base import BaseWorkerProvider
from miles.utils.workers.worker_spec import RPC_PORT_NAME, HostAndPort, NamedHostAndPorts


class _AddressBookProvider(BaseWorkerProvider):
    def __init__(self, addrs_by_worker_name: dict[str, HostAndPort]) -> None:
        self._addrs_by_worker_name = addrs_by_worker_name

    async def get_addrs(self, worker_name: str) -> NamedHostAndPorts:
        return {RPC_PORT_NAME: self._addrs_by_worker_name[worker_name]}

    def get_worker_infos(self, *, cell_ids: list[str]) -> list[list[WorkerInfo]]:
        raise NotImplementedError


def _trainer_args(tmp_path, **overrides) -> Namespace:
    return make_args_with_sglang_config(tmp_path, deploy_component="trainer", **overrides)


async def _describe(monkeypatch, *, args: Namespace, addrs: dict[str, HostAndPort]) -> str:
    capability = FakeBackendCapability(static_provider=_AddressBookProvider(addrs))
    monkeypatch.setattr(wiring, "get_backend_capability", lambda _args: capability)
    return await addressing.describe_how_the_run_reaches_this_deployment(args)


def _entries_after(description: str, flag: str) -> list[str]:
    tokens = description.split()
    tail = tokens[tokens.index(flag) + 1 :]
    return list(itertools.takewhile(lambda entry: not entry.startswith("--"), tail))


class TestDescribeControllerAddrs:
    async def test_the_trainer_address_it_prints_is_the_one_the_next_launch_takes(self, monkeypatch, tmp_path):
        """This string is the only place a user reads the address from, so it has to parse back unchanged."""
        description = await _describe(
            monkeypatch,
            args=_trainer_args(tmp_path),
            addrs={"trainer-controller-actor-0-0": HostAndPort(host="trainer-host", port=8000)},
        )

        entries = _entries_after(description, "--trainer-controller-addrs")

        assert (
            trainer_controller_url(
                Namespace(trainer_controller_addrs=entries), trainer_id="actor", trainer_ids=["actor"]
            )
            == "trainer-host:8000"
        )

    async def test_a_run_with_a_critic_prints_both_of_its_controllers(self, monkeypatch, tmp_path):
        """A critic is its own controller, and an unnamed one leaves the next launch unable to reach it."""
        description = await _describe(
            monkeypatch,
            args=_trainer_args(tmp_path, use_critic=True),
            addrs={
                "trainer-controller-actor-0-0": HostAndPort(host="actor-host", port=8000),
                "trainer-controller-critic-0-0": HostAndPort(host="critic-host", port=8000),
            },
        )

        entries = _entries_after(description, "--trainer-controller-addrs")
        args = Namespace(trainer_controller_addrs=entries)

        assert trainer_controller_url(args, trainer_id="actor", trainer_ids=["actor", "critic"]) == "actor-host:8000"
        assert trainer_controller_url(args, trainer_id="critic", trainer_ids=["actor", "critic"]) == "critic-host:8000"
