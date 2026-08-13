from __future__ import annotations

import asyncio

import pytest
from tests.fast.fixtures.capability_fixtures import FakeBackendCapability
from tests.fast.ray.rollout.conftest import make_args_with_sglang_config

from miles.ray.specs.static_addrs import (
    assert_deployment_is_this_runs_trainer,
    static_trainer_controller_addrs,
    trainer_controller_url,
)
from miles.ray.specs.train import compute_trainer_controller_provider, compute_trainer_ids
from miles.utils.workers.types import DeploymentIdentity

_TRAINER_IDS = ["actor", "critic"]


def _args(tmp_path, **overrides):
    return make_args_with_sglang_config(
        tmp_path,
        server_groups=[{"worker_type": "regular", "num_gpus": 8, "num_gpus_per_engine": 4}],
        rollout_num_gpus=8,
        **overrides,
    )


class TestTrainerControllerUrl:
    def test_a_run_without_the_flag_names_no_controller(self, tmp_path):
        """An all-in-one run finds its own controller, so nothing may be invented for it."""
        assert trainer_controller_url(_args(tmp_path), trainer_id="actor", trainer_ids=_TRAINER_IDS) is None

    def test_a_bare_address_belongs_to_the_first_trainer(self, tmp_path):
        """Most runs train one model, and writing 'actor=' on every launch is noise."""
        args = _args(tmp_path, trainer_controller_addrs=["10.0.0.1:8000"])

        assert trainer_controller_url(args, trainer_id="actor", trainer_ids=_TRAINER_IDS) == "10.0.0.1:8000"

    def test_each_trainer_is_addressed_separately(self, tmp_path):
        """A critic is its own controller in its own pod, and calling the actor's would train the wrong model."""
        args = _args(tmp_path, trainer_controller_addrs=["actor=10.0.0.1:8000", "critic=10.0.0.2:8000"])

        assert trainer_controller_url(args, trainer_id="critic", trainer_ids=_TRAINER_IDS) == "10.0.0.2:8000"

    def test_refuses_a_trainer_id_that_is_not_one_of_the_run_s(self, tmp_path):
        """A typo would otherwise leave the trainer it meant to name silently unaddressed."""
        args = _args(tmp_path, trainer_controller_addrs=["actro=10.0.0.1:8000"])

        with pytest.raises(AssertionError, match="not one of"):
            trainer_controller_url(args, trainer_id="actor", trainer_ids=_TRAINER_IDS)

    def test_refuses_a_run_whose_second_trainer_was_left_unaddressed(self, tmp_path):
        """A trainer named by nothing would be reached at the first one's controller and train the wrong model."""
        args = _args(tmp_path, trainer_controller_addrs=["actor=10.0.0.1:8000"])

        with pytest.raises(AssertionError, match="exactly one"):
            trainer_controller_url(args, trainer_id="critic", trainer_ids=_TRAINER_IDS)

    def test_refuses_two_controllers_for_one_trainer_id(self, tmp_path):
        """One trainer id is one trainer, and silently using one of the two would drop the other."""
        args = _args(tmp_path, trainer_controller_addrs=["actor=10.0.0.1:8000", "actor=10.0.0.2:8000"])

        with pytest.raises(AssertionError, match="exactly one"):
            trainer_controller_url(args, trainer_id="actor", trainer_ids=_TRAINER_IDS)

    def test_refuses_mixing_bare_and_prefixed_entries(self, tmp_path):
        """Half a flag keyed by trainer id and half positional has no obviously intended reading."""
        args = _args(tmp_path, trainer_controller_addrs=["10.0.0.1:8000", "critic=10.0.0.2:8000"])

        with pytest.raises(AssertionError, match="uniformly"):
            trainer_controller_url(args, trainer_id="actor", trainer_ids=_TRAINER_IDS)

    def test_collects_the_address_of_every_trainer_to_wait_on(self, tmp_path):
        """Nothing is called before every deployment the run reaches is listening."""
        args = _args(tmp_path, trainer_controller_addrs=["actor=10.0.0.1:8000", "critic=http://10.0.0.2:9000"])

        addrs = static_trainer_controller_addrs(args, trainer_ids=_TRAINER_IDS)

        assert [(addr.host, addr.port) for addr in addrs] == [("10.0.0.1", 8000), ("10.0.0.2", 9000)]


class TestTrainerIds:
    def test_a_single_policy_run_drives_one_trainer(self, tmp_path):
        """The flag keys on trainer ids, so a run with one trainer is the case that may write a bare address."""
        assert compute_trainer_ids(_args(tmp_path)) == ["actor"]

    def test_a_critic_run_drives_a_trainer_per_trainer_id(self, tmp_path):
        """A critic is deployed as its own trainer and so takes its own entry of the flag."""
        assert compute_trainer_ids(_args(tmp_path, use_critic=True)) == ["actor", "critic"]


class TestProviderSelection:
    def test_a_given_trainer_controller_address_is_used_instead_of_the_backend_s(self, tmp_path):
        """The trainer lives in another deployment, whose names this one's backend cannot resolve."""
        args = _args(tmp_path, trainer_controller_addrs=["10.0.0.1:8000"])
        capability = FakeBackendCapability(static_provider=object())

        provider = compute_trainer_controller_provider(args, capability=capability, trainer_id="actor")

        addrs = asyncio.run(provider.get_addrs("trainer-controller-actor-0-0"))
        assert addrs["rpc"].addr == "http://10.0.0.1:8000"
        assert capability.requested_static_pool_ids == []

    def test_an_all_in_one_run_still_asks_its_own_backend_for_the_trainer_controller(self, tmp_path):
        """Nothing addresses a ray actor statically, so the all-in-one path must be untouched."""
        capability = FakeBackendCapability(static_provider=object())

        provider = compute_trainer_controller_provider(_args(tmp_path), capability=capability, trainer_id="actor")

        assert provider is capability.static_provider
        assert capability.requested_static_pool_ids == ["trainer-controller-actor"]


class TestTheAddressesNameOneRun:
    @staticmethod
    def _identity(*, run_uuid: str, deploy_component: str = "trainer") -> DeploymentIdentity:
        return DeploymentIdentity(run_uuid=run_uuid, deploy_component=deploy_component)

    def test_a_deployment_of_this_run_is_accepted(self, tmp_path):
        """Every deployment of one run carries the same run uuid, so the usual case must pass silently."""
        args = _args(tmp_path)

        assert_deployment_is_this_runs_trainer(self._identity(run_uuid=args.run_uuid), args=args)

    def test_a_deployment_of_another_run_stops_the_launch(self, tmp_path):
        """Pointing at last run's release trains weights this run never updates, and looks like bad rewards."""
        args = _args(tmp_path)

        with pytest.raises(AssertionError, match="drives run"):
            assert_deployment_is_this_runs_trainer(self._identity(run_uuid="ffffffffffffffff"), args=args)

    def test_an_unsplit_release_of_this_run_stops_the_launch(self, tmp_path):
        """It carries an orchestration script of its own, which drives the very trainer this launch would drive."""
        args = _args(tmp_path)

        with pytest.raises(AssertionError, match="nothing but the trainer"):
            assert_deployment_is_this_runs_trainer(
                self._identity(run_uuid=args.run_uuid, deploy_component="all"), args=args
            )
