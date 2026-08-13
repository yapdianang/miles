from __future__ import annotations

import itertools
import json
import shlex

import pytest
from tests.fast.ray.rollout.conftest import make_args_with_sglang_config

from miles.ray.specs.entrypoint import compute_specs
from miles.utils.external_utils.command_utils.base_backend import ExecuteTrainConfig
from miles.utils.external_utils.command_utils.helm_backend.launcher import entrypoint
from miles.utils.external_utils.command_utils.helm_backend.launcher.entrypoint import _describe_reachable_addrs
from miles.utils.external_utils.command_utils.helm_backend.launcher.values.misc import MooncakeInfo, MooncakePlan
from miles.utils.external_utils.command_utils.helm_backend.naming import (
    HELM_RELEASE_NAME_MAX,
    RUN_ID_MAX_LENGTH,
    RunNames,
)
from miles.utils.workers.types import DeployComponent
from miles.utils.workers.worker_provider.kubernetes.helm.naming import component_name, static_worker_host

RUN_ID = "260101-000000-000"
NAMESPACE = "rl"
RPC_PORT = 8000

_SPLIT_COMPONENTS = [DeployComponent.PRIMARY, DeployComponent.TRAINER]


def _args(tmp_path, *, component: DeployComponent):
    return make_args_with_sglang_config(
        tmp_path,
        rollout_num_gpus=8,
        use_session_server=False,
        use_critic=False,
        sglang_router_port=None,
        deploy_component=component.value,
    )


def _release(component: DeployComponent) -> str:
    return RunNames.release(run_id=RUN_ID, deploy_component=component)


def _object_names(tmp_path, *, component: DeployComponent) -> set[str]:
    release = _release(component)
    return {component_name(release, spec.name) for spec in compute_specs(_args(tmp_path, component=component))}


class TestTwoReleasesOfOneRun:
    def test_no_two_releases_name_the_same_object(self, tmp_path):
        """One name shared between two releases is one launch quietly upgrading another launch's workload."""
        names_by_component = {
            component: _object_names(tmp_path, component=component) for component in _SPLIT_COMPONENTS
        }

        for first, second in itertools.combinations(_SPLIT_COMPONENTS, 2):
            assert not names_by_component[first] & names_by_component[second]

    def test_the_trainer_launch_prints_the_address_its_own_release_answers_on(self, tmp_path):
        """This string is pasted straight into the primary launch, so a derived name would be a dead address."""
        args = _args(tmp_path, component=DeployComponent.TRAINER)
        release = _release(DeployComponent.TRAINER)

        printed = _describe_reachable_addrs(args, specs=compute_specs(args), release=release)

        host = static_worker_host(release, "trainer-controller-actor", 0)
        assert printed == f"--trainer-controller-addrs actor={host}:{RPC_PORT}"

    def test_the_store_flags_it_prints_carry_every_init_kwarg_the_run_was_launched_with(self):
        """The other launch pastes this line, and a dropped kwarg leaves the deployments on different protocols."""
        primary = _release(DeployComponent.PRIMARY)
        plan = MooncakePlan(init_kwargs={"master_server_address": "0.0.0.0:50051", "protocol": "tcp"}, port=50051)

        printed = entrypoint._describe_shared_object_store(plan, release=primary, namespace=NAMESPACE)

        tokens = shlex.split(printed)
        init_kwargs = json.loads(tokens[tokens.index("--mooncake-store-init-kwargs") + 1])
        assert init_kwargs == {
            "master_server_address": f"{MooncakeInfo.master_service_host(primary, NAMESPACE)}:50051",
            "protocol": "tcp",
        }

    def test_the_object_store_master_the_trainer_release_names_is_the_primary_releases_own(self):
        """The trainer launch types this address by hand, so one place has to compute it."""
        primary = _release(DeployComponent.PRIMARY)

        master = MooncakeInfo.master_service_host(primary, NAMESPACE)

        assert master == f"{component_name(primary, 'mooncake-master')}.{NAMESPACE}.svc.cluster.local"
        assert master != MooncakeInfo.master_service_host(_release(DeployComponent.TRAINER), NAMESPACE)


def test_a_run_id_ending_in_a_component_name_is_refused() -> None:
    """Its unsplit release would carry the very name another run's split launch installs its own release under."""
    with pytest.raises(AssertionError, match="ends in a component name"):
        entrypoint.execute_train(
            request=None, config=ExecuteTrainConfig(run_id=f"{RUN_ID}-trainer", namespace=NAMESPACE)
        )


class TestTheRunIdLeavesRoomForTheComponentSuffix:
    def test_the_longest_accepted_run_id_names_a_legal_release_for_every_component(self):
        """A run id that only fits unsplit is a trap: the split launch of it fails inside helm."""
        run_id = "a" * RUN_ID_MAX_LENGTH

        for component in DeployComponent:
            assert len(RunNames.release(run_id=run_id, deploy_component=component)) <= HELM_RELEASE_NAME_MAX

    def test_a_longer_run_id_is_refused_where_the_release_is_named(self):
        """helm would refuse the install itself, long after the launch computed every object name from it."""
        with pytest.raises(AssertionError, match=str(HELM_RELEASE_NAME_MAX)):
            RunNames.release(run_id="a" * (RUN_ID_MAX_LENGTH + 1))
