from __future__ import annotations

import pytest
from tests.fast.ray.rollout.conftest import make_args, make_args_with_sglang_config, make_sglang_config_yaml

from miles.ray.specs.entrypoint import compute_specs
from miles.utils.workers.worker_provider.kubernetes.helm.builder import compute_helm_backend_capability
from miles.utils.workers.worker_provider.kubernetes.helm.env import NAMESPACE_ENV_VAR, RELEASE_ENV_VAR
from miles.utils.workers.worker_spec import WorkerCtorContext


class TestComputeSpecs:
    def test_launches_the_controller_then_routers_then_the_session_server_then_every_engine(self, tmp_path):
        """The manager's whole inventory comes from here, so every component must be listed exactly once."""
        config_path = tmp_path / "sglang.yaml"
        config_path.write_text(
            make_sglang_config_yaml(
                server_groups=[
                    {"worker_type": "regular", "num_gpus": 4, "num_gpus_per_engine": 2},
                    {"worker_type": "placeholder", "num_gpus": 4, "num_gpus_per_engine": 4},
                    {"worker_type": "decode", "num_gpus": 8, "num_gpus_per_engine": 4},
                ]
            )
        )
        args = make_args(sglang_config=str(config_path), rollout_num_gpus=16, use_session_server=False)

        specs = compute_specs(args)

        assert [spec.name for spec in specs] == [
            "rollout-executor",
            "multi-lora-controller",
            "inference-controller",
            "inference-router-0",
            "session-server",
            "inference-engine-0-0",
            "inference-engine-0-2",
            "rollout-executor",
            "trainer-controller-actor",
            "trainer-engine-actor",
        ]

    def test_a_disabled_session_server_is_listed_with_no_cells(self, tmp_path):
        """Disabling the session server must not remove it from the inventory, only empty it."""
        config_path = tmp_path / "sglang.yaml"
        config_path.write_text(
            make_sglang_config_yaml(
                server_groups=[{"worker_type": "regular", "num_gpus": 4, "num_gpus_per_engine": 2}]
            )
        )
        args = make_args(sglang_config=str(config_path), rollout_num_gpus=4, use_session_server=False)

        specs = {spec.name: spec for spec in compute_specs(args)}

        assert specs["session-server"].scheduling.num_cells == 0
        assert specs["inference-engine-0-0"].scheduling.num_cells == 2

    def test_debug_train_only_lists_no_inference_engine(self, tmp_path):
        """--debug-train-only must instantiate no sglang engine, since its bundles are the trainer's own gpus."""
        config_path = tmp_path / "sglang.yaml"
        config_path.write_text(
            make_sglang_config_yaml(
                server_groups=[{"worker_type": "regular", "num_gpus": 8, "num_gpus_per_engine": 1}]
            )
        )
        args = make_args(
            sglang_config=str(config_path),
            rollout_num_gpus=8,
            use_session_server=False,
            colocate=True,
            debug_train_only=True,
        )

        specs = compute_specs(args)

        assert [spec.name for spec in specs if spec.name.startswith("inference-engine")] == []


class TestDeployComponentFiltering:
    @staticmethod
    def _args(tmp_path, **overrides):
        return make_args_with_sglang_config(
            tmp_path,
            server_groups=[{"worker_type": "regular", "num_gpus": 4, "num_gpus_per_engine": 2}],
            rollout_num_gpus=4,
            use_session_server=True,
            use_critic=True,
            critic_num_nodes=1,
            critic_num_gpus_per_node=2,
            **overrides,
        )

    def test_a_trainer_deployment_holds_the_trainer_controllers_and_their_ranks_only(self, tmp_path):
        """A trainer release that also installed engines would double the run's gpu bill."""
        specs = compute_specs(self._args(tmp_path, deploy_component="trainer"))

        assert [spec.name for spec in specs] == [
            "trainer-controller-actor",
            "trainer-controller-critic",
            "trainer-engine-actor",
            "trainer-engine-critic",
        ]

    def test_the_primary_deployment_holds_everything_the_trainer_does_not(self, tmp_path):
        """primary is defined by subtraction, so anything unclaimed has to land here rather than nowhere."""
        specs = compute_specs(self._args(tmp_path, deploy_component="primary"))

        assert not [spec.name for spec in specs if spec.name.startswith("trainer-")]
        assert "inference-controller" in [spec.name for spec in specs]
        assert "inference-router-0" in [spec.name for spec in specs]

    def test_every_worker_of_a_trainer_deployment_can_build_its_constructor_arguments(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch
    ):
        """A spec that asks this release for a pool it does not install aborts the pod before it ever serves."""
        monkeypatch.setenv(RELEASE_ENV_VAR, "miles-run-260813-trainer")
        monkeypatch.setenv(NAMESPACE_ENV_VAR, "rl")
        specs = compute_specs(self._args(tmp_path, deploy_component="trainer"))
        capability = compute_helm_backend_capability(specs=specs)
        context = WorkerCtorContext(cell_index=0, worker_in_cell_index=0, gpu_ids=[], capability=capability)

        kwargs_by_name = {
            spec.name: spec.ctor_kwargs(context) for spec in specs if spec.name.startswith("trainer-controller-")
        }

        assert kwargs_by_name["trainer-controller-actor"]["inference_controller"] is None
        assert kwargs_by_name["trainer-controller-critic"]["inference_controller"] is None

    def test_the_two_subsets_partition_the_whole_run(self, tmp_path):
        """A worker in neither subset would never be deployed, and one in both would be deployed twice."""
        whole = [spec.name for spec in compute_specs(self._args(tmp_path, deploy_component="all"))]
        parts = [
            spec.name
            for component in ("primary", "trainer")
            for spec in compute_specs(self._args(tmp_path, deploy_component=component))
        ]

        assert sorted(whole) == sorted(parts)

    def test_a_trainer_role_deployment_holds_that_role_only(self, tmp_path):
        """One release per role is what lets a run's trainers be sized, upgraded and lost independently."""
        specs = compute_specs(self._args(tmp_path, deploy_component="trainer:critic"))

        assert [spec.name for spec in specs] == ["trainer-controller-critic", "trainer-engine-critic"]

    def test_the_roles_of_a_trainer_partition_the_trainer_deployment(self, tmp_path):
        """A role in neither release would never be deployed, and one in both would be deployed twice."""
        whole = [spec.name for spec in compute_specs(self._args(tmp_path, deploy_component="trainer"))]
        parts = [
            spec.name
            for component in ("trainer:actor", "trainer:critic")
            for spec in compute_specs(self._args(tmp_path, deploy_component=component))
        ]

        assert sorted(whole) == sorted(parts)

    def test_an_inference_deployment_holds_the_engines_and_the_one_reporter(self, tmp_path):
        """An engine release carries no controller and no router; it only announces the engines it launches."""
        specs = compute_specs(
            self._args(
                tmp_path,
                deploy_component="inference",
                inference_controller_addrs=["controller:8000"],
                registration_token="secret",
            )
        )

        assert [spec.name for spec in specs] == ["registration-reporter", "inference-engine-0-0"]

    def test_the_primary_deployment_keeps_engines_of_its_own(self, tmp_path):
        """An engine deployment adds engines to a run rather than moving the run's own engines out of it."""
        specs = compute_specs(self._args(tmp_path, deploy_component="primary"))

        assert "inference-engine-0-0" in [spec.name for spec in specs]

    @pytest.mark.parametrize("component", ["all", "primary", "trainer"])
    def test_only_an_inference_deployment_carries_a_reporter(self, tmp_path, component):
        """A reporter beside the controller it reports into would register a deployment into itself."""
        specs = compute_specs(self._args(tmp_path, deploy_component=component))

        assert "registration-reporter" not in [spec.name for spec in specs]
