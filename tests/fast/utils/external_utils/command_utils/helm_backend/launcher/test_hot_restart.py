from types import SimpleNamespace

import pytest

from miles.utils.external_utils.command_utils.helm_backend.launcher.hot_restart import HotRestartPlan, plan_hot_restart
from miles.utils.external_utils.command_utils.helm_backend.launcher.manifest_types import (
    RESTART_AT_ANNOTATION,
    Manifest,
)
from miles.utils.external_utils.command_utils.helm_backend.naming import component_name
from miles.utils.workers.types import DeployComponent, DeploySelector, HotRestartComponent

_RELEASE = "miles-run-260101-000000-000-primary"
_BOTH = frozenset(HotRestartComponent)
_STAMP = "2026-08-12T09:00:00+00:00"
_ORCHESTRATOR_OBJECT = component_name(_RELEASE, "orchestrator")
_EXECUTOR_OBJECT = component_name(_RELEASE, "rollout-executor")


def _args(**overrides) -> SimpleNamespace:
    return SimpleNamespace(
        **{
            "trainer_controller_addrs": {"actor": "host:8000"},
            "indep_dp": False,
            "multi_lora": False,
            "lora_rank": 0,
            "lora_adapter_path": None,
            "megatron_to_hf_mode": "raw",
            "train_backend": "megatron",
            "requested_checkpoint_source": dict(no_load_optim=False, no_load_rng=False, finetune=False),
            **overrides,
        }
    )


def _plan(
    *,
    components: frozenset[HotRestartComponent] = frozenset(),
    selector: DeploySelector | None = None,
    installed_manifest: Manifest | None = None,
    **arg_overrides,
) -> HotRestartPlan:
    return plan_hot_restart(
        _args(**arg_overrides),
        components=components,
        selector=selector or DeploySelector(component=DeployComponent.PRIMARY),
        release=_RELEASE,
        installed_manifest=installed_manifest,
    )


def _installed_manifest(*, stamped: bool, stamped_components: tuple[str, ...] = ("orchestrator", "rollout-executor")):
    return Manifest(
        objects=[
            dict(
                kind="StatefulSet",
                metadata=dict(name=component_name(_RELEASE, component)),
                spec=dict(
                    template=dict(
                        metadata=dict(
                            annotations=(
                                {RESTART_AT_ANNOTATION: _STAMP} if stamped and component in stamped_components else {}
                            )
                        )
                    )
                ),
            )
            for component in ("orchestrator", "rollout-executor")
        ]
    )


class TestPlanHotRestart:
    def test_no_components_plans_nothing(self):
        """An ordinary launch must not stamp a restart annotation onto a live run."""
        plan = _plan()

        assert plan.restart_at == ""
        assert plan.restart_pools == frozenset()
        assert plan.rebuilt_object_keys == frozenset()
        assert plan.restarts_orchestration is False

    def test_a_restart_stamps_a_timestamp_and_replaces_both_components(self):
        """The pods only roll because the stamp moved, and the two components are replaced together."""
        plan = _plan(components=_BOTH, installed_manifest=_installed_manifest(stamped=False))

        assert plan.restart_at != ""
        assert plan.restart_pools == {"rollout-executor"}
        assert plan.rebuilt_object_keys == {
            f"StatefulSet/{_ORCHESTRATOR_OBJECT}",
            f"StatefulSet/{_EXECUTOR_OBJECT}",
        }
        assert plan.restarts_orchestration is True


class TestThePreconditions:
    @pytest.mark.parametrize("component", [DeployComponent.ALL, DeployComponent.PRIMARY])
    def test_every_release_that_carries_the_orchestration_script_may_hot_restart(self, component: DeployComponent):
        """The requirement asks for a plain single-release run too, and the machinery does not need a split one."""
        plan = _plan(
            components=_BOTH,
            selector=DeploySelector(component=component),
            installed_manifest=_installed_manifest(stamped=False),
            trainer_controller_addrs=None,
        )

        assert plan.restarts_orchestration is True

    @pytest.mark.parametrize("component", [DeployComponent.TRAINER, DeployComponent.INFERENCE])
    def test_a_release_without_an_orchestration_script_cannot_hot_restart_one(self, component: DeployComponent):
        """These releases carry neither of the two components the flag replaces, so there is nothing to restart."""
        selector = DeploySelector(component=component, instance="a" if component.takes_instance() else None)

        with pytest.raises(AssertionError, match="deploys neither of them"):
            _plan(components=_BOTH, selector=selector)

    @pytest.mark.parametrize(
        "overrides, match",
        [
            (dict(indep_dp=True), "indep-dp"),
            (dict(train_backend="fsdp"), "train-backend"),
            (dict(multi_lora=True), "multi-lora"),
            (dict(lora_rank=8), "--lora"),
            (dict(megatron_to_hf_mode="bridge"), "bridge"),
            (
                dict(requested_checkpoint_source=dict(no_load_optim=True, no_load_rng=False, finetune=False)),
                "no_load_optim",
            ),
            (
                dict(requested_checkpoint_source=dict(no_load_optim=False, no_load_rng=True, finetune=False)),
                "no_load_rng",
            ),
            (
                dict(requested_checkpoint_source=dict(no_load_optim=False, no_load_rng=False, finetune=True)),
                "finetune",
            ),
        ],
    )
    def test_a_run_that_cannot_be_taken_over_in_place_is_refused(self, overrides: dict, match: str):
        """Each of these makes an in-place take-over unsafe, and the refusal lands before anything is aborted."""
        with pytest.raises(AssertionError, match=match):
            _plan(components=_BOTH, installed_manifest=_installed_manifest(stamped=False), **overrides)

    def test_a_release_that_is_not_installed_yet_cannot_be_hot_restarted(self):
        """Installing it here would put a second orchestration script beside the one still driving the trainers."""
        with pytest.raises(AssertionError, match="is installed"):
            _plan(components=_BOTH, installed_manifest=None)

    def test_the_preconditions_are_not_checked_without_a_hot_restart(self):
        """Every ordinary launch calls this, and an ordinary launch needs none of those flags."""
        plan = _plan(
            selector=DeploySelector(component=DeployComponent.ALL),
            trainer_controller_addrs=None,
            indep_dp=True,
        )

        assert plan.restart_at == ""


class TestTheStampAnOrdinaryRelaunchRenders:
    def test_a_relaunch_of_a_never_hot_restarted_run_stamps_nothing(self):
        """The annotation must not appear out of nowhere, or the first relaunch would roll the pods."""
        plan = _plan(installed_manifest=_installed_manifest(stamped=False))

        assert plan.restart_at == ""
        assert plan.restart_pools == frozenset()

    def test_a_relaunch_after_a_hot_restart_renders_the_stamp_it_finds_and_replaces_nothing(self):
        """Dropping it would make the pod template differ, so the diff gate refuses an ordinary relaunch forever."""
        plan = _plan(installed_manifest=_installed_manifest(stamped=True))

        assert plan.restart_at == _STAMP
        assert plan.restart_pools == {"rollout-executor"}
        assert plan.rebuilt_object_keys == frozenset()
        assert plan.restarts_orchestration is False

    def test_only_the_pools_that_really_carry_the_stamp_are_rendered_with_it(self):
        """Rendering it onto a pool that never got it makes an ordinary relaunch a diff the gate refuses forever."""
        plan = _plan(installed_manifest=_installed_manifest(stamped=True, stamped_components=("orchestrator",)))

        assert plan.restart_at == _STAMP
        assert plan.restart_pools == frozenset()

    def test_a_hot_restart_stamps_a_value_of_its_own_over_the_installed_one(self):
        """The pod only rolls because the value moved, so a carried-forward stamp would restart nothing."""
        plan = _plan(components=_BOTH, installed_manifest=_installed_manifest(stamped=True))

        assert plan.restart_at != _STAMP

    def test_a_first_install_has_no_manifest_to_read(self):
        """Every launch calls this, including the one that installs the release."""
        assert _plan(installed_manifest=None).restart_at == ""
