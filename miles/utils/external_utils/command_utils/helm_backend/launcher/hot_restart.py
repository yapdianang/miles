from __future__ import annotations

import datetime
from argparse import Namespace

from miles.ray.specs.rollout import ROLLOUT_EXECUTOR_POOL_ID
from miles.utils.external_utils.command_utils.helm_backend import naming
from miles.utils.external_utils.command_utils.helm_backend.launcher.manifest_types import (
    STATEFUL_SET_KIND,
    Manifest,
    compute_manifest_object_key,
)
from miles.utils.lora import is_lora_enabled
from miles.utils.multi_lora import is_multi_lora_enabled
from miles.utils.pydantic_utils import FrozenStrictBaseModel
from miles.utils.workers.types import DeployComponent, DeploySelector, HotRestartComponent


class HotRestartPlan(FrozenStrictBaseModel):
    restart_at: str = ""
    restart_pools: frozenset[str] = frozenset()
    rebuilt_object_keys: frozenset[str] = frozenset()

    @property
    def restarts_orchestration(self) -> bool:
        return bool(self.rebuilt_object_keys)


def plan_hot_restart(
    args: Namespace,
    *,
    components: frozenset[HotRestartComponent],
    selector: DeploySelector,
    release: str,
    installed_manifest: Manifest | None = None,
) -> HotRestartPlan:
    if not components:
        return _carry_installed_stamp(installed_manifest, release=release)

    _assert_preconditions(args, selector=selector)
    assert installed_manifest is not None, (
        f"--hot-restart takes over the trainers and the inference side of a run that is already up, and no release "
        f"{release!r} is installed; installing it here would build a second orchestration script beside the one "
        f"still driving those trainers, so launch this run normally instead"
    )

    return HotRestartPlan(
        restart_at=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="microseconds"),
        restart_pools=frozenset({ROLLOUT_EXECUTOR_POOL_ID}),
        rebuilt_object_keys=frozenset(
            _stateful_set_key(release, component)
            for component in (naming.ORCHESTRATOR_COMPONENT, ROLLOUT_EXECUTOR_POOL_ID)
        ),
    )


def _stateful_set_key(release: str, component: str) -> str:
    return compute_manifest_object_key(kind=STATEFUL_SET_KIND, name=naming.component_name(release, component))


def _carry_installed_stamp(installed_manifest: Manifest | None, *, release: str) -> HotRestartPlan:
    """Render the stamp a previous hot restart left, so an ordinary relaunch is a zero diff and rolls no pod."""
    if installed_manifest is None:
        return HotRestartPlan()

    stamp = installed_manifest.restart_at(object_name=naming.component_name(release, naming.ORCHESTRATOR_COMPONENT))
    if stamp is None:
        return HotRestartPlan()

    executor_stamp = installed_manifest.restart_at(
        object_name=naming.component_name(release, ROLLOUT_EXECUTOR_POOL_ID)
    )
    return HotRestartPlan(
        restart_at=stamp,
        restart_pools=frozenset({ROLLOUT_EXECUTOR_POOL_ID} if executor_stamp == stamp else ()),
    )


def _assert_preconditions(args: Namespace, *, selector: DeploySelector) -> None:
    assert selector.component.deploys_orchestration_script(), (
        f"--hot-restart replaces the orchestration script and the rollout executor, and a --deploy-component "
        f"{selector.value} release deploys neither of them; run it against the release that carries them "
        f"({DeployComponent.ALL.value} or {DeployComponent.PRIMARY.value})"
    )
    assert not args.indep_dp, (
        "--hot-restart is not supported with --indep-dp: the independent-DP quorum and its store are rebuilt by "
        "the trainer controller's init, and nothing yet re-derives that state for a new orchestration script"
    )
    assert args.train_backend == "megatron", (
        f"--hot-restart reloads a trainer's state in place, and only the megatron trainer answers that; "
        f"--train-backend {args.train_backend} would abort every rollout in flight and only then fail"
    )
    assert not is_multi_lora_enabled(args), (
        "--hot-restart is not supported with --multi-lora: a reload hides the adapters in the megatron slots while "
        "it loads the base checkpoint, so they physically survive it while the bookkeeping that owns their slots "
        "does not, and the next reconcile would load every adapter into an occupied slot"
    )
    assert not is_lora_enabled(args), (
        "--hot-restart is not supported with --lora: a save writes the adapter beside the base checkpoint and "
        "nothing re-derives that adapter path for a resumed trainer, so a reload would either restore the adapter "
        "the run started from or leave the in-memory adapter and its optimizer untouched at a rolled-back base"
    )
    assert args.megatron_to_hf_mode != "bridge", (
        "--hot-restart is not supported with --megatron-to-hf-mode bridge: the bridge path pins start_rollout_id to "
        "0, so a take-over would restart the orchestration at rollout 0 against trainers whose weights are at the "
        "checkpoint they were rolled back to"
    )

    requested = args.requested_checkpoint_source
    unrebuildable = sorted(name for name in ("no_load_optim", "no_load_rng", "finetune") if requested[name])
    assert not unrebuildable, (
        f"--hot-restart is not supported together with {unrebuildable}: a take-over reloads a checkpoint into a "
        f"trainer that is still alive, and skipping the optimizer or the rng state leaves the live optimizer "
        f"moments and generators of the rollout the run had already reached on top of rolled-back weights, which "
        f"is a state a cold start can never produce; nothing rebuilds them, so this run is refused instead"
    )
