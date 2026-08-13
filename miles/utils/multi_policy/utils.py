import logging
from dataclasses import dataclass
from pathlib import Path

from miles.backends.megatron_utils.megatron_config import MegatronConfig, compute_trainer_args, resolve_megatron_config
from miles.backends.sglang_utils.sglang_config import resolve_sglang_config
from miles.ray.hot_restart import TakeOverDeadline
from miles.ray.placement_group import QuietTrainers, create_training_model
from miles.ray.specs.train import compute_trainer_configs
from miles.utils.arguments import validate_async_off_policy_correction
from miles.utils.multi_policy.checkpoint_state import MultiPolicyCheckpointState
from miles.utils.tracking_utils.tracking import define_step_key_metric_group
from miles.utils.workers.worker_handle import BaseWorkerHandle

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainerInfo:
    model_id: str
    start_rollout_id: int
    handle: BaseWorkerHandle


async def create_trainers(
    args, *, rollout_executor: BaseWorkerHandle, quiet_trainers: QuietTrainers
) -> dict[str, TrainerInfo]:
    deadline = TakeOverDeadline(gate="the trainer state")

    trainers: dict[str, TrainerInfo] = {}
    for trainer_config in compute_trainer_configs(args):
        model_id = trainer_config.model_id
        assert model_id is not None, f"{trainer_config} carries no policy model id"
        created = await create_training_model(
            compute_trainer_args(args, trainer_config),
            trainer_id=trainer_config.trainer_id,
            quiet_trainers=quiet_trainers,
            deadline=deadline,
        )
        trainers[model_id] = TrainerInfo(
            model_id=model_id, start_rollout_id=created.start_rollout_id, handle=created.handle
        )

    for model_id, trainer in trainers.items():
        await rollout_executor.set_train_parallel_config(
            await trainer.handle.get_train_parallel_config(), trainer_model_id=model_id
        )
    leader_model_id = resolve_megatron_config(args).leader_model_id
    await rollout_executor.load(trainers[leader_model_id].start_rollout_id - 1)

    return trainers


def assert_consistent_restore(args, *, trainers: dict[str, TrainerInfo], leader_model_id: str) -> None:
    leader_rollout_id = trainers[leader_model_id].start_rollout_id - 1
    if leader_rollout_id < 0:
        fresh = [model_id for model_id, trainer in trainers.items() if trainer.start_rollout_id != 0]
        assert not fresh, (
            f"the leader policy {leader_model_id!r} starts from scratch, but {fresh} restored a checkpoint; "
            f"the data source and the rollout executor are global, so they would be replayed from zero into "
            f"policies that already trained on them"
        )
        return

    state_dir = args.load or args.save
    assert state_dir is not None, (
        f"the leader policy {leader_model_id!r} restored rollout {leader_rollout_id} without --load or "
        f"--save, so where the other policies stood cannot be read back"
    )
    state = MultiPolicyCheckpointState.load(Path(state_dir), leader_rollout_id=leader_rollout_id)
    assert state is not None, (
        f"resuming at rollout {leader_rollout_id} but {state_dir} holds no record of where the other "
        f"policies stood; the checkpoint was not written by a multi policy run, so the policies cannot be "
        f"proven to resume at consistent positions"
    )
    assert state.leader_model_id == leader_model_id, (
        f"the checkpoint was written with {state.leader_model_id!r} as the leader policy, but this run "
        f"makes {leader_model_id!r} the leader; the global rollout index would change meaning"
    )

    recorded = state.rollout_ids
    restored = {model_id: trainer.start_rollout_id - 1 for model_id, trainer in trainers.items()}
    assert recorded == restored, (
        f"the record at {state_dir} says the policies stood at {recorded}, but this run restored {restored}; "
        f"every policy runs at its own pace, but each one must resume exactly where the global checkpoint "
        f"recorded it"
    )
    logger.info(f"Restored multi policy run at {recorded} (leader {leader_model_id})")


def validate_multi_policy_args(args, *, megatron_config: MegatronConfig) -> None:
    assert megatron_config.is_multi_policy, (
        f"train_multi_policy.py trains several policies, but --megatron-config names "
        f"only {megatron_config.model_ids}; run train_async.py instead"
    )
    assert not args.colocate, "multi policy training does not support --colocate"
    assert args.fully_async, (
        f"multi policy training {megatron_config.model_ids} is only supported for --fully-async: every other rollout "
        f"mode drives one policy per rollout round"
    )
    assert not args.use_critic, "multi policy training does not support --use-critic"
    assert args.eval_interval is None, (
        "train_multi_policy.py does not evaluate: it has no eval dispatcher, so "
        "--eval-interval and the --eval-* arguments beside it would be accepted and never used. Drop them "
        "and read the per policy training curves instead."
    )
    assert args.sglang_config is not None, (
        "multi policy training needs --sglang-config to deploy one inference model per policy, so that a "
        "weight update reaches exactly the engines of its own policy"
    )
    trainable = [model.name for model in resolve_sglang_config(args).models if model.update_weights]
    missing = [model_id for model_id in megatron_config.model_ids if model_id not in trainable]
    assert not missing, (
        f"--megatron-config models {missing} have no matching --sglang-config model with "
        f"update_weights: true (found {trainable}); the names are the same model id on both sides"
    )
    validate_async_off_policy_correction(args)


def define_policy_metric_groups(megatron_config: MegatronConfig) -> None:
    for model_id in megatron_config.model_ids:
        define_step_key_metric_group(prefix=model_id, step_key=f"{model_id}/rollout/step")
        define_step_key_metric_group(prefix=f"{model_id}/train", step_key=f"{model_id}/train/step")
