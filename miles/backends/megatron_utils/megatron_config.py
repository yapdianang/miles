import argparse
import copy
import logging
import re
from argparse import Namespace
from pathlib import Path
from typing import Any

import pydantic
import yaml

from miles.utils.file_arg_utils import resolve_file_arg
from miles.utils.pydantic_utils import FrozenStrictBaseModel
from miles.utils.workers.argv_utils import compute_arg_types
from miles.utils.workers.worker_provider.kubernetes.helm.naming import COMPONENT_NAME_BUDGET, RELEASE_DIGEST_LENGTH

logger = logging.getLogger(__name__)


# ---------------------------- constants -----------------------------

ACTOR_ROLE = "actor"
CRITIC_ROLE = "critic"
DEFAULT_MODEL_ROLE = ACTOR_ROLE
TRAINER_CHECKPOINT_DIRNAME = "trainers"
MODEL_ID_PATTERN = re.compile(r"\A[a-z0-9]([a-z0-9-]*[a-z0-9])?\Z")
LONGEST_TRAINER_POOL_PREFIX = "trainer-controller-"
MIN_RELEASE_PREFIX_LENGTH = RELEASE_DIGEST_LENGTH + len("x-")
TRAINER_ID_MAX_LENGTH = COMPONENT_NAME_BUDGET - MIN_RELEASE_PREFIX_LENGTH - len("-") - len(LONGEST_TRAINER_POOL_PREFIX)
MODEL_ID_MAX_LENGTH = TRAINER_ID_MAX_LENGTH - len("-") - len(DEFAULT_MODEL_ROLE)

PER_POLICY_ARGS: frozenset[str] = frozenset(
    {
        "hf_checkpoint",
        "ref_load",
        "megatron_to_hf_mode",
        "num_layers",
        "hidden_size",
        "ffn_hidden_size",
        "num_attention_heads",
        "group_query_attention",
        "num_query_groups",
        "kv_channels",
        "add_qkv_bias",
        "qk_layernorm",
        "swiglu",
        "normalization",
        "norm_epsilon",
        "disable_bias_linear",
        "use_rotary_position_embeddings",
        "rotary_base",
        "vocab_size",
        "optimizer",
        "lr",
        "min_lr",
        "lr_decay_style",
        "lr_warmup_iters",
        "lr_warmup_fraction",
        "weight_decay",
        "adam_beta1",
        "adam_beta2",
        "clip_grad",
        "tensor_model_parallel_size",
        "pipeline_model_parallel_size",
        "context_parallel_size",
        "expert_model_parallel_size",
        "expert_tensor_parallel_size",
        "sequence_parallel",
        "global_batch_size",
        "micro_batch_size",
        "max_tokens_per_gpu",
        "use_dynamic_batch_size",
        "advantage_estimator",
        "use_kl_loss",
        "kl_loss_coef",
        "kl_loss_type",
        "entropy_coef",
        "eps_clip",
        "eps_clip_high",
    }
)


# ---------------------------- raw config -----------------------------


class _RawMegatronTrainerConfig(FrozenStrictBaseModel):
    model_id: str
    role: str = DEFAULT_MODEL_ROLE
    trainer_id: str | None = None
    overrides: dict[str, Any] = {}


class _RawMegatronConfig(FrozenStrictBaseModel):
    trainers: list[_RawMegatronTrainerConfig] = pydantic.Field(
        validation_alias=pydantic.AliasChoices("trainers", "megatron")
    )

    @classmethod
    def from_file_arg(cls, value: str) -> "_RawMegatronConfig":
        return cls.model_validate(yaml.safe_load(resolve_file_arg(value)))


# ---------------------------- resolved config -----------------------------


class MegatronTrainerConfig(FrozenStrictBaseModel):
    trainer_id: str
    model_id: str | None
    role: str
    overrides: dict[str, Any]

    @classmethod
    def resolve(cls, raw: _RawMegatronTrainerConfig) -> "MegatronTrainerConfig":
        return cls(
            trainer_id=raw.trainer_id if raw.trainer_id is not None else f"{raw.model_id}-{raw.role}",
            model_id=raw.model_id,
            role=raw.role,
            overrides=_resolve_overrides(raw.overrides, model_id=raw.model_id),
        )


class MegatronConfig(FrozenStrictBaseModel):
    trainers: list[MegatronTrainerConfig]

    @pydantic.model_validator(mode="after")
    def _validate_ids(self) -> "MegatronConfig":
        _assert_valid_ids(self.model_ids, kind="model", max_length=MODEL_ID_MAX_LENGTH)
        _assert_valid_trainer_ids([t.trainer_id for t in self.trainers])
        return self

    @property
    def model_ids(self) -> list[str]:
        return list(dict.fromkeys(t.model_id for t in self.trainers if t.model_id is not None))

    @property
    def leader_model_id(self) -> str | None:
        return self.trainers[0].model_id

    @property
    def is_multi_policy(self) -> bool:
        return len(set(self.model_ids)) > 1

    def get(self, model_id: str) -> MegatronTrainerConfig:
        for trainer in self.trainers:
            if trainer.model_id == model_id:
                return trainer
        raise KeyError(f"Unknown trainer model id {model_id!r}, known ids: {self.model_ids}")


def resolve_megatron_config(args) -> MegatronConfig:
    return MegatronConfig(trainers=_compute_trainers(args))


def _compute_trainers(args) -> list[MegatronTrainerConfig]:
    if (raw := _resolve_raw_megatron_config(args.megatron_config)) is None:
        trainers = [MegatronTrainerConfig(trainer_id=ACTOR_ROLE, model_id=None, role=ACTOR_ROLE, overrides={})]
    else:
        trainers = [MegatronTrainerConfig.resolve(raw=t) for t in raw.trainers]
        assert trainers, "--megatron-config must declare at least one trainer"

    if getattr(args, "use_critic", False) and not any(trainer.role == CRITIC_ROLE for trainer in trainers):
        assert (
            len({trainer.model_id for trainer in trainers}) == 1
        ), "training several policy models does not support --use-critic"
        trainers = [*trainers, _compute_critic_trainer(args, policy=trainers[0])]

    return trainers


def _compute_critic_trainer(args, *, policy: MegatronTrainerConfig) -> MegatronTrainerConfig:
    model_id = policy.model_id
    return MegatronTrainerConfig(
        trainer_id=CRITIC_ROLE if model_id is None else f"{model_id}-{CRITIC_ROLE}",
        model_id=model_id,
        role=CRITIC_ROLE,
        overrides={**policy.overrides, **_compute_critic_overrides(args)},
    )


def _compute_critic_overrides(args) -> dict[str, Any]:
    return {
        "kl_coef": 0,
        "use_opd": False,
        "disable_param_buffers_cpu_backup": False,
        "load": args.critic_load,
        "save": args.critic_save,
        "lr": args.critic_lr,
        "lr_warmup_iters": args.critic_lr_warmup_iters,
    }


def _resolve_raw_megatron_config(value: str | None) -> "_RawMegatronConfig | None":
    if value is None:
        return None
    return _RawMegatronConfig.from_file_arg(value)


# ---------------------------- per policy args -----------------------------


def compute_trainer_args(args, trainer: MegatronTrainerConfig) -> Namespace:
    megatron_config = resolve_megatron_config(args)
    ans = copy.deepcopy(args)
    ans.trainer_model_id = trainer.model_id if megatron_config.is_multi_policy else None

    for key, value in trainer.overrides.items():
        assert hasattr(ans, key), (
            f"--megatron-config trainer {trainer.trainer_id!r} overrides {key!r}, which this run's argument "
            f"parser does not know"
        )
        setattr(ans, key, value)

    _apply_critical_derived_overrides(ans, base=args, trainer=trainer)

    if megatron_config.is_multi_policy:
        from miles.utils.arguments import resolve_checkpoint_source

        trainer_id = trainer.trainer_id
        requested = dict(ans.requested_checkpoint_source)
        requested["load"] = _compute_trainer_checkpoint_dir(base_dir=requested["load"], trainer_id=trainer_id)
        requested["save"] = _compute_trainer_checkpoint_dir(base_dir=requested["save"], trainer_id=trainer_id)
        requested["critic_save"] = _compute_trainer_checkpoint_dir(
            base_dir=requested["critic_save"], trainer_id=trainer_id
        )
        ans.requested_checkpoint_source = requested
        resolve_checkpoint_source(ans)
    return ans


def _apply_critical_derived_overrides(ans: Namespace, *, base, trainer: MegatronTrainerConfig) -> None:
    # TODO: most derived defaults are still computed from the base args; revisit after the arguments refactor
    if "hf_checkpoint" in trainer.overrides and base.tokenizer_model == base.hf_checkpoint:
        ans.tokenizer_model = ans.hf_checkpoint


# ---------------------------- checkpoint dirs -----------------------------


def _compute_trainer_checkpoint_dir(*, base_dir: str | None, trainer_id: str) -> str | None:
    if base_dir is None:
        return None
    return str(Path(base_dir) / TRAINER_CHECKPOINT_DIRNAME / trainer_id)


# ---------------------------- validation -----------------------------


def _assert_valid_trainer_ids(trainer_ids: list[str]) -> None:
    assert len(set(trainer_ids)) == len(trainer_ids), (
        f"--megatron-config trainer ids must be unique, got {trainer_ids}; a trainer id addresses one trainer "
        f"controller and its engine pool, so two entries sharing it would land in the same pool"
    )
    _assert_valid_ids(trainer_ids, kind="trainer", max_length=TRAINER_ID_MAX_LENGTH)


def _assert_valid_ids(ids: list[str], *, kind: str, max_length: int) -> None:
    bad_ids = [identifier for identifier in ids if MODEL_ID_PATTERN.match(identifier) is None]
    assert not bad_ids, (
        f"--megatron-config {kind} ids {bad_ids} are not usable as Kubernetes pool names or path components: "
        f"each must match {MODEL_ID_PATTERN.pattern}"
    )
    long_ids = [identifier for identifier in ids if len(identifier) > max_length]
    assert not long_ids, f"--megatron-config {kind} ids {long_ids} are longer than {max_length} characters"


# ---------------------------- override coercion -----------------------------


def _resolve_overrides(overrides: dict[str, Any], *, model_id: str) -> dict[str, Any]:
    if not overrides:
        return overrides

    arg_types = compute_arg_types(get_megatron_arg_parser())
    return {
        key: _coerce_override_value(value, key=key, model_id=model_id, arg_types=arg_types)
        for key, value in overrides.items()
    }


def _coerce_override_value(value: Any, *, key: str, model_id: str, arg_types: dict[str, type]) -> Any:
    assert key in PER_POLICY_ARGS, (
        f"--megatron-config model {model_id!r} overrides {key!r}, which is not a per-policy argument; "
        f"only these may differ between policies: {sorted(PER_POLICY_ARGS)}. Everything else is read "
        f"from the base command line, so overriding it here would be silently ignored"
    )
    assert value is not None, f"--megatron-config model {model_id!r} overrides {key!r} with no value"

    expected = arg_types.get(key)
    assert expected is not None, (
        f"{key!r} is a per-policy argument, but the argument parser declares no such argument, so the "
        f"value {value!r} of model {model_id!r} cannot be typed"
    )

    if expected is bool:
        assert isinstance(
            value, bool
        ), f"--megatron-config model {model_id!r} overrides {key!r} with {value!r}, which is not a boolean"
        return value

    assert not isinstance(value, bool) and isinstance(value, (int, float, str)), (
        f"--megatron-config model {model_id!r} overrides {key!r} with {value!r}, which is not a "
        f"{expected.__name__}"
    )
    return expected(value)


def get_megatron_arg_parser() -> argparse.ArgumentParser:
    # TODO: revisit once the args refactor lands; this throwaway construction may then be optimized
    from miles.backends.megatron_utils.arguments import parse_args
    from miles.utils.arguments import get_miles_extra_args_provider

    class ParserCaptured(Exception):
        def __init__(self, parser: argparse.ArgumentParser) -> None:
            super().__init__("the throwaway parser was captured before anything was parsed")
            self.parser = parser

    def capture(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        raise ParserCaptured(get_miles_extra_args_provider()(parser))

    try:
        parse_args(extra_args_provider=capture)
    except ParserCaptured as captured:
        return captured.parser
    raise AssertionError(
        "megatron's parse_args returned without calling the extra args provider, so the arguments this "
        "run declares could not be read; --megatron-config overrides cannot be typed"
    )
