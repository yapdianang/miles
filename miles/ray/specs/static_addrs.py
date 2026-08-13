from __future__ import annotations

from miles.utils.workers.types import DeployComponent, DeploymentIdentity
from miles.utils.workers.worker_provider.static import parse_host_and_port
from miles.utils.workers.worker_spec import HostAndPort

TRAINER_CONTROLLER_ADDRS_FLAG = "--trainer-controller-addrs"
INFERENCE_CONTROLLER_ADDRS_FLAG = "--inference-controller-addrs"


def inference_controller_urls(args) -> list[str] | None:
    if (entries := args.inference_controller_addrs) is None:
        return None
    assert len(entries) == 1, (
        f"{INFERENCE_CONTROLLER_ADDRS_FLAG} names {len(entries)} controllers, but a run holds exactly one of them, "
        f"and every engine deployment of that run registers its cells into that one"
    )
    return list(entries)


def trainer_controller_url(args, *, trainer_id: str, trainer_ids: list[str]) -> str | None:
    if (entries := args.trainer_controller_addrs) is None:
        return None
    urls = _group_by_trainer_id(entries, trainer_ids=trainer_ids).get(trainer_id, [])
    assert len(urls) == 1, (
        f"{TRAINER_CONTROLLER_ADDRS_FLAG} names {len(urls)} controllers for trainer {trainer_id!r}, but a run "
        f"drives exactly one of them"
    )
    return urls[0]


def static_trainer_controller_addrs(args, *, trainer_ids: list[str]) -> list[HostAndPort]:
    return [
        parse_host_and_port(url)
        for trainer_id in trainer_ids
        if (url := trainer_controller_url(args, trainer_id=trainer_id, trainer_ids=trainer_ids)) is not None
    ]


def assert_deployment_is_this_runs_trainer(identity: DeploymentIdentity, *, args, instance: str | None = None) -> None:
    assert identity.run_uuid == args.run_uuid, (
        f"{TRAINER_CONTROLLER_ADDRS_FLAG} names the {identity.deploy_component} deployment of run "
        f"{identity.run_uuid}, but this launch drives run {args.run_uuid}: every deployment a split run reaches has "
        f"to be a deployment of that same run, or its weight updates and its rollout samples belong to different runs"
    )
    assert identity.deploy_component == DeployComponent.TRAINER.value, (
        f"{TRAINER_CONTROLLER_ADDRS_FLAG} names the {identity.deploy_component} deployment of run "
        f"{identity.run_uuid}, and only a deployment that carries nothing but the trainer is reached by address: "
        f"an {DeployComponent.ALL.value} release of this run runs an orchestration script of its own, so both "
        f"scripts would drive the same trainer"
    )
    assert instance is None or identity.deploy_instance == instance, (
        f"{TRAINER_CONTROLLER_ADDRS_FLAG} names the {identity.deploy_instance!r} of run {identity.run_uuid} as its "
        f"{instance!r}, so this launch would drive the ranks of one trainer through the workflow of another; "
        f"the entries of {TRAINER_CONTROLLER_ADDRS_FLAG} are keyed by trainer id, and two of them are swapped"
    )


def _group_by_trainer_id(entries: list[str], *, trainer_ids: list[str]) -> dict[str, list[str]]:
    prefixed = [split for entry in entries if (split := _split_trainer_id(entry)) is not None]
    assert len(prefixed) in (0, len(entries)), (
        f"{TRAINER_CONTROLLER_ADDRS_FLAG} must be uniformly bare addresses or uniformly '<trainer_id>=<address>' "
        f"entries (got {entries})"
    )

    if not prefixed:
        assert (
            len(entries) == 1
        ), f"{TRAINER_CONTROLLER_ADDRS_FLAG} takes bare addresses only when one entry is expected (got {entries})"
        return {trainer_ids[0]: list(entries)}

    grouped: dict[str, list[str]] = {}
    for trainer_id, addr in prefixed:
        assert (
            trainer_id in trainer_ids
        ), f"{TRAINER_CONTROLLER_ADDRS_FLAG} names {trainer_id!r}, which is not one of {trainer_ids}"
        grouped.setdefault(trainer_id, []).append(addr)
    return grouped


def _split_trainer_id(entry: str) -> tuple[str, str] | None:
    prefix, separator, rest = entry.partition("=")
    if not separator or "://" in prefix or ":" in prefix or "/" in prefix:
        return None
    return prefix, rest
