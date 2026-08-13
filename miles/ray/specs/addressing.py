from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import NamedTuple

from miles.ray.specs.inference import INFERENCE_CONTROLLER_POOL_ID, inference_controller_worker_name
from miles.ray.specs.static_addrs import INFERENCE_CONTROLLER_ADDRS_FLAG, TRAINER_CONTROLLER_ADDRS_FLAG
from miles.ray.specs.train import (
    compute_trainer_controller_pool_id,
    compute_trainer_ids,
    trainer_controller_worker_name,
)
from miles.utils.workers.backend_capability.base import BackendCapability
from miles.utils.workers.types import DeployComponent, DeploySelector
from miles.utils.workers.worker_spec import RPC_PORT_NAME, HostAndPort


class AddressedWorker(NamedTuple):
    flag: str
    key: str | None
    pool_id: str
    worker_name: str
    port_name: str


def compute_addressed_workers(args, *, selector: DeploySelector) -> list[AddressedWorker]:
    if selector.component is DeployComponent.INFERENCE:
        return []

    assert selector.component is DeployComponent.TRAINER, (
        f"only the {DeployComponent.TRAINER.value} deployment is reached by address, not {selector.value}: "
        f"the engines of an {DeployComponent.INFERENCE.value} deployment register themselves, and everything "
        f"else lives with the orchestration script"
    )
    return [
        AddressedWorker(
            flag=TRAINER_CONTROLLER_ADDRS_FLAG,
            key=trainer_id,
            pool_id=compute_trainer_controller_pool_id(trainer_id),
            worker_name=trainer_controller_worker_name(trainer_id),
            port_name=RPC_PORT_NAME,
        )
        for trainer_id in compute_trainer_ids(args)
        if selector.instance is None or selector.instance == trainer_id
    ]


def compute_registration_addressed_workers(args) -> list[AddressedWorker]:
    if args.expected_registration_reporters == 0:
        return []
    return [
        AddressedWorker(
            flag=INFERENCE_CONTROLLER_ADDRS_FLAG,
            key=None,
            pool_id=INFERENCE_CONTROLLER_POOL_ID,
            worker_name=inference_controller_worker_name(),
            port_name=RPC_PORT_NAME,
        )
    ]


async def describe_how_the_run_reaches_this_deployment(args) -> str:
    from miles.ray.wiring import get_backend_capability

    if not (workers := compute_addressed_workers(args, selector=DeploySelector.of(args))):
        return describe_registering_deployment(args)

    capability = get_backend_capability(args)
    addrs = await asyncio.gather(*[_addr_of(capability, worker=worker) for worker in workers])
    return f"Reach it with {format_addressed_workers(list(zip(workers, addrs, strict=True)))}"


def describe_registering_deployment(args) -> str:
    return (
        f"Nothing addresses it: its engines register themselves into the inference controller at "
        f"{args.inference_controller_addrs}, which counts them once every deployment it expects has reported"
    )


def format_addressed_workers(entries: Sequence[tuple[AddressedWorker, HostAndPort]]) -> str:
    values_by_flag: dict[str, list[str]] = {}
    for worker, addr in entries:
        value = f"{addr.host}:{addr.port}" if worker.key is None else f"{worker.key}={addr.host}:{addr.port}"
        values_by_flag.setdefault(worker.flag, []).append(value)
    return " ".join(f"{flag} {' '.join(values)}" for flag, values in values_by_flag.items())


async def _addr_of(capability: BackendCapability, *, worker: AddressedWorker) -> HostAndPort:
    addrs = await capability.static_worker_provider(pool_id=worker.pool_id).get_addrs(worker.worker_name)
    return addrs[worker.port_name]
