from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import NamedTuple

from miles.ray.specs.static_addrs import TRAINER_CONTROLLER_ADDRS_FLAG
from miles.ray.specs.train import (
    compute_trainer_controller_pool_id,
    compute_trainer_ids,
    trainer_controller_worker_name,
)
from miles.utils.workers.backend_capability.base import BackendCapability
from miles.utils.workers.worker_spec import RPC_PORT_NAME, HostAndPort


class AddressedTrainerController(NamedTuple):
    trainer_id: str
    pool_id: str
    worker_name: str


def compute_addressed_trainer_controllers(args) -> list[AddressedTrainerController]:
    return [
        AddressedTrainerController(
            trainer_id=trainer_id,
            pool_id=compute_trainer_controller_pool_id(trainer_id),
            worker_name=trainer_controller_worker_name(trainer_id),
        )
        for trainer_id in compute_trainer_ids(args)
    ]


async def describe_how_the_run_reaches_this_deployment(args) -> str:
    from miles.ray.wiring import get_backend_capability

    capability = get_backend_capability(args)
    controllers = compute_addressed_trainer_controllers(args)
    addrs = await asyncio.gather(*[_addr_of(capability, controller=controller) for controller in controllers])
    return f"Reach it with {format_trainer_controller_addrs(list(zip(controllers, addrs, strict=True)))}"


def format_trainer_controller_addrs(entries: Sequence[tuple[AddressedTrainerController, HostAndPort]]) -> str:
    values = " ".join(f"{controller.trainer_id}={addr.host}:{addr.port}" for controller, addr in entries)
    return f"{TRAINER_CONTROLLER_ADDRS_FLAG} {values}"


async def _addr_of(capability: BackendCapability, *, controller: AddressedTrainerController) -> HostAndPort:
    addrs = await capability.static_worker_provider(pool_id=controller.pool_id).get_addrs(controller.worker_name)
    return addrs[RPC_PORT_NAME]
