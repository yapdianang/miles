from __future__ import annotations

import asyncio
import logging
import time
from argparse import Namespace
from collections.abc import Coroutine
from typing import Any, Protocol, TypeVar

from miles.utils.retry_utils import retry_until_deadline
from miles.utils.workers.rpc.client.misc import RETRYABLE_ERRORS, ServerRestartedError
from miles.utils.workers.worker_handle import BaseWorkerHandle, WorkerUnreachableError

logger = logging.getLogger(__name__)

TAKE_OVER_GATE_TIMEOUT_SECONDS = 600.0
EXECUTOR_FREE_TIMEOUT_SECONDS = 1800.0

_EXECUTOR_POLL_INTERVAL_SECONDS = 5.0

_T = TypeVar("_T")


class ResumableTrainer(Protocol):
    async def is_initialized(self) -> bool: ...

    async def init(self, model_args: Namespace) -> list[Any]: ...

    async def load_state(self) -> list[Any]: ...

    async def wait_idle(self, *, timeout: float) -> None: ...

    async def claim_driver_epoch(self) -> None: ...


class _StillInitializedError(Exception):
    pass


_EXECUTOR_WAITABLE_ERRORS = (_StillInitializedError, WorkerUnreachableError, *RETRYABLE_ERRORS)


class TakeOverDeadline:
    """One budget per gate: everything the gate does shares it, so a stuck take-over cannot stack up budgets."""

    def __init__(self, *, gate: str, seconds: float | None = None) -> None:
        self._gate = gate
        self._seconds = seconds if seconds is not None else TAKE_OVER_GATE_TIMEOUT_SECONDS
        self._expires_at = time.monotonic() + self._seconds

    @property
    def remaining(self) -> float:
        return max(self._expires_at - time.monotonic(), 0.0)

    async def within(self, call: Coroutine[Any, Any, _T], *, action: str) -> _T:
        try:
            return await asyncio.wait_for(call, timeout=self.remaining)
        except (TimeoutError, asyncio.TimeoutError) as e:
            raise TimeoutError(
                f"Taking over {self._gate} ran out of its {self._seconds}s budget while trying to {action} ({e!r}); "
                f"a take-over is a bounded gate, so this run stops instead of driving a system it does not own"
            ) from e


async def quiesce_and_claim_trainer(trainer: ResumableTrainer, *, trainer_id: str, deadline: TakeOverDeadline) -> bool:
    """Gate 1: let a surviving trainer finish the call it is running, then take ownership of it."""
    if resumed := await trainer.is_initialized():
        logger.info(
            f"Trainer {trainer_id!r} is already initialized, so a previous orchestration script built it; "
            f"waiting until it finished whatever it is still running before taking it over"
        )
        await deadline.within(
            trainer.wait_idle(timeout=deadline.remaining),
            action=(
                f"wait until trainer {trainer_id!r} finished the call of the previous orchestration script, "
                f"because reloading a checkpoint into a model a train step is still writing would corrupt it"
            ),
        )

    await trainer.claim_driver_epoch()
    return resumed


async def init_or_load_trainer(
    trainer: ResumableTrainer,
    model_args: Namespace,
    *,
    trainer_id: str,
    resumed: bool,
    deadline: TakeOverDeadline,
) -> list[Any]:
    if not resumed:
        return await trainer.init(model_args)

    start_rollout_ids = await deadline.within(
        trainer.load_state(),
        action=f"roll trainer {trainer_id!r} back to its checkpoint",
    )
    logger.info(f"Resumed the already-initialized trainer {trainer_id!r} at rollout ids {start_rollout_ids}")
    return start_rollout_ids


async def wait_until_rollout_executor_is_free(
    handle: BaseWorkerHandle, *, timeout: float = EXECUTOR_FREE_TIMEOUT_SECONDS
) -> None:
    """Wait until the rollout executor answering us is a fresh one, not the process the previous script drove."""

    async def attempt(remaining: float) -> None:
        try:
            initialized = await handle.is_initialized()
        except ServerRestartedError:
            await handle.wait_ready(timeout=min(remaining, _EXECUTOR_POLL_INTERVAL_SECONDS))
            raise _StillInitializedError("the rollout executor is being replaced right now") from None
        if initialized:
            raise _StillInitializedError("the rollout executor still belongs to the previous orchestration script")

    try:
        await retry_until_deadline(
            attempt,
            total_seconds=timeout,
            retry_on=_EXECUTOR_WAITABLE_ERRORS,
            initial_delay=_EXECUTOR_POLL_INTERVAL_SECONDS,
            max_delay=_EXECUTOR_POLL_INTERVAL_SECONDS,
            log_fields=dict(tag="hot_restart", op="wait_rollout_executor_free"),
        )
    except _EXECUTOR_WAITABLE_ERRORS as e:
        raise TimeoutError(
            f"The rollout executor answering us was still not a fresh process after {timeout}s ({e!r}); a hot "
            f"restart replaces its pod, and everything up to and including that replacement has to fit in this "
            f"budget, so either the pod is not being replaced or the previous script's executor never went away"
        ) from e


async def init_or_reset_inference_controller(inference_controller: BaseWorkerHandle) -> None:
    """Gate 2: initialize the inference side, or reset the one a previous orchestration script left running."""
    if not await inference_controller.is_initialized():
        await inference_controller.init()
        await inference_controller.claim_driver_epoch()
        return

    logger.info("The inference controller outlived a previous orchestration script; taking it over as it is")
    deadline = TakeOverDeadline(gate="the inference controller")
    await deadline.within(
        inference_controller.wait_idle(timeout=deadline.remaining),
        action="wait for the calls of the previous orchestration script to end",
    )
    await _reset_broadcast_lock(inference_controller, deadline=deadline)
    await deadline.within(
        inference_controller.wait_expected_num_cells(timeout=deadline.remaining),
        action="wait for every engine this run expects to be in the fleet it takes over",
    )
    await _abort_inflight_rollouts(inference_controller, deadline=deadline)
    await inference_controller.claim_driver_epoch()


async def _reset_broadcast_lock(inference_controller: BaseWorkerHandle, *, deadline: TakeOverDeadline) -> None:
    """A script that died between start_update_weights and end_update_weights left the lock detached and held."""
    reset = await deadline.within(
        inference_controller.reset_broadcast_lock(),
        action="reset the broadcast lock the previous orchestration script left held",
    )
    if reset:
        logger.warning(
            "The previous orchestration script stopped inside a weight update, so the inference controller was "
            "still holding the broadcast lock that update took; it is released and health checking is resumed, and "
            "the full update-weights pass this script starts with repairs whatever that broadcast half-wrote"
        )


async def _abort_inflight_rollouts(inference_controller: BaseWorkerHandle, *, deadline: TakeOverDeadline) -> None:
    """Stop every generation a previous orchestration script left behind; this run owns the fleet now."""
    refused = await deadline.within(
        inference_controller.abort_all(),
        action="abort the generations the previous orchestration script left in flight",
    )
    assert not refused, (
        f"Cells {sorted(refused)} refused to abort the generations of the previous orchestration script, so its "
        f"samples could still reach this run and be trained on as if this run had produced them; the whole expected "
        f"fleet was already present when the abort ran, so a refusal is a sick engine and not a late arrival, and "
        f"this run stops instead of taking the fleet over dirty"
    )

    logger.info("Aborted every in-flight generation, so this run starts from a quiet inference fleet")
