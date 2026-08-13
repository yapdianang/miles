from __future__ import annotations

import asyncio
import logging
import time
from argparse import Namespace
from typing import Any

import httpx
import pytest

from miles.ray import hot_restart as hot_restart_module
from miles.ray.hot_restart import (
    TakeOverDeadline,
    _abort_inflight_rollouts,
    init_or_load_trainer,
    init_or_reset_inference_controller,
    quiesce_and_claim_trainer,
    wait_until_rollout_executor_is_free,
)
from miles.ray.rollout.inference_controller import InferenceController
from miles.ray.rollout.rollout_executor import RolloutExecutor
from miles.ray.train.group import TrainerController
from miles.utils.workers.rpc.client.misc import ServerRestartedError
from miles.utils.workers.rpc.common.metadata import collect_rpc_method_specs
from miles.utils.workers.worker_handle import WorkerUnreachableError

_MODEL_ID = "policy_a"
_STALLED_SECONDS = 5.0
_SHORT_BUDGET_SECONDS = 0.05


def _deadline(seconds: float = 30.0) -> TakeOverDeadline:
    return TakeOverDeadline(gate="a test gate", seconds=seconds)


class _FakeTrainer:
    def __init__(self, *, initialized: bool, idle_seconds: float = 0.0) -> None:
        self.initialized = initialized
        self.idle_seconds = idle_seconds
        self.calls: list[str] = []
        self.idle_timeouts: list[float] = []
        self.model_ids: list[str] = []

    async def is_initialized(self, model_id: str) -> bool:
        self.calls.append("is_initialized")
        self.model_ids.append(model_id)
        return self.initialized

    async def init(self, model_args: Namespace, model_id: str) -> list[Any]:
        self.calls.append("init")
        self.model_ids.append(model_id)
        return [7]

    async def load_state(self, model_id: str) -> list[Any]:
        self.calls.append("load_state")
        self.model_ids.append(model_id)
        return [3]

    async def wait_idle(self, *, timeout: float) -> None:
        self.calls.append("wait_idle")
        self.idle_timeouts.append(timeout)
        await asyncio.sleep(self.idle_seconds)

    async def claim_driver_epoch(self) -> None:
        self.calls.append("claim_driver_epoch")


class _FakeInferenceController:
    def __init__(
        self,
        *,
        initialized: bool,
        broadcast_lock_held: bool = False,
        busy: bool = False,
        wedged: bool = False,
        cells_refusing_the_abort: list[str] | None = None,
        fleet_incomplete: bool = False,
    ) -> None:
        self.initialized = initialized
        self.broadcast_lock_held = broadcast_lock_held
        self.busy = busy
        self.wedged = wedged
        self.cells_refusing_the_abort = cells_refusing_the_abort or []
        self.fleet_incomplete = fleet_incomplete
        self.calls: list[str] = []
        self.idle_timeouts: list[float] = []

    async def is_initialized(self) -> bool:
        return self.initialized

    async def init(self) -> None:
        self.calls.append("init")

    async def wait_idle(self, *, timeout: float) -> None:
        self.calls.append("wait_idle")
        self.idle_timeouts.append(timeout)
        if self.busy:
            raise TimeoutError("InferenceController was still busy")

    async def reset_broadcast_lock(self) -> bool:
        self.calls.append("reset_broadcast_lock")
        await self._maybe_hang()
        return self.broadcast_lock_held

    async def abort_all(self) -> list[str]:
        self.calls.append("abort_all")
        await self._maybe_hang()
        return self.cells_refusing_the_abort

    async def wait_expected_num_cells(self, timeout: float) -> None:
        self.calls.append("wait_expected_num_cells")
        if self.fleet_incomplete:
            raise TimeoutError("the fleet is short of engines")

    async def claim_driver_epoch(self) -> None:
        self.calls.append("claim_driver_epoch")

    async def _maybe_hang(self) -> None:
        if self.wedged:
            await asyncio.sleep(_STALLED_SECONDS)


class _FakeExecutor:
    def __init__(self, answers: list[bool | Exception], *, ready_error: Exception | None = None) -> None:
        self._answers = list(answers)
        self._ready_error = ready_error
        self.ready_calls = 0

    async def is_initialized(self) -> bool:
        answer = self._answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer

    async def wait_ready(self, *, timeout: float) -> None:
        self.ready_calls += 1
        if self._ready_error is not None:
            raise self._ready_error


@pytest.fixture
def fast_polling(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(hot_restart_module, "_EXECUTOR_POLL_INTERVAL_SECONDS", 0.01)


@pytest.fixture
def short_take_over_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(hot_restart_module, "TAKE_OVER_GATE_TIMEOUT_SECONDS", _SHORT_BUDGET_SECONDS)


class TestGateOneTheTrainerIsQuiet:
    async def test_a_trainer_that_never_ran_is_not_waited_for_but_is_still_claimed(self):
        """A cold start must be untouched by the resume protocol, and still fence out any older driver."""
        trainer = _FakeTrainer(initialized=False)

        assert await quiesce_and_claim_trainer(trainer, model_id=_MODEL_ID, deadline=_deadline()) is False
        assert trainer.calls == ["is_initialized", "claim_driver_epoch"]

    async def test_a_surviving_trainer_is_waited_out_before_it_is_claimed(self):
        """Claiming a trainer that is still running its predecessor's train step would fence that step out mid-flight."""
        trainer = _FakeTrainer(initialized=True)

        assert await quiesce_and_claim_trainer(trainer, model_id=_MODEL_ID, deadline=_deadline()) is True
        assert trainer.calls == ["is_initialized", "wait_idle", "claim_driver_epoch"]

    async def test_the_wait_carries_what_is_left_of_the_gate_budget(self):
        """One budget covers the whole gate, so a multi policy run cannot stack a budget per trainer."""
        trainer = _FakeTrainer(initialized=True)
        deadline = _deadline(seconds=30.0)

        await quiesce_and_claim_trainer(trainer, model_id=_MODEL_ID, deadline=deadline)

        assert 0.0 < trainer.idle_timeouts[0] <= 30.0

    async def test_a_second_trainer_inherits_what_the_first_one_left_of_the_budget(self):
        """The audit's failure was serial 600s waits, one per policy, adding up to an unbounded take-over."""
        deadline = _deadline(seconds=30.0)
        first = _FakeTrainer(initialized=True)
        second = _FakeTrainer(initialized=True)

        await quiesce_and_claim_trainer(first, model_id=_MODEL_ID, deadline=deadline)
        await quiesce_and_claim_trainer(second, model_id="policy_b", deadline=deadline)

        assert second.idle_timeouts[0] <= first.idle_timeouts[0]

    async def test_a_trainer_that_never_goes_idle_fails_loud(self):
        """Taking a trainer over mid-step is exactly the corruption these gates exist to refuse."""
        trainer = _FakeTrainer(initialized=True, idle_seconds=_STALLED_SECONDS)

        started = time.monotonic()
        with pytest.raises(TimeoutError, match="reloading a checkpoint into a model a train step is still writing"):
            await quiesce_and_claim_trainer(trainer, model_id=_MODEL_ID, deadline=_deadline(_SHORT_BUDGET_SECONDS))

        assert time.monotonic() - started < _STALLED_SECONDS
        assert "claim_driver_epoch" not in trainer.calls

    async def test_a_cold_trainer_is_initialized_and_a_resumed_one_only_reloads(self):
        """Init rebuilds a trainer; a survivor must only be rolled back to its checkpoint."""
        cold = _FakeTrainer(initialized=False)
        warm = _FakeTrainer(initialized=True)

        assert await init_or_load_trainer(
            cold, Namespace(), model_id=_MODEL_ID, resumed=False, deadline=_deadline()
        ) == [7]
        assert await init_or_load_trainer(
            warm, Namespace(), model_id=_MODEL_ID, resumed=True, deadline=_deadline()
        ) == [3]
        assert cold.calls == ["init"] and warm.calls == ["load_state"]

    async def test_every_call_names_the_policy_model_it_drives(self):
        """A multi policy run resumes one policy at a time, and an unrouted call reaches the wrong trainer."""
        trainer = _FakeTrainer(initialized=True)

        await quiesce_and_claim_trainer(trainer, model_id=_MODEL_ID, deadline=_deadline())
        await init_or_load_trainer(trainer, Namespace(), model_id=_MODEL_ID, resumed=True, deadline=_deadline())

        assert trainer.model_ids == [_MODEL_ID, _MODEL_ID]


class TestGateTwoTheInferenceSideIsReset:
    async def test_a_fresh_controller_is_initialized_and_claimed(self):
        """A cold start initializes the inference side as it always did, and then owns it."""
        controller = _FakeInferenceController(initialized=False)

        await init_or_reset_inference_controller(controller)

        assert controller.calls == ["init", "claim_driver_epoch"]

    async def test_the_abort_runs_only_once_the_whole_expected_fleet_is_present(self):
        """A cell that was away during the abort would rejoin still generating the previous run's requests."""
        controller = _FakeInferenceController(initialized=True)

        await init_or_reset_inference_controller(controller)

        assert controller.calls == [
            "wait_idle",
            "reset_broadcast_lock",
            "wait_expected_num_cells",
            "abort_all",
            "claim_driver_epoch",
        ]

    async def test_a_broadcast_lock_left_held_is_announced(self, caplog):
        """An operator reading the log has to know the previous script died inside a weight update."""
        controller = _FakeInferenceController(initialized=True, broadcast_lock_held=True)

        with caplog.at_level(logging.WARNING):
            await init_or_reset_inference_controller(controller)

        assert "broadcast lock" in caplog.text

    async def test_a_call_of_the_previous_script_that_never_ends_fails_loud(self):
        """The script that died inside start_update_weights is exactly the case this wait exists for."""
        controller = _FakeInferenceController(initialized=True, busy=True)

        with pytest.raises(TimeoutError, match="calls of the previous orchestration script to end"):
            await init_or_reset_inference_controller(controller)

        assert "abort_all" not in controller.calls

    async def test_the_gate_shares_one_budget_across_all_of_its_steps(self):
        """Four operations of 600s each is 40 minutes, which is not the bounded gate the design promises."""
        controller = _FakeInferenceController(initialized=True)

        await init_or_reset_inference_controller(controller)

        assert controller.idle_timeouts[0] <= hot_restart_module.TAKE_OVER_GATE_TIMEOUT_SECONDS

    async def test_a_controller_that_never_answers_fails_loud(self, short_take_over_budget: None):
        """Hanging here would leave the operator with a silent hot restart that never starts training."""
        controller = _FakeInferenceController(initialized=True, wedged=True)

        started = time.monotonic()
        with pytest.raises(TimeoutError, match="reset the broadcast lock"):
            await init_or_reset_inference_controller(controller)

        assert time.monotonic() - started < _STALLED_SECONDS

    async def test_a_take_over_waits_for_the_whole_fleet_just_as_a_cold_start_does(self):
        """Generating on half a fleet because an engine was being rescheduled is not what the command asked for."""
        controller = _FakeInferenceController(initialized=True, fleet_incomplete=True)

        with pytest.raises(TimeoutError, match="every engine this run expects"):
            await init_or_reset_inference_controller(controller)

    async def test_a_cell_that_refused_the_abort_fails_the_take_over(self):
        """The whole fleet was already there, so a refusal is a sick engine that may still be generating."""
        controller = _FakeInferenceController(initialized=True, cells_refusing_the_abort=["west-engine-0-0-0"])

        with pytest.raises(AssertionError, match="west-engine-0-0-0"):
            await init_or_reset_inference_controller(controller)

        assert "claim_driver_epoch" not in controller.calls


class TestAbortInflightRollouts:
    async def test_a_refusing_cell_stops_the_run_instead_of_being_logged_past(self):
        """A cell that kept generating pollutes this run's data, so the take-over cannot continue over it."""
        controller = _FakeInferenceController(initialized=True, cells_refusing_the_abort=["west-engine-0-0-0"])

        with pytest.raises(AssertionError, match="refused to abort"):
            await _abort_inflight_rollouts(controller, deadline=_deadline())

    async def test_a_fleet_that_answered_every_abort_is_announced_quiet(self, caplog):
        """The ordinary take-over says so, and an operator reads that line as a fleet with no request left on it."""
        controller = _FakeInferenceController(initialized=True)

        with caplog.at_level(logging.INFO):
            await _abort_inflight_rollouts(controller, deadline=_deadline())

        assert "quiet inference fleet" in caplog.text


class TestWaitUntilRolloutExecutorIsFree:
    async def test_a_fresh_executor_is_accepted_at_once(self):
        """The normal case must not pay a polling delay."""
        executor = _FakeExecutor([False])

        await wait_until_rollout_executor_is_free(executor, timeout=5.0)

    async def test_an_executor_of_the_previous_script_is_waited_out(self, fast_polling: None):
        """Initializing the old executor a second time would drive the process that is about to die."""
        executor = _FakeExecutor([True, True, False])

        await wait_until_rollout_executor_is_free(executor, timeout=5.0)

    async def test_an_executor_being_replaced_mid_wait_is_re_baselined(self, fast_polling: None):
        """The executor is expected to restart during this wait, so its boot uuid change is not a violation."""
        executor = _FakeExecutor([True, ServerRestartedError("replaced"), False])

        await wait_until_rollout_executor_is_free(executor, timeout=5.0)

        assert executor.ready_calls == 1

    async def test_an_executor_that_never_frees_up_times_out(self, fast_polling: None):
        """A new script must not silently share a run with the executor of its predecessor."""
        executor = _FakeExecutor([True] * 100)

        with pytest.raises(TimeoutError, match="not a fresh process"):
            await wait_until_rollout_executor_is_free(executor, timeout=_SHORT_BUDGET_SECONDS)

    async def test_an_executor_pod_being_recreated_is_waited_out(self, fast_polling: None):
        """Replacing the pod is the whole point, and it is unreachable for as long as that takes."""
        executor = _FakeExecutor([True, WorkerUnreachableError("pod is gone"), False])

        await wait_until_rollout_executor_is_free(executor, timeout=5.0)

    async def test_a_transport_error_is_waited_out_too(self, fast_polling: None):
        """A connection refused while kubernetes reschedules the pod is the expected state, not a failure."""
        executor = _FakeExecutor([httpx.ConnectError("refused"), False])

        await wait_until_rollout_executor_is_free(executor, timeout=5.0)

    async def test_an_executor_that_stays_unreachable_reports_a_timeout(self, fast_polling: None):
        """The message has to say the executor never came back, not leak a transport error."""
        executor = _FakeExecutor([WorkerUnreachableError("pod is gone")] * 100)

        with pytest.raises(TimeoutError, match="not a fresh process"):
            await wait_until_rollout_executor_is_free(executor, timeout=_SHORT_BUDGET_SECONDS)

    async def test_a_readiness_wait_that_fails_is_retried_rather_than_fatal(self, fast_polling: None):
        """wait_ready itself throws while the replacement pod is still being scheduled."""
        executor = _FakeExecutor([ServerRestartedError("replaced"), False], ready_error=WorkerUnreachableError("no"))

        await wait_until_rollout_executor_is_free(executor, timeout=5.0)


class TestTheTakeOverSurfaceCrossesTheWire:
    @pytest.mark.parametrize(
        "worker_cls, methods",
        [
            (TrainerController, {"is_initialized", "load_state"}),
            (RolloutExecutor, {"is_initialized"}),
            (InferenceController, {"is_initialized", "reset_broadcast_lock", "abort_all", "wait_expected_num_cells"}),
        ],
    )
    def test_the_take_over_surface_is_exposed_over_rpc(self, worker_cls: type, methods: set[str]):
        """A restarted orchestration script drives the whole take-over through exactly these rpc methods."""
        assert methods <= set(collect_rpc_method_specs(worker_cls))
