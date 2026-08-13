from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import os
import random

from miles.utils.async_utils import AsyncLoopThread
from miles.utils.workers.naming import compute_cell_id, compute_worker_name, parse_cell_id, parse_worker_name
from miles.utils.workers.registration.models import (
    CONTROLLER_READY_TIMEOUT_SECONDS,
    SNAPSHOT_DEBOUNCE_SECONDS,
    SNAPSHOT_INTERVAL_SECONDS,
    SNAPSHOT_JITTER_RATIO,
    SUPPORTED_WORKER_TYPE,
    RegisteredCell,
    RegisteredWorker,
    RegistrationReporterStatus,
    RegistrationSnapshot,
    compute_snapshot_digest,
)
from miles.utils.workers.worker_handle import BaseWorkerHandle
from miles.utils.workers.worker_info import WorkerInfo
from miles.utils.workers.worker_provider.base import BaseWorkerProvider, CellInfo

logger = logging.getLogger(__name__)


class RegistrationReporter:
    def __init__(
        self,
        *,
        reporter_id: str,
        controller: BaseWorkerHandle,
        engine_provider: BaseWorkerProvider,
        expected_num_cells_by_model: dict[str, int],
        token: str | None = None,
        interval_seconds: float = SNAPSHOT_INTERVAL_SECONDS,
        jitter_ratio: float = SNAPSHOT_JITTER_RATIO,
        debounce_seconds: float = SNAPSHOT_DEBOUNCE_SECONDS,
        rng: random.Random | None = None,
    ) -> None:
        self._reporter_id = reporter_id
        self._controller = controller
        self._engine_provider = engine_provider
        self._expected_num_cells_by_model = dict(expected_num_cells_by_model)
        self._token = token
        self._interval_seconds = interval_seconds
        self._jitter_ratio = jitter_ratio
        self._debounce_seconds = debounce_seconds
        self._rng = rng if rng is not None else random.Random()
        self._observed: dict[str, CellInfo] = {}
        self._changed = asyncio.Event()
        self._has_synced = False
        self._sequence = 0
        self._acknowledged_digest: str | None = None

    async def run(self) -> None:
        await self._controller.wait_ready(timeout=CONTROLLER_READY_TIMEOUT_SECONDS)
        stop_watch = await self._engine_provider.watch_cells(self._observe)
        self._has_synced = True

        try:
            logger.info(
                f"Reporter {self._reporter_id} observes {len(self._observed)} cells of its own deployment and "
                f"reports them every {self._interval_seconds}s"
            )
            while True:
                await self._wait_next_send()
                try:
                    await self.send_once()
                except Exception:
                    logger.warning(f"Reporting the cells of {self._reporter_id} failed", exc_info=True)
        finally:
            await stop_watch()

    async def send_once(self) -> None:
        assert self._has_synced, (
            f"reporter {self._reporter_id} has not finished its first look at its own deployment, and an empty "
            f"snapshot would drop every cell of this deployment from the run"
        )

        snapshot = self._compute_snapshot()
        ack = await asyncio.wait_for(
            self._controller.apply_registration_snapshot(snapshot=snapshot), timeout=self._interval_seconds
        )
        if ack.excluded_cell_ids:
            logger.error(
                f"The run refused {len(ack.excluded_cell_ids)} of the cells reporter {self._reporter_id} reported "
                f"({ack.excluded_cell_ids}); they serve no request of this run, and the run's own log says why"
            )
        self._acknowledged_digest = ack.applied_digest

    def compute_status(self) -> RegistrationReporterStatus:
        return RegistrationReporterStatus(
            reporter_id=self._reporter_id,
            sequence=self._sequence,
            num_observed_cells=len(self._observed),
            acknowledged_digest=self._acknowledged_digest,
        )

    def _assert_every_cell_is_regular(self) -> None:
        offenders = {
            cell_id: worker_type
            for cell_id, info in sorted(self._observed.items())
            if (worker_type := info.meta.get("worker_type")) != SUPPORTED_WORKER_TYPE
        }
        assert not offenders, (
            f"reporter {self._reporter_id} observes {offenders} in its own deployment, and pairing a prefill "
            f"engine of one deployment with a decode engine of another needs a router-side pairing policy that "
            f"does not exist yet, so this deployment cannot register its engines into another run"
        )

    async def _observe(self, cell_id: str, observed: CellInfo | None) -> None:
        if observed is None:
            self._observed.pop(cell_id, None)
        else:
            self._observed[cell_id] = observed
        self._changed.set()

    def _compute_snapshot(self) -> RegistrationSnapshot:
        self._assert_every_cell_is_regular()
        cells = self._compute_cells()
        digest = compute_snapshot_digest(cells=cells, expected_num_cells_by_model=self._expected_num_cells_by_model)
        self._sequence += 1
        return RegistrationSnapshot(
            reporter_id=self._reporter_id,
            sequence=self._sequence,
            digest=digest,
            expected_num_cells_by_model=self._expected_num_cells_by_model,
            token=self._token,
            cells=None if digest == self._acknowledged_digest else cells,
        )

    def _compute_cells(self) -> list[RegisteredCell]:
        observed = sorted(self._observed.items())
        infos_per_cell = self._engine_provider.get_worker_infos(cell_ids=[cell_id for cell_id, _ in observed])
        return [
            self._compute_cell(info, worker_infos=worker_infos)
            for (_cell_id, info), worker_infos in zip(observed, infos_per_cell, strict=True)
        ]

    def _compute_cell(self, info: CellInfo, *, worker_infos: list[WorkerInfo]) -> RegisteredCell:
        pool_id = f"{self._reporter_id}-{info.pool_id}"
        cell_index = parse_cell_id(info.cell_id).cell_index
        indexed_workers = sorted(
            ((parse_worker_name(one.name)[2], one) for one in worker_infos), key=lambda pair: pair[0]
        )
        return RegisteredCell(
            cell_id=compute_cell_id(pool_id=pool_id, cell_index=cell_index),
            pool_id=pool_id,
            workers_hash=info.workers_hash,
            workers=[
                RegisteredWorker(
                    name=compute_worker_name(
                        pool_id=pool_id, cell_index=cell_index, worker_in_cell_index=worker_in_cell_index
                    ),
                    addrs=dict(worker_info.self_addrs),
                    gpu_ids=list(worker_info.gpu_ids),
                )
                for worker_in_cell_index, worker_info in indexed_workers
            ],
            meta=dict(info.meta),
        )

    async def _wait_next_send(self) -> None:
        try:
            await asyncio.wait_for(self._changed.wait(), timeout=self._compute_next_interval_seconds())
        except TimeoutError:
            return
        self._changed.clear()
        await asyncio.sleep(self._debounce_seconds)
        self._changed.clear()

    def _compute_next_interval_seconds(self) -> float:
        return self._interval_seconds * (1.0 + self._rng.uniform(-self._jitter_ratio, self._jitter_ratio))


class RegistrationReporterWorker:
    def __init__(self, *, reporter: RegistrationReporter) -> None:
        self._reporter = reporter
        self._loop_thread = AsyncLoopThread()
        self._loop_thread.submit(reporter.run()).add_done_callback(_exit_because_the_reporter_stopped)

    async def get_registration_status(self) -> RegistrationReporterStatus:
        return self._reporter.compute_status()


def _exit_because_the_reporter_stopped(future: concurrent.futures.Future[None]) -> None:
    logger.error(
        "The registration reporter of this deployment stopped, so the deployment exits with it: nothing else here "
        "registers these engines into the run, and a pod that keeps running would look healthy while the run waits "
        "for cells that are never announced",
        exc_info=future.exception(),
    )
    os._exit(1)
