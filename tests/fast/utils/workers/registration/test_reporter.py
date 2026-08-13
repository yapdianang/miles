from __future__ import annotations

import asyncio
import random

import pytest

from miles.utils.workers.registration.models import RegistrationAck, RegistrationSnapshot
from miles.utils.workers.registration.reporter import RegistrationReporter, RegistrationReporterWorker
from miles.utils.workers.rpc.common.metadata import collect_rpc_method_specs
from miles.utils.workers.worker_info import WorkerInfo
from miles.utils.workers.worker_provider.base import BaseWorkerProvider, CellInfo, ReconcileFn, StopWatchFn
from miles.utils.workers.worker_spec import HostAndPort, NamedHostAndPorts

_POOL_ID = "inference-engine-0-0"
_REPORTER_ID = "miles-run-r1-inference"


class _FakeEngineProvider(BaseWorkerProvider):
    def __init__(self, *, cell_indices: list[int], worker_type: str = "regular") -> None:
        self.cell_indices = list(cell_indices)
        self.worker_type = worker_type
        self.reconcile: ReconcileFn | None = None
        self.stopped = False

    async def get_addrs(self, worker_name: str) -> NamedHostAndPorts:
        raise NotImplementedError

    def get_worker_infos(self, *, cell_ids: list[str]) -> list[list[WorkerInfo]]:
        return [
            [
                WorkerInfo(
                    name=f"{cell_id}-0",
                    generation=0,
                    self_addrs={"primary": HostAndPort(host="10.0.0.5", port=8000)},
                    gpu_ids=[0],
                    worker_class=None,
                )
            ]
            for cell_id in cell_ids
        ]

    async def watch_cells(self, reconcile: ReconcileFn) -> StopWatchFn:
        self.reconcile = reconcile
        for cell_index in self.cell_indices:
            await reconcile(f"{_POOL_ID}-{cell_index}", _cell_info(cell_index, worker_type=self.worker_type))

        async def _stop() -> None:
            self.stopped = True

        return _stop


class _FakeController:
    def __init__(self, *, acks: list[RegistrationAck] | None = None) -> None:
        self.snapshots: list[RegistrationSnapshot] = []
        self.ready_timeouts: list[float] = []
        self._acks = list(acks or [])

    async def wait_ready(self, *, timeout: float) -> None:
        self.ready_timeouts.append(timeout)

    async def apply_registration_snapshot(self, *, snapshot: RegistrationSnapshot) -> RegistrationAck:
        self.snapshots.append(snapshot)
        if self._acks:
            return self._acks.pop(0)
        return RegistrationAck(applied_digest=snapshot.digest)


def _cell_info(cell_index: int, *, workers_hash: str = "hash-1", worker_type: str = "regular") -> CellInfo:
    return CellInfo(
        cell_id=f"{_POOL_ID}-{cell_index}",
        pool_id=_POOL_ID,
        alive=True,
        worker_names=[f"{_POOL_ID}-{cell_index}-0"],
        workers_hash=workers_hash,
        meta=dict(model_id="default", worker_type=worker_type),
    )


def _reporter(*, provider: _FakeEngineProvider, controller: _FakeController, token: str | None = None):
    return RegistrationReporter(
        reporter_id=_REPORTER_ID,
        controller=controller,
        engine_provider=provider,
        expected_num_cells_by_model={"default": 2},
        token=token,
        rng=random.Random(0),
    )


async def _synced(
    *,
    cell_indices: tuple[int, ...] = (0,),
    worker_type: str = "regular",
    controller: _FakeController | None = None,
    token: str | None = None,
) -> tuple[RegistrationReporter, _FakeEngineProvider, _FakeController]:
    provider = _FakeEngineProvider(cell_indices=list(cell_indices), worker_type=worker_type)
    controller = controller if controller is not None else _FakeController()
    reporter = _reporter(provider=provider, controller=controller, token=token)
    await provider.watch_cells(reporter._observe)
    reporter._has_synced = True
    return reporter, provider, controller


class TestSnapshotContents:
    async def test_a_snapshot_carries_every_cell_of_this_deployment(self):
        """The controller replaces this deployment's membership with the snapshot, so it has to be the whole one."""
        reporter, _provider, controller = await _synced(cell_indices=(0, 1))

        await reporter.send_once()

        (snapshot,) = controller.snapshots
        assert [cell.cell_id for cell in snapshot.cells] == [
            f"{_REPORTER_ID}-{_POOL_ID}-0",
            f"{_REPORTER_ID}-{_POOL_ID}-1",
        ]

    async def test_the_pool_ids_it_reports_are_namespaced_by_this_deployment(self):
        """Two deployments run the same pool ids, and one would otherwise remove the cells of the other."""
        reporter, _provider, controller = await _synced()

        await reporter.send_once()

        (cell,) = controller.snapshots[0].cells
        assert cell.pool_id == f"{_REPORTER_ID}-{_POOL_ID}"
        assert [worker.name for worker in cell.workers] == [f"{_REPORTER_ID}-{_POOL_ID}-0-0"]

    async def test_it_reports_the_addresses_its_own_deployment_observed(self):
        """The run calls the engine there, so an address computed anywhere else would be a guess."""
        reporter, _provider, controller = await _synced()

        await reporter.send_once()

        (cell,) = controller.snapshots[0].cells
        assert cell.workers[0].addrs["primary"] == HostAndPort(host="10.0.0.5", port=8000)

    async def test_it_presents_the_token_both_sides_were_given(self):
        """The controller refuses a snapshot whose token it does not know."""
        reporter, _provider, controller = await _synced(token="secret")

        await reporter.send_once()

        assert controller.snapshots[0].token == "secret"

    async def test_it_reports_how_many_cells_this_deployment_expects(self):
        """The run waits for the cells of every deployment, and only that deployment knows how many it brings."""
        reporter, _provider, controller = await _synced()

        await reporter.send_once()

        assert controller.snapshots[0].expected_num_cells_by_model == {"default": 2}

    async def test_it_refuses_to_register_a_prefill_or_decode_deployment(self):
        """Pairing engines across deployments needs a router-side policy that does not exist yet."""
        reporter, _provider, _controller = await _synced(worker_type="prefill")

        with pytest.raises(AssertionError, match="prefill"):
            await reporter.send_once()

    async def test_it_never_sends_a_snapshot_before_it_has_looked_at_its_own_deployment(self):
        """An empty first snapshot would drop every cell of this deployment from the run."""
        reporter = _reporter(provider=_FakeEngineProvider(cell_indices=[0]), controller=_FakeController())

        with pytest.raises(AssertionError, match="first look"):
            await reporter.send_once()


class TestSnapshotSequencing:
    async def test_every_snapshot_carries_a_higher_sequence_than_the_last(self):
        """The controller drops a snapshot that arrived late, and only the sequence tells it which one that is."""
        reporter, _provider, controller = await _synced()

        await reporter.send_once()
        await reporter.send_once()

        assert [snapshot.sequence for snapshot in controller.snapshots] == [1, 2]

    async def test_an_unchanged_membership_is_sent_as_a_digest_alone(self):
        """A run of 10k engines would otherwise ship its whole membership every period for nothing."""
        reporter, _provider, controller = await _synced()

        await reporter.send_once()
        await reporter.send_once()

        assert controller.snapshots[0].cells is not None
        assert controller.snapshots[1].cells is None
        assert controller.snapshots[1].digest == controller.snapshots[0].digest

    async def test_a_membership_the_run_does_not_hold_is_sent_whole_again(self):
        """The unconfirmed digest is how the controller asks for the whole snapshot back."""
        reporter, _provider, controller = await _synced(
            controller=_FakeController(acks=[RegistrationAck(applied_digest=None)])
        )

        await reporter.send_once()
        await reporter.send_once()

        assert controller.snapshots[1].cells is not None

    async def test_a_changed_membership_is_sent_whole(self):
        """The digest short circuit must not hide a cell that appeared or went away."""
        reporter, provider, controller = await _synced()
        await reporter.send_once()

        await provider.reconcile(f"{_POOL_ID}-1", _cell_info(1))
        await reporter.send_once()

        assert [cell.cell_id for cell in controller.snapshots[1].cells] == [
            f"{_REPORTER_ID}-{_POOL_ID}-0",
            f"{_REPORTER_ID}-{_POOL_ID}-1",
        ]


class TestReporterWorker:
    def test_the_worker_the_engine_release_serves_exposes_an_rpc_surface(self):
        """A worker class with no public method is refused by the rpc app, so its pod would crash loop."""
        assert "get_registration_status" in collect_rpc_method_specs(RegistrationReporterWorker)

    async def test_its_status_says_what_this_deployment_has_reported(self):
        """It is the only way to ask a running engine release whether the run took its engines in."""
        reporter, _provider, _controller = await _synced()
        await reporter.send_once()

        status = reporter.compute_status()

        assert status.reporter_id == _REPORTER_ID
        assert status.sequence == 1
        assert status.num_observed_cells == 1
        assert status.acknowledged_digest is not None


class TestReporterLifecycle:
    async def test_it_waits_for_the_controller_before_it_watches_its_own_cells(self):
        """The controller comes up with the run, and a snapshot into nothing is a wasted period."""
        provider = _FakeEngineProvider(cell_indices=[0])
        controller = _FakeController()
        run = asyncio.create_task(_reporter(provider=provider, controller=controller).run())
        for _ in range(100):
            await asyncio.sleep(0)
            if provider.reconcile is not None:
                break

        run.cancel()
        with pytest.raises(asyncio.CancelledError):
            await run

        assert controller.ready_timeouts
        assert provider.stopped
