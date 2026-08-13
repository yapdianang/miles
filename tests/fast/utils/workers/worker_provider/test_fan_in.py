from __future__ import annotations

import pytest

from miles.utils.workers.worker_info import WorkerInfo
from miles.utils.workers.worker_provider.base import BaseWorkerProvider, CellInfo, ReconcileFn, StopWatchFn
from miles.utils.workers.worker_provider.fan_in import FanInWorkerProvider
from miles.utils.workers.worker_spec import HostAndPort, NamedHostAndPorts


class _FakeProvider(BaseWorkerProvider):
    def __init__(self, *, host: str, expected: int | None = None, extra: int = 0) -> None:
        self.host = host
        self.initialized = False
        self.stopped = False
        self._expected = expected
        self._extra = extra
        self._reconcile: ReconcileFn | None = None

    async def init(self) -> None:
        self.initialized = True

    async def get_addrs(self, worker_name: str) -> NamedHostAndPorts:
        return {"primary": HostAndPort(host=self.host, port=8000)}

    def get_worker_infos(self, *, cell_ids: list[str]) -> list[list[WorkerInfo]]:
        return [
            [
                WorkerInfo(
                    name=f"{cell_id}-0",
                    generation=0,
                    self_addrs={"primary": HostAndPort(host=self.host, port=8000)},
                    gpu_ids=[],
                    worker_class=None,
                )
            ]
            for cell_id in cell_ids
        ]

    async def watch_cells(self, reconcile: ReconcileFn) -> StopWatchFn:
        self._reconcile = reconcile

        async def _stop() -> None:
            self.stopped = True

        return _stop

    def expected_num_cells(self, *, model_id: str) -> int | None:
        return self._expected

    def extra_expected_num_cells(self, *, model_id: str) -> int:
        return self._extra

    async def announce(self, cell_id: str, observed: CellInfo | None) -> None:
        await self._reconcile(cell_id, observed)


def _cell_info(cell_id: str) -> CellInfo:
    return CellInfo(
        cell_id=cell_id,
        pool_id=cell_id.rsplit("-", maxsplit=1)[0],
        alive=True,
        worker_names=[f"{cell_id}-0"],
        workers_hash="hash-1",
        meta={},
    )


class _Watcher:
    def __init__(self) -> None:
        self.observations: list[tuple[str, CellInfo | None]] = []

    async def reconcile(self, cell_id: str, observed: CellInfo | None) -> None:
        self.observations.append((cell_id, observed))


class TestFanInWorkerProvider:
    async def test_every_provider_it_fans_in_is_initialized(self):
        """One uninitialized provider would answer for cells it never observed."""
        local, registered = _FakeProvider(host="local"), _FakeProvider(host="registered")
        provider = FanInWorkerProvider(providers=[local, registered])

        await provider.init()

        assert local.initialized and registered.initialized

    async def test_a_cell_is_answered_for_by_the_provider_that_announced_it(self):
        """The address of a registered engine only exists in the registry that was told it."""
        local, registered = _FakeProvider(host="local"), _FakeProvider(host="registered")
        provider = FanInWorkerProvider(providers=[local, registered])
        watcher = _Watcher()
        await provider.watch_cells(watcher.reconcile)

        await local.announce("pool-a-0", _cell_info("pool-a-0"))
        await registered.announce("pool-b-0", _cell_info("pool-b-0"))

        assert (await provider.get_addrs("pool-a-0-0"))["primary"].host == "local"
        assert (await provider.get_addrs("pool-b-0-0"))["primary"].host == "registered"

    async def test_a_cell_nobody_announced_is_not_answered_for(self):
        """A guess here would send a request of this run to an engine that never joined it."""
        provider = FanInWorkerProvider(providers=[_FakeProvider(host="local")])

        with pytest.raises(AssertionError, match="no provider"):
            await provider.get_addrs("pool-a-0-0")

    async def test_a_removed_cell_is_forgotten_so_its_id_can_be_announced_again(self):
        """Cell ids come back after a deployment is rebuilt, sometimes from another provider."""
        local, registered = _FakeProvider(host="local"), _FakeProvider(host="registered")
        provider = FanInWorkerProvider(providers=[local, registered])
        watcher = _Watcher()
        await provider.watch_cells(watcher.reconcile)

        await local.announce("pool-a-0", _cell_info("pool-a-0"))
        await local.announce("pool-a-0", None)
        await registered.announce("pool-a-0", _cell_info("pool-a-0"))

        assert (await provider.get_addrs("pool-a-0-0"))["primary"].host == "registered"

    async def test_a_cell_two_providers_claim_at_once_stops_the_run(self):
        """The second claimant could remove a cell of the first, which would then never be announced again."""
        local, registered = _FakeProvider(host="local"), _FakeProvider(host="registered")
        provider = FanInWorkerProvider(providers=[local, registered])
        watcher = _Watcher()
        await provider.watch_cells(watcher.reconcile)
        await local.announce("pool-a-0", _cell_info("pool-a-0"))

        with pytest.raises(AssertionError, match="announced by"):
            await registered.announce("pool-a-0", _cell_info("pool-a-0"))

    async def test_stopping_the_watch_stops_every_provider(self):
        """A watch left running would keep calling a controller that is tearing down."""
        local, registered = _FakeProvider(host="local"), _FakeProvider(host="registered")
        provider = FanInWorkerProvider(providers=[local, registered])
        stop = await provider.watch_cells(_Watcher().reconcile)

        await stop()

        assert local.stopped and registered.stopped

    async def test_the_cells_other_deployments_bring_are_added_to_the_ones_this_one_waits_for(self):
        """The run waits for every engine it will serve from, whichever deployment launched it."""
        provider = FanInWorkerProvider(
            providers=[_FakeProvider(host="local"), _FakeProvider(host="registered", extra=6)]
        )

        assert provider.extra_expected_num_cells(model_id="default") == 6

    async def test_a_count_nobody_answers_is_left_to_the_run_configuration(self):
        """The run falls back to the engine count its own arguments describe."""
        provider = FanInWorkerProvider(providers=[_FakeProvider(host="local"), _FakeProvider(host="registered")])

        assert provider.expected_num_cells(model_id="default") is None

    async def test_counts_are_only_added_up_when_every_provider_answers(self):
        """Adding a counted provider to an uncounted one would silently halve what the run waits for."""
        provider = FanInWorkerProvider(
            providers=[_FakeProvider(host="local", expected=2), _FakeProvider(host="registered")]
        )

        with pytest.raises(AssertionError, match="count the cells"):
            provider.expected_num_cells(model_id="default")
