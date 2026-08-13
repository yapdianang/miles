from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest

from miles.utils.workers.registration.models import (
    RegisteredCell,
    RegisteredWorker,
    RegistrationAck,
    RegistrationSnapshot,
    compute_snapshot_digest,
)
from miles.utils.workers.registration.provider import RegistrationWorkerProvider
from miles.utils.workers.worker_provider.base import CellInfo
from miles.utils.workers.worker_spec import HostAndPort

_REPORTER = "miles-run-r1-inference"
_POOL_ID = f"{_REPORTER}-inference-engine-0-0"
_PROVIDER_MODULE = "miles.utils.workers.registration.provider"
_POLL_INTERVAL_SECONDS = 0.001


def _cell(cell_index: int, *, host: str = "10.0.0.5", model_id: str = "default") -> RegisteredCell:
    cell_id = f"{_POOL_ID}-{cell_index}"
    return RegisteredCell(
        cell_id=cell_id,
        pool_id=_POOL_ID,
        workers_hash=f"hash-{host}",
        workers=[
            RegisteredWorker(
                name=f"{cell_id}-0",
                addrs={"primary": HostAndPort(host=host, port=8000)},
                gpu_ids=[0],
            )
        ],
        meta=dict(model_id=model_id, worker_type="regular"),
    )


def _snapshot(
    cells: list[RegisteredCell] | None,
    *,
    reporter_id: str = _REPORTER,
    sequence: int = 1,
    token: str | None = None,
    expected_num_cells_by_model: dict[str, int] | None = None,
    digest: str | None = None,
) -> RegistrationSnapshot:
    expected = expected_num_cells_by_model if expected_num_cells_by_model is not None else {"default": 2}
    return RegistrationSnapshot(
        reporter_id=reporter_id,
        sequence=sequence,
        digest=(
            digest
            if digest is not None
            else compute_snapshot_digest(cells=cells or [], expected_num_cells_by_model=expected)
        ),
        expected_num_cells_by_model=expected,
        token=token,
        cells=cells,
    )


class _Watcher:
    def __init__(self) -> None:
        self.observations: list[tuple[str, CellInfo | None]] = []
        self.failing_cell_ids: set[str] = set()

    async def reconcile(self, cell_id: str, observed: CellInfo | None) -> None:
        if cell_id in self.failing_cell_ids:
            raise RuntimeError(f"cell {cell_id} refuses to be reconciled")
        self.observations.append((cell_id, observed))


async def _watched(**kwargs) -> tuple[RegistrationWorkerProvider, _Watcher]:
    watcher = _Watcher()
    provider = RegistrationWorkerProvider(expected_num_reporters=1, **kwargs)
    await _start_watch(provider, watcher)
    return provider, watcher


async def _start_watch(provider: RegistrationWorkerProvider, watcher: _Watcher) -> None:
    with patch(f"{_PROVIDER_MODULE}.REGISTERED_CELLS_POLL_INTERVAL_SECONDS", _POLL_INTERVAL_SECONDS):
        await provider.watch_cells(watcher.reconcile)


async def _apply(provider: RegistrationWorkerProvider, snapshot: RegistrationSnapshot) -> RegistrationAck:
    ack = await provider.apply_snapshot(snapshot)
    await _drain()
    return ack


async def _drain() -> None:
    for _ in range(50):
        await asyncio.sleep(_POLL_INTERVAL_SECONDS)


class TestSnapshotMembership:
    async def test_a_first_snapshot_announces_every_cell_it_carries(self):
        """A snapshot is the whole membership of its deployment, so the run takes all of it in at once."""
        provider, watcher = await _watched()

        await _apply(provider, _snapshot([_cell(0), _cell(1)]))

        assert provider.cell_ids() == [f"{_POOL_ID}-0", f"{_POOL_ID}-1"]
        assert [cell_id for cell_id, _observed in watcher.observations] == [f"{_POOL_ID}-0", f"{_POOL_ID}-1"]

    async def test_an_unchanged_cell_is_not_announced_twice(self):
        """The run adds a cell once; announcing it again would tear down a serving engine to rebuild it."""
        provider, watcher = await _watched()

        await _apply(provider, _snapshot([_cell(0)], sequence=1))
        await _apply(provider, _snapshot([_cell(0)], sequence=2))

        assert len(watcher.observations) == 1

    async def test_a_cell_that_changed_is_announced_anew(self):
        """A cell rebuilt on another host serves from another address, and the old one answers nothing."""
        provider, watcher = await _watched()

        await _apply(provider, _snapshot([_cell(0)], sequence=1))
        await _apply(provider, _snapshot([_cell(0, host="10.0.0.6")], sequence=2))

        (_first, (cell_id, observed)) = watcher.observations
        assert cell_id == f"{_POOL_ID}-0"
        assert observed.workers_hash == "hash-10.0.0.6"

    async def test_a_cell_the_snapshot_stops_naming_is_removed(self):
        """Membership is level, so omission is how a deployment says a cell is gone; there is no death message."""
        provider, watcher = await _watched()

        await _apply(provider, _snapshot([_cell(0), _cell(1)], sequence=1))
        await _apply(provider, _snapshot([_cell(0)], sequence=2))

        assert provider.cell_ids() == [f"{_POOL_ID}-0"]
        assert watcher.observations[-1] == (f"{_POOL_ID}-1", None)

    async def test_a_late_snapshot_is_ignored(self):
        """A snapshot that crossed the wan slowly would otherwise resurrect cells the run already dropped."""
        provider, _watcher = await _watched()

        await _apply(provider, _snapshot([_cell(0), _cell(1)], sequence=5))
        await _apply(provider, _snapshot([_cell(0)], sequence=4))

        assert provider.cell_ids() == [f"{_POOL_ID}-0", f"{_POOL_ID}-1"]

    async def test_a_cell_reported_by_two_deployments_is_refused(self):
        """One cell id names one cell, and the second owner could remove the cell of the first."""
        provider, _watcher = await _watched()
        await _apply(provider, _snapshot([_cell(0)]))

        ack = await _apply(provider, _snapshot([_cell(0)], reporter_id="other"))

        assert ack.excluded_cell_ids == [f"{_POOL_ID}-0"]

    async def test_a_cell_carried_twice_by_one_snapshot_is_refused(self):
        """Either entry could be the truth, so confirming the digest would confirm a membership nobody holds."""
        provider, _watcher = await _watched()

        ack = await _apply(provider, _snapshot([_cell(0), _cell(0, host="10.0.0.6")]))

        assert ack.excluded_cell_ids == [f"{_POOL_ID}-0"]
        assert provider.cell_ids() == []

    async def test_a_cell_of_a_model_this_run_does_not_serve_is_refused(self):
        """No router of this run would ever send it a request, so counting it would stall the wait for cells."""
        provider, _watcher = await _watched(
            refuse_cell=lambda info: None if info.meta["model_id"] == "default" else "unknown model"
        )

        ack = await _apply(provider, _snapshot([_cell(0, model_id="other")]))

        assert ack.excluded_cell_ids == [f"{_POOL_ID}-0"]

    async def test_a_cell_that_does_not_name_its_own_pool_is_refused(self):
        """The run parses a cell id to address the workers of that cell."""
        provider, _watcher = await _watched()
        cell = _cell(0).model_copy(update=dict(cell_id="not-a-cell-id"))

        ack = await _apply(provider, _snapshot([cell]))

        assert ack.excluded_cell_ids == ["not-a-cell-id"]


class TestSnapshotAuthentication:
    async def test_a_snapshot_with_the_expected_token_is_taken_in(self):
        """Both sides read the same secret from --registration-token."""
        provider, _watcher = await _watched(token="secret")

        await _apply(provider, _snapshot([_cell(0)], token="secret"))

        assert provider.cell_ids() == [f"{_POOL_ID}-0"]

    async def test_a_snapshot_with_another_token_is_refused(self):
        """Otherwise a deployment of another run could put its engines into this run's routers."""
        provider, _watcher = await _watched(token="secret")

        with pytest.raises(AssertionError, match="registration token"):
            await provider.apply_snapshot(_snapshot([_cell(0)], token="guessed"))

        assert provider.cell_ids() == []

    async def test_a_snapshot_without_a_token_is_refused_when_one_is_expected(self):
        """A missing token is a wrong token: it proves nothing about who is reporting."""
        provider, _watcher = await _watched(token="secret")

        with pytest.raises(AssertionError, match="registration token"):
            await provider.apply_snapshot(_snapshot([_cell(0)]))


class TestDigestHeartbeats:
    async def test_a_heartbeat_for_the_held_snapshot_keeps_the_membership(self):
        """Steady state costs one digest, not the whole membership of the deployment."""
        provider, _watcher = await _watched()
        first = _snapshot([_cell(0)], sequence=1)
        await _apply(provider, first)

        ack = await _apply(provider, _snapshot(None, sequence=2, digest=first.digest))

        assert ack.applied_digest == first.digest
        assert provider.cell_ids() == [f"{_POOL_ID}-0"]

    async def test_a_heartbeat_for_a_membership_this_run_does_not_hold_is_not_confirmed(self):
        """The reporter learns from the unconfirmed digest that it has to send the whole snapshot again."""
        provider, _watcher = await _watched()
        await _apply(provider, _snapshot([_cell(0)], sequence=1))

        ack = await _apply(provider, _snapshot(None, sequence=2, digest="another-digest"))

        assert ack.applied_digest != "another-digest"


class TestExpectedReporters:
    async def test_a_run_does_not_count_cells_before_every_deployment_has_reported(self):
        """A run that counted only the deployments it heard from would start half its engines short."""
        provider = RegistrationWorkerProvider(expected_num_reporters=2)

        with pytest.raises(AssertionError, match="0/2"):
            provider.extra_expected_num_cells(model_id="default")

    async def test_a_run_counts_the_cells_every_reported_deployment_expects(self):
        """The count is what the run waits for, and it is the sum over the deployments, not over what arrived."""
        provider = RegistrationWorkerProvider(expected_num_reporters=2)
        await _start_watch(provider, _Watcher())

        await _apply(provider, _snapshot([_cell(0)], expected_num_cells_by_model={"default": 2}))
        await _apply(provider, _snapshot([], reporter_id="other", expected_num_cells_by_model={"default": 3}))

        assert provider.extra_expected_num_cells(model_id="default") == 5
        assert provider.extra_expected_num_cells(model_id="unknown-model") == 0

    async def test_a_run_expecting_nobody_counts_nothing(self):
        """An unsplit run must not wait for cells that no deployment will ever announce."""
        provider = RegistrationWorkerProvider(expected_num_reporters=0)

        assert provider.extra_expected_num_cells(model_id="default") == 0


class TestAddressingRegisteredCells:
    async def test_the_addresses_of_a_registered_worker_are_the_ones_reported(self):
        """The run calls the engine at the address its own deployment observed, never at one derived here."""
        provider, _watcher = await _watched()
        await _apply(provider, _snapshot([_cell(0)]))

        addrs = await provider.get_addrs(f"{_POOL_ID}-0-0")

        assert addrs["primary"] == HostAndPort(host="10.0.0.5", port=8000)

    async def test_a_watcher_that_starts_late_is_replayed_the_cells_already_reported(self):
        """A snapshot may land before the controller watches, and nothing announces that cell a second time."""
        provider = RegistrationWorkerProvider(expected_num_reporters=1)
        await _apply(provider, _snapshot([_cell(0)]))

        watcher = _Watcher()
        await _start_watch(provider, watcher)

        assert [cell_id for cell_id, _observed in watcher.observations] == [f"{_POOL_ID}-0"]


class TestFailedReconciliation:
    async def test_a_cell_the_run_could_not_take_in_is_offered_again_on_the_next_poll(self):
        """A cell whose reconcile raised must be retried from the membership this run already holds."""
        provider, watcher = await _watched()
        watcher.failing_cell_ids = {f"{_POOL_ID}-0"}

        ack = await _apply(provider, _snapshot([_cell(0)], sequence=1))

        assert watcher.observations == []
        assert provider.cell_ids() == [f"{_POOL_ID}-0"]
        assert ack.applied_digest is not None

        watcher.failing_cell_ids = set()
        await _drain()

        assert [cell_id for cell_id, _observed in watcher.observations] == [f"{_POOL_ID}-0"]
