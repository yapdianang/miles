from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field

from miles.utils.workers.naming import parse_cell_id
from miles.utils.workers.reconcile.list_based import ListBasedReconcileLoop
from miles.utils.workers.registration.models import RegisteredCell, RegistrationAck, RegistrationSnapshot
from miles.utils.workers.worker_info import WorkerInfo
from miles.utils.workers.worker_provider.base import (
    BaseWorkerProvider,
    CellInfo,
    ReconcileFn,
    StopWatchFn,
    cell_id_of_worker,
)
from miles.utils.workers.worker_spec import NamedHostAndPorts

logger = logging.getLogger(__name__)

REGISTERED_CELLS_POLL_INTERVAL_SECONDS = 5.0


@dataclass
class _ReporterState:
    sequence: int = -1
    digest: str | None = None
    expected_num_cells_by_model: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class _OwnedCell:
    reporter_id: str
    info: CellInfo
    addrs_by_worker: dict[str, NamedHostAndPorts]
    gpu_ids_by_worker: dict[str, list[int]]


@dataclass(frozen=True)
class _ParsedSnapshot:
    cells: dict[str, _OwnedCell]
    excluded_cell_ids: list[str]


class RegistrationWorkerProvider(BaseWorkerProvider):
    def __init__(
        self,
        *,
        expected_num_reporters: int,
        token: str | None = None,
        refuse_cell: Callable[[CellInfo], str | None] = lambda _info: None,
    ) -> None:
        self._expected_num_reporters = expected_num_reporters
        self._token = token
        self._refuse_cell = refuse_cell
        self._reporters: dict[str, _ReporterState] = {}
        self._cells: dict[str, _OwnedCell] = {}
        self._watched = False

    async def apply_snapshot(self, snapshot: RegistrationSnapshot) -> RegistrationAck:
        self._assert_authentic(snapshot)
        parsed = None if snapshot.cells is None else self._parse_cells(snapshot)
        return self._commit_snapshot(snapshot, parsed=parsed)

    def cell_ids(self) -> list[str]:
        return sorted(self._cells)

    def extra_expected_num_cells(self, *, model_id: str) -> int:
        assert len(self._reporters) >= self._expected_num_reporters, (
            f"{len(self._reporters)}/{self._expected_num_reporters} engine deployments have reported themselves "
            f"({sorted(self._reporters)}), so the cells of the missing ones are not known yet"
        )
        return sum(state.expected_num_cells_by_model.get(model_id, 0) for state in self._reporters.values())

    async def get_addrs(self, worker_name: str) -> NamedHostAndPorts:
        cell = self._cells[cell_id_of_worker(worker_name)]
        assert worker_name in cell.addrs_by_worker, (
            f"{worker_name} is not one of the workers {sorted(cell.addrs_by_worker)} that reporter "
            f"{cell.reporter_id} reported for cell {cell.info.cell_id}"
        )
        return cell.addrs_by_worker[worker_name]

    def get_worker_infos(self, *, cell_ids: list[str]) -> list[list[WorkerInfo]]:
        return [self._worker_infos_of_cell(cell_id) for cell_id in cell_ids]

    async def watch_cells(self, reconcile: ReconcileFn) -> StopWatchFn:
        assert not self._watched, "a registration provider reports to exactly one watcher"
        self._watched = True
        loop = ListBasedReconcileLoop(
            list_cells=self._list_cells,
            poll_interval_seconds=REGISTERED_CELLS_POLL_INTERVAL_SECONDS,
        )
        return await loop.start(reconcile)

    async def _list_cells(self) -> dict[str, CellInfo]:
        return {cell_id: cell.info for cell_id, cell in self._cells.items()}

    def _assert_authentic(self, snapshot: RegistrationSnapshot) -> None:
        assert snapshot.token == self._token, (
            f"reporter {snapshot.reporter_id} presented a registration token this run does not accept; both sides "
            f"read it from --registration-token"
        )

    def _parse_cells(self, snapshot: RegistrationSnapshot) -> _ParsedSnapshot:
        occurrences = Counter(cell.cell_id for cell in snapshot.cells)
        parsed: dict[str, _OwnedCell] = {}
        excluded: list[str] = []
        for cell in snapshot.cells:
            info = _compute_cell_info(cell)
            reason = (
                f"one snapshot carries it {occurrences[cell.cell_id]} times, and a cell id names exactly one cell, "
                f"so keeping either entry would confirm a digest for a membership this run does not hold"
                if occurrences[cell.cell_id] > 1
                else _compute_refusal_reason(cell) or self._refuse_cell(info)
            )
            if reason is not None:
                logger.error(
                    f"Excluding cell {cell.cell_id} of reporter {snapshot.reporter_id} from this run: {reason}. "
                    f"The rest of the snapshot is taken in, and the whole snapshot is asked for again"
                )
                excluded.append(cell.cell_id)
                continue
            parsed[cell.cell_id] = _OwnedCell(
                reporter_id=snapshot.reporter_id,
                info=info,
                addrs_by_worker={worker.name: worker.addrs for worker in cell.workers},
                gpu_ids_by_worker={worker.name: list(worker.gpu_ids) for worker in cell.workers},
            )
        return _ParsedSnapshot(cells=parsed, excluded_cell_ids=sorted(set(excluded)))

    def _commit_snapshot(self, snapshot: RegistrationSnapshot, *, parsed: _ParsedSnapshot | None) -> RegistrationAck:
        state = self._reporters.get(snapshot.reporter_id)
        if state is not None and snapshot.sequence <= state.sequence:
            logger.warning(
                f"Ignoring snapshot {snapshot.sequence} of reporter {snapshot.reporter_id}: snapshot "
                f"{state.sequence} is at least as new, so this one arrived late"
            )
            return _ack(state)

        if parsed is None:
            return self._apply_heartbeat(snapshot, state=state)

        state = self._reporters.setdefault(snapshot.reporter_id, _ReporterState())
        excluded = self._replace_membership(reporter_id=snapshot.reporter_id, parsed=parsed)
        state.sequence = snapshot.sequence
        state.expected_num_cells_by_model = dict(snapshot.expected_num_cells_by_model)
        state.digest = None if excluded else snapshot.digest
        return _ack(state, excluded_cell_ids=excluded)

    def _apply_heartbeat(self, snapshot: RegistrationSnapshot, *, state: _ReporterState | None) -> RegistrationAck:
        if state is None or state.digest != snapshot.digest:
            logger.warning(
                f"Reporter {snapshot.reporter_id} sent a heartbeat for a snapshot this run does not hold, so it "
                f"will be asked for the whole snapshot again"
            )
        if state is None:
            return RegistrationAck(applied_digest=None)

        state.sequence = snapshot.sequence
        return _ack(state)

    def _replace_membership(self, *, reporter_id: str, parsed: _ParsedSnapshot) -> list[str]:
        excluded = list(parsed.excluded_cell_ids)
        cells: dict[str, _OwnedCell] = {}
        for cell_id, cell in parsed.cells.items():
            if (owner := self._cells.get(cell_id)) is not None and owner.reporter_id != reporter_id:
                logger.error(
                    f"Excluding cell {cell_id} of reporter {reporter_id} from this run: reporter "
                    f"{owner.reporter_id} already reported it, so two deployments share a pool id"
                )
                excluded.append(cell_id)
                continue
            cells[cell_id] = cell

        held = {cell_id for cell_id, cell in self._cells.items() if cell.reporter_id == reporter_id}
        for cell_id in held - set(cells):
            del self._cells[cell_id]
        self._cells.update(cells)
        return sorted(set(excluded))

    def _worker_infos_of_cell(self, cell_id: str) -> list[WorkerInfo]:
        cell = self._cells[cell_id]
        return [
            WorkerInfo(
                name=worker_name,
                generation=0,
                self_addrs=cell.addrs_by_worker[worker_name],
                gpu_ids=cell.gpu_ids_by_worker[worker_name],
                worker_class=None,
            )
            for worker_name in cell.info.worker_names
        ]


def _compute_cell_info(cell: RegisteredCell) -> CellInfo:
    return CellInfo(
        cell_id=cell.cell_id,
        pool_id=cell.pool_id,
        alive=True,
        worker_names=[worker.name for worker in cell.workers],
        workers_hash=cell.workers_hash,
        meta=dict(cell.meta),
    )


def _compute_refusal_reason(cell: RegisteredCell) -> str | None:
    try:
        pool_id = parse_cell_id(cell.cell_id).pool_id
        worker_cell_ids = {cell_id_of_worker(worker.name) for worker in cell.workers}
    except ValueError:
        return (
            "its cell id, or the name of one of its workers, does not read as <pool id>-<cell index>, and this run "
            "parses those names to address the workers of that cell"
        )
    if pool_id != cell.pool_id:
        return (
            f"it does not name its own pool {cell.pool_id}, and a reporter namespaces its pool ids so that two "
            f"deployments never collide"
        )
    if not cell.workers:
        return "it carries no worker to address"
    if worker_cell_ids != {cell.cell_id}:
        return f"its workers {sorted(worker.name for worker in cell.workers)} do not all belong to it"
    return None


def _ack(state: _ReporterState, *, excluded_cell_ids: list[str] | None = None) -> RegistrationAck:
    return RegistrationAck(applied_digest=state.digest, excluded_cell_ids=excluded_cell_ids or [])
