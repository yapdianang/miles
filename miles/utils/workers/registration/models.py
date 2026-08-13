from __future__ import annotations

import hashlib
import json
from typing import Any

from miles.utils.pydantic_utils import FrozenStrictBaseModel
from miles.utils.workers.worker_spec import NamedHostAndPorts

SNAPSHOT_INTERVAL_SECONDS = 15.0
SNAPSHOT_JITTER_RATIO = 0.2
SNAPSHOT_DEBOUNCE_SECONDS = 1.0
CONTROLLER_READY_TIMEOUT_SECONDS = 3600.0

SUPPORTED_WORKER_TYPE = "regular"


class RegisteredWorker(FrozenStrictBaseModel):
    name: str
    addrs: NamedHostAndPorts
    gpu_ids: list[int]


class RegisteredCell(FrozenStrictBaseModel):
    cell_id: str
    pool_id: str
    workers_hash: str
    workers: list[RegisteredWorker]
    meta: dict[str, Any]


class RegistrationSnapshot(FrozenStrictBaseModel):
    reporter_id: str
    sequence: int
    digest: str
    expected_num_cells_by_model: dict[str, int]
    token: str | None = None
    cells: list[RegisteredCell] | None = None


class RegistrationAck(FrozenStrictBaseModel):
    applied_digest: str | None
    excluded_cell_ids: list[str] = []


class RegistrationReporterStatus(FrozenStrictBaseModel):
    reporter_id: str
    sequence: int
    num_observed_cells: int
    acknowledged_digest: str | None


def compute_snapshot_digest(*, cells: list[RegisteredCell], expected_num_cells_by_model: dict[str, int]) -> str:
    payload = json.dumps(
        dict(
            cells=[cell.model_dump(mode="json") for cell in sorted(cells, key=lambda cell: cell.cell_id)],
            expected_num_cells_by_model=expected_num_cells_by_model,
        ),
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()
