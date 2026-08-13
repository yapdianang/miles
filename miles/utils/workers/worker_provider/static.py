from __future__ import annotations

from collections.abc import Callable, Iterable
from functools import partial

from miles.utils.http_utils import _wrap_ipv6, wait_tcp_ready
from miles.utils.workers.naming import compute_worker_name, parse_cell_id, parse_worker_name
from miles.utils.workers.worker_handle import BaseWorkerHandle
from miles.utils.workers.worker_info import WorkerInfo
from miles.utils.workers.worker_provider.base import BaseWorkerProvider
from miles.utils.workers.worker_provider.kubernetes.helm import naming
from miles.utils.workers.worker_provider.utils import build_rpc_handle_of_worker_info
from miles.utils.workers.worker_spec import (
    RPC_PORT_NAME,
    BaseWorkerSpec,
    HostAndPort,
    NamedHostAndPorts,
    ServeWorkerSpec,
)

STATIC_ADDRS_READY_TIMEOUT_SECONDS = 600.0


class StaticWorkerProvider(BaseWorkerProvider):
    def __init__(
        self,
        *,
        pool_id: str,
        num_cells: int,
        num_workers_per_cell: int,
        compute_addrs: Callable[[int, int], NamedHostAndPorts],
        worker_class: str | None,
    ) -> None:
        assert num_cells > 0, f"pool {pool_id} is addressed statically, so it needs at least one cell"
        self._pool_id = pool_id
        self._num_cells = num_cells
        self._num_workers_per_cell = num_workers_per_cell
        self._compute_addrs = compute_addrs
        self._worker_class = worker_class

    @classmethod
    def of_release(cls, *, release: str, spec: BaseWorkerSpec) -> StaticWorkerProvider:
        scheduling = spec.scheduling
        return cls(
            pool_id=spec.name,
            num_cells=scheduling.num_cells,
            num_workers_per_cell=scheduling.num_workers_per_cell,
            compute_addrs=partial(_addrs_from_release_naming, spec=spec, release=release),
            worker_class=spec.worker_class if isinstance(spec, ServeWorkerSpec) else None,
        )

    @classmethod
    def of_rpc_urls(cls, *, pool_id: str, urls: list[str], worker_class: str) -> StaticWorkerProvider:
        addrs_by_cell = [{RPC_PORT_NAME: parse_host_and_port(url)} for url in urls]
        return cls(
            pool_id=pool_id,
            num_cells=len(addrs_by_cell),
            num_workers_per_cell=1,
            compute_addrs=lambda cell_index, _worker_in_cell_index: addrs_by_cell[cell_index],
            worker_class=worker_class,
        )

    async def get_addrs(self, worker_name: str) -> NamedHostAndPorts:
        return self._addrs_of_worker(worker_name)

    def get_worker_infos(self, *, cell_ids: list[str]) -> list[list[WorkerInfo]]:
        return [
            [
                self._worker_info(cell_index=self._cell_index_of(cell_id), worker_in_cell_index=worker_in_cell_index)
                for worker_in_cell_index in range(self._num_workers_per_cell)
            ]
            for cell_id in cell_ids
        ]

    def _build_handle_of_worker_info(self, info: WorkerInfo) -> BaseWorkerHandle:
        assert (
            info.worker_class is not None
        ), f"pool {self._pool_id} is launched as a command rather than served, so its rpc methods are unknown"
        return build_rpc_handle_of_worker_info(info)

    def _worker_info(self, *, cell_index: int, worker_in_cell_index: int) -> WorkerInfo:
        return WorkerInfo(
            name=compute_worker_name(
                pool_id=self._pool_id, cell_index=cell_index, worker_in_cell_index=worker_in_cell_index
            ),
            generation=0,
            self_addrs=self._compute_addrs(cell_index, worker_in_cell_index),
            gpu_ids=[],
            worker_class=self._worker_class,
        )

    def _addrs_of_worker(self, worker_name: str) -> NamedHostAndPorts:
        pool_id, cell_index, worker_in_cell_index = parse_worker_name(worker_name)
        self._check_cell(pool_id=pool_id, cell_index=cell_index)
        assert worker_in_cell_index < self._num_workers_per_cell, (
            f"a cell of pool {pool_id} holds {self._num_workers_per_cell} workers, "
            f"so worker {worker_in_cell_index} is not one of them"
        )
        return self._compute_addrs(cell_index, worker_in_cell_index)

    def _cell_index_of(self, cell_id: str) -> int:
        pool_id, cell_index = parse_cell_id(cell_id)
        self._check_cell(pool_id=pool_id, cell_index=cell_index)
        return cell_index

    def _check_cell(self, *, pool_id: str, cell_index: int) -> None:
        assert pool_id == self._pool_id, f"this provider answers for pool {self._pool_id}, not {pool_id}"
        assert (
            cell_index < self._num_cells
        ), f"pool {pool_id} deploys {self._num_cells} cells, so cell {cell_index} is not one of them"


def _addrs_from_release_naming(
    cell_index: int, worker_in_cell_index: int, *, spec: BaseWorkerSpec, release: str
) -> NamedHostAndPorts:
    scheduling = spec.scheduling
    assert scheduling.pods_per_cell() == 1, (
        f"pool {spec.name} spreads a cell over {scheduling.pods_per_cell()} pods, "
        f"so its workers do not share one host"
    )
    return naming.static_cell_addrs(
        spec=spec,
        release=release,
        cell_index=cell_index,
        worker_in_pod_index=worker_in_cell_index,
    )


def wait_static_addrs_ready(addrs: Iterable[HostAndPort]) -> None:
    for addr in addrs:
        wait_tcp_ready(addr.host, addr.port, timeout=STATIC_ADDRS_READY_TIMEOUT_SECONDS)


def parse_host_and_port(addr: str) -> HostAndPort:
    rest = addr.split("://", 1)[1] if "://" in addr else addr
    host, separator, port = rest.rstrip("/").rpartition(":")
    assert separator and port.isdigit(), f"static address {addr!r} must be host:port or http://host:port"
    return HostAndPort(host=_wrap_ipv6(host), port=int(port))
