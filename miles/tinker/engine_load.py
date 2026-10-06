"""Live load of each SGLang engine the router serves, for rollout admission and placement."""

import asyncio
import time
from dataclasses import dataclass

import httpx
from prometheus_client.parser import text_string_to_metric_families

from miles.tinker.core.types import EngineUnavailableError
from miles.utils.http_utils import router_worker_base_urls

# SGLang publishes scheduler gauges about once per decode-log interval; polling faster only repeats them.
_REFRESH_SECONDS = 1.0
_REQUEST_TIMEOUT_SECONDS = 5.0


@dataclass(frozen=True)
class KvPool:
    # Slots held by running requests, slots holding evictable prefix cache, and the pool size.
    used_tokens: int
    evictable_tokens: int
    total_tokens: int

    @property
    def free_tokens(self) -> int:
        return self.total_tokens - self.used_tokens - self.evictable_tokens


@dataclass(frozen=True)
class EngineLoad:
    rank: int
    url: str
    running_requests: int
    queued_requests: int
    max_running_requests: int
    full_kv: KvPool
    swa_kv: KvPool | None
    host_kv: KvPool | None

    def measure_room(self, sequence_tokens: float) -> float:
        """Sequences of this length that fit both the free request slots and the free device and host KV."""
        slots = self.max_running_requests - self.running_requests - self.queued_requests
        free_tokens = self.full_kv.free_tokens + (self.host_kv.free_tokens if self.host_kv is not None else 0)
        return min(slots, free_tokens / sequence_tokens)


def parse_engine_load(rank: int, url: str, loads: dict, metrics: str) -> EngineLoad:
    """Combine an engine's ``/v1/loads`` core section with the pool gauges of its ``/metrics``."""
    dp_loads = loads["loads"]
    gauges = _read_gauges(metrics)

    def read_pool(used: str, available: str, total: str) -> KvPool | None:
        if total not in gauges:
            return None
        return KvPool(
            used_tokens=int(gauges[used]),
            evictable_tokens=int(gauges[total] - gauges[used] - gauges[available]),
            total_tokens=int(gauges[total]),
        )

    host_kv = None
    if "sglang:hicache_host_total_tokens" in gauges:
        host_used = int(gauges["sglang:hicache_host_used_tokens"])
        host_kv = KvPool(
            used_tokens=host_used, evictable_tokens=0, total_tokens=int(gauges["sglang:hicache_host_total_tokens"])
        )
    return EngineLoad(
        rank=rank,
        url=url,
        running_requests=sum(load["num_running_reqs"] for load in dp_loads),
        queued_requests=sum(load["num_waiting_reqs"] for load in dp_loads),
        max_running_requests=sum(load["max_running_requests"] for load in dp_loads),
        full_kv=read_pool("sglang:kv_used_tokens", "sglang:kv_available_tokens", "sglang:max_total_num_tokens"),
        swa_kv=read_pool("sglang:swa_used_tokens", "sglang:swa_available_tokens", "sglang:max_total_num_tokens_swa"),
        host_kv=host_kv,
    )


def _read_gauges(metrics: str) -> dict[str, float]:
    """Gauge values summed over DP ranks; the TP ranks of one DP rank report the same pool."""
    by_dp_rank: dict[str, dict[str, float]] = {}
    for family in text_string_to_metric_families(metrics):
        for sample in family.samples:
            ranks = by_dp_rank.setdefault(sample.name, {})
            dp_rank = sample.labels.get("dp_rank", "0")
            ranks[dp_rank] = max(ranks.get(dp_rank, sample.value), sample.value)
    return {name: sum(ranks.values()) for name, ranks in by_dp_rank.items()}


class EngineLoadMonitor:
    """The engines the router lists, ranked by URL, with their load at most one refresh interval old."""

    def __init__(self, router_url: str, client: httpx.AsyncClient | None = None) -> None:
        self.router_url = router_url
        self._client = client or httpx.AsyncClient(timeout=_REQUEST_TIMEOUT_SECONDS)
        self._lock = asyncio.Lock()
        self._loads: list[EngineLoad] = []
        self._refreshed_at = float("-inf")

    async def get_loads(self) -> list[EngineLoad]:
        async with self._lock:
            if time.monotonic() - self._refreshed_at >= _REFRESH_SECONDS:
                self._loads = await self._fetch_loads()
                self._refreshed_at = time.monotonic()
            return self._loads

    async def _fetch_loads(self) -> list[EngineLoad]:
        workers = (await self._get(f"{self.router_url}/workers")).json()["workers"]
        urls = sorted(router_worker_base_urls([worker["url"] for worker in workers]))
        return list(await asyncio.gather(*(self._fetch_engine(rank, url) for rank, url in enumerate(urls))))

    async def _fetch_engine(self, rank: int, url: str) -> EngineLoad:
        loads, metrics = await asyncio.gather(self._get(f"{url}/v1/loads?include=core"), self._get(f"{url}/metrics"))
        return parse_engine_load(rank, url, loads.json(), metrics.text)

    async def _get(self, url: str) -> httpx.Response:
        try:
            response = await self._client.get(url)
            response.raise_for_status()
        except httpx.HTTPError as error:
            raise EngineUnavailableError(f"engine load is unavailable: {error}") from error
        return response
