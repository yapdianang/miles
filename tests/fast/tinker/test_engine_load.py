"""Engine load combines SGLang's /v1/loads and pool gauges per engine the router lists."""

import httpx
import pytest

from miles.tinker.core.types import EngineUnavailableError
from miles.tinker.engine_load import EngineLoad, EngineLoadMonitor, KvPool, parse_engine_load

LOADS = {"loads": [{"dp_rank": 0, "num_running_reqs": 40, "num_waiting_reqs": 6, "max_running_requests": 64}]}

# Two TP ranks of one DP rank report the same pools; SWA and HiCache gauges appear when those are on.
METRICS = """\
# TYPE sglang:max_total_num_tokens gauge
sglang:max_total_num_tokens{tp_rank="0"} 17096192.0
sglang:max_total_num_tokens{tp_rank="1"} 17096192.0
# TYPE sglang:kv_used_tokens gauge
sglang:kv_used_tokens{tp_rank="0"} 2000000.0
sglang:kv_used_tokens{tp_rank="1"} 2000000.0
# TYPE sglang:kv_available_tokens gauge
sglang:kv_available_tokens{tp_rank="0"} 14096192.0
sglang:kv_available_tokens{tp_rank="1"} 14096192.0
# TYPE sglang:max_total_num_tokens_swa gauge
sglang:max_total_num_tokens_swa{tp_rank="0"} 512768.0
# TYPE sglang:swa_used_tokens gauge
sglang:swa_used_tokens{tp_rank="0"} 6000.0
# TYPE sglang:swa_available_tokens gauge
sglang:swa_available_tokens{tp_rank="0"} 500000.0
# TYPE sglang:hicache_host_used_tokens gauge
sglang:hicache_host_used_tokens{tp_rank="0"} 1000000.0
# TYPE sglang:hicache_host_total_tokens gauge
sglang:hicache_host_total_tokens{tp_rank="0"} 34000000.0
"""


def test_pools_split_active_and_cached_tokens_without_double_counting_tp_ranks():
    load = parse_engine_load(0, "http://engine-a", LOADS, METRICS)

    assert (load.running_requests, load.queued_requests, load.max_running_requests) == (40, 6, 64)
    assert load.full_kv == KvPool(used_tokens=2_000_000, evictable_tokens=1_000_000, total_tokens=17_096_192)
    assert load.swa_kv == KvPool(used_tokens=6_000, evictable_tokens=6_768, total_tokens=512_768)
    assert load.host_kv == KvPool(used_tokens=1_000_000, evictable_tokens=0, total_tokens=34_000_000)


def test_an_engine_without_swa_or_hicache_reports_only_the_full_pool():
    metrics = "\n".join(line for line in METRICS.splitlines() if "swa" not in line and "hicache" not in line)

    load = parse_engine_load(0, "http://engine-a", LOADS, metrics)

    assert (load.swa_kv, load.host_kv) == (None, None)


def test_room_is_the_fewer_of_free_request_slots_and_kv_not_held_by_running_requests():
    pool = KvPool(used_tokens=0, evictable_tokens=600, total_tokens=1_000)
    load = EngineLoad(0, "http://engine-a", 2, 1, 8, pool, None, None)

    assert load.measure_room(100) == 5
    assert load.measure_room(500) == 2


WORKERS = {
    "workers": [
        {"url": "http://engine-b", "is_healthy": True},
        {"url": "http://engine-a", "is_healthy": True},
        {"url": "http://engine-sick", "is_healthy": False},
    ]
}


def _router_and_engines(requests: list[str], down: str = "") -> httpx.MockTransport:
    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(str(request.url))
        if request.url.path == "/workers":
            return httpx.Response(200, json=WORKERS)
        if request.url.host == down:
            return httpx.Response(502)
        if request.url.path == "/v1/loads":
            return httpx.Response(200, json=LOADS)
        return httpx.Response(200, text=METRICS)

    return httpx.MockTransport(handle)


async def test_the_monitor_ranks_healthy_engines_by_url_and_reuses_a_fresh_snapshot():
    requests: list[str] = []
    monitor = EngineLoadMonitor("http://router", httpx.AsyncClient(transport=_router_and_engines(requests)))

    loads = await monitor.get_loads()
    await monitor.get_loads()

    assert [(load.rank, load.url) for load in loads] == [(0, "http://engine-a"), (1, "http://engine-b")]
    assert len(requests) == 5


async def test_an_engine_that_does_not_answer_is_left_out():
    monitor = EngineLoadMonitor("http://router", httpx.AsyncClient(transport=_router_and_engines([], "engine-b")))

    assert [load.url for load in await monitor.get_loads()] == ["http://engine-a"]


async def test_an_unreachable_router_is_reported_once_per_interval():
    requests: list[str] = []

    def refuse(request: httpx.Request) -> httpx.Response:
        requests.append(str(request.url))
        return httpx.Response(502)

    monitor = EngineLoadMonitor("http://router", httpx.AsyncClient(transport=httpx.MockTransport(refuse)))

    for _ in range(2):
        with pytest.raises(EngineUnavailableError):
            await monitor.get_loads()
    assert requests == ["http://router/workers"]
