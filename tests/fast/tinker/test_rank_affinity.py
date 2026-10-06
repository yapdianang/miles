"""Rank affinity keeps a rollout's turns on one engine and starts new rollouts where there is most room."""

from miles.tinker.engine_load import EngineLoad, KvPool
from miles.tinker.rank_affinity import RankAffinity
from miles.tinker.runtime import MilesBackend

PROMPT = [7, 8, 9]


def _load(url: str, running: int, free_tokens: int = 10**9) -> EngineLoad:
    pool = KvPool(used_tokens=0, evictable_tokens=0, total_tokens=free_tokens)
    return EngineLoad(0, url, running, 0, 64, pool, None, None)


def test_a_continuation_returns_to_the_engine_that_sampled_its_context():
    affinity = RankAffinity()
    affinity.record(PROMPT, [1, 2], "http://busy")

    engine = affinity.choose_engine(PROMPT + [1, 2, 5, 6], [_load("http://busy", 60), _load("http://idle", 0)])

    assert engine == "http://busy"


def test_a_new_rollout_starts_on_the_engine_with_the_most_room():
    affinity = RankAffinity()

    assert affinity.choose_engine(PROMPT, [_load("http://busy", 60), _load("http://idle", 0)]) == "http://idle"
    # Free KV binds before free slots once rollouts are long.
    affinity.mean_context_tokens = 1_000
    loads = [_load("http://idle", 0, free_tokens=2_000), _load("http://busy", 50, free_tokens=10**6)]
    assert affinity.choose_engine(PROMPT, loads) == "http://busy"


def test_samples_of_one_prompt_spread_instead_of_following_each_other():
    affinity = RankAffinity()
    affinity.record(PROMPT, [1, 2], "http://busy")

    assert affinity.choose_engine(PROMPT, [_load("http://busy", 60), _load("http://idle", 0)]) == "http://idle"


def test_a_rollout_whose_engine_left_the_router_starts_over():
    affinity = RankAffinity()
    affinity.record(PROMPT, [1, 2], "http://gone")

    assert affinity.choose_engine(PROMPT + [1, 2, 5], [_load("http://idle", 0)]) == "http://idle"


def test_the_oldest_contexts_are_evicted():
    affinity = RankAffinity(max_contexts=1)
    affinity.record(PROMPT, [1], "http://busy")
    affinity.record([4], [5], "http://busy")

    loads = [_load("http://busy", 60), _load("http://idle", 0)]
    assert affinity.choose_engine(PROMPT + [1, 3], loads) == "http://idle"


class FakeEngines:
    def __init__(self) -> None:
        self.urls: list[str] = []

    async def post(self, url: str, request: dict) -> dict:
        self.urls.append(url)
        meta = {"output_token_logprobs": [(-1.0, 11), (-1.0, 12)], "finish_reason": {"type": "stop"}}
        return {"meta_info": meta}


def _payload(prompt: list[int]) -> dict:
    return {
        "prompt_tokens": prompt,
        "sampling_params": {"max_tokens": 2},
        "num_samples": 1,
        "prompt_logprobs": False,
        "topk_prompt_logprobs": 0,
    }


async def test_the_backend_posts_each_turn_to_the_rollout_engine(monkeypatch):
    engines = FakeEngines()
    monkeypatch.setattr("miles.tinker.runtime.post", engines.post)
    backend = MilesBackend(None, "http://router", rank_affinity=RankAffinity())
    loads = [_load("http://engine-a", 10), _load("http://engine-b", 0)]

    async def engine_loads():
        # engine-b fills up after the first turn; the second turn must still follow its context to engine-b
        snapshot, loads[1] = list(loads), _load("http://engine-b", 63)
        return snapshot

    backend.engine_loads = engine_loads
    await backend.sample(_payload(PROMPT), None)
    await backend.sample(_payload(PROMPT + [11, 12, 3]), None)

    assert engines.urls == ["http://engine-b/generate", "http://engine-b/generate"]


async def test_without_affinity_the_router_places_every_request(monkeypatch):
    engines = FakeEngines()
    monkeypatch.setattr("miles.tinker.runtime.post", engines.post)

    await MilesBackend(None, "http://router").sample(_payload(PROMPT), None)

    assert engines.urls == ["http://router/generate"]
