"""Keep every turn of a rollout on the engine that holds its KV cache."""

from collections import Counter, OrderedDict

import numpy as np

from miles.tinker.engine_load import EngineLoad
from miles.tinker.sampler_records import prefix_hashes

_MAX_CONTEXTS = 1 << 16
# Weight of each new context length in the running estimate of a rollout's length.
_CONTEXT_LENGTH_DECAY = 0.01


class RankAffinity:
    """Engines by sampled context: a prompt that extends an earlier sample's prompt and output goes where it ran.

    A prompt that extends no recorded context starts a rollout. It goes to the engine with the most room, the
    fewer of its free request slots and its free KV in rollouts of the recent mean context length, so samples
    of one prompt spread out instead of following each other.
    """

    def __init__(self, max_contexts: int = _MAX_CONTEXTS) -> None:
        self.max_contexts = max_contexts
        self._engines: OrderedDict[int, str] = OrderedDict()
        self._keys = np.empty(0, dtype=np.uint64)
        self.mean_context_tokens = 0.0
        # Rollouts started since the snapshot was taken, which its load does not show yet.
        self._snapshot: list[EngineLoad] | None = None
        self._started: Counter[str] = Counter()

    def choose_engine(self, prompt_tokens: list[int], loads: list[EngineLoad]) -> str:
        if loads is not self._snapshot:
            self._snapshot, self._started = loads, Counter()
        engine = self._find_engine(prompt_tokens)
        if engine is not None and any(load.url == engine for load in loads):
            return engine
        rollout_tokens = max(self.mean_context_tokens, len(prompt_tokens), 1)
        engine = max(loads, key=lambda load: load.measure_room(rollout_tokens) - self._started[load.url]).url
        self._started[engine] += 1
        return engine

    def record(self, prompt_tokens: list[int], output_tokens: list[int], engine: str) -> None:
        context = np.asarray(prompt_tokens + output_tokens, dtype=np.int64)
        key = int(prefix_hashes(context)[-1])
        if key not in self._engines:
            self._keys = np.insert(self._keys, np.searchsorted(self._keys, np.uint64(key)), np.uint64(key))
        self._engines[key] = engine
        self._engines.move_to_end(key)
        if len(self._engines) > self.max_contexts:
            evicted, _ = self._engines.popitem(last=False)
            self._keys = np.delete(self._keys, np.searchsorted(self._keys, np.uint64(evicted)))
        self.mean_context_tokens += _CONTEXT_LENGTH_DECAY * (len(context) - self.mean_context_tokens)

    def _find_engine(self, prompt_tokens: list[int]) -> str | None:
        """The engine of the longest recorded context that prefixes the prompt."""
        if not len(self._keys) or not prompt_tokens:
            return None
        hashes = prefix_hashes(np.asarray(prompt_tokens, dtype=np.int64))
        found = self._keys[np.minimum(np.searchsorted(self._keys, hashes), len(self._keys) - 1)] == hashes
        hits = np.flatnonzero(found)
        if not len(hits):
            return None
        key = int(hashes[hits[-1]])
        self._engines.move_to_end(key)
        return self._engines[key]
