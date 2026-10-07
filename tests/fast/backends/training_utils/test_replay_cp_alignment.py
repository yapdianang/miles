"""R3 under zigzag context parallelism: every local token replays the routes recorded for it.

``fill_replay_data`` slices rollout routes per sample the way ``get_batch`` slices tokens. Both run
here under the same CP/TP state with routes tagged by their token, so a misaligned row is visible.
"""

from types import SimpleNamespace

import pytest
import torch
from tests.ci.ci_register import register_cpu_ci

import miles.backends.training_utils.data.context_parallel as cp_utils_mod
import miles.backends.training_utils.data.rollout as data_mod
import miles.backends.training_utils.replay.data as replay_mod
from miles.backends.training_utils.data.rollout import get_batch
from miles.backends.training_utils.replay.data import fill_replay_data

register_cpu_ci(est_time=5, suite="stage-a-cpu", labels=[])


class _FakeIterator:
    def __init__(self, batch: dict):
        self._batch = batch
        self.rollout_data = {}

    def get_next(self, keys):
        return {key: self._batch.get(key) for key in keys}

    def reset(self):
        pass


@pytest.mark.parametrize("cp_size", [1, 2, 4])
@pytest.mark.parametrize("tp_size", [1, 2])
def test_local_routes_belong_to_local_tokens(monkeypatch, cp_size, tp_size):
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self, *args, **kwargs: self, raising=False)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: "cpu", raising=False)
    # token values encode (sample, position) and start at 1, so zero marks padding
    tokens = [1000 * (i + 1) + torch.arange(n) for i, n in enumerate([301, 40, 7])]
    last_tokens = torch.stack([t[-1] for t in tokens])
    # rollout routes cover every token but the last: [T - 1, layers, topk], tagged with their token
    routes = [t[:-1].view(-1, 1, 1).expand(-1, 3, 2).clone() for t in tokens]
    args = SimpleNamespace(
        qkv_format="thd", allgather_cp=False, data_pad_size_multiplier=128, sequence_parallel=tp_size > 1
    )

    for cp_rank in range(cp_size):
        for tp_rank in range(tp_size):
            state = SimpleNamespace(
                cp=SimpleNamespace(rank=cp_rank, size=cp_size), tp=SimpleNamespace(rank=tp_rank, size=tp_size)
            )
            for module in (data_mod, cp_utils_mod, replay_mod):
                monkeypatch.setattr(module, "get_parallel_state", lambda state=state: state)
            rollout = {
                "tokens": tokens,
                "loss_masks": [torch.ones(t.numel() - 1, dtype=torch.int) for t in tokens],
                "total_lengths": [t.numel() for t in tokens],
                "response_lengths": [t.numel() - 1 for t in tokens],
            }
            batch = get_batch(_FakeIterator(rollout), list(rollout), pad_multiplier=128)
            recorded = []
            fill_replay_data(
                args=args,
                models=None,
                data_iterator=[_FakeIterator({"tokens": tokens, "routes": routes})],
                num_microbatches=[1],
                rollout_data={"routes": routes},
                data_key="routes",
                replay_list=None,
                register_replay_list_func=lambda _replays, data, recorded=recorded, **_: recorded.append(data),
            )

            # sequence parallelism hands each TP rank a contiguous slice of the local tokens
            local_tokens = batch["tokens"][0].chunk(tp_size)[tp_rank]
            [local_routes] = recorded
            unrouted = (local_tokens == 0) | torch.isin(local_tokens, last_tokens)
            expected = torch.where(unrouted, -1, local_tokens).view(-1, 1, 1).expand(-1, 3, 2)
            assert torch.equal(local_routes, expected)
