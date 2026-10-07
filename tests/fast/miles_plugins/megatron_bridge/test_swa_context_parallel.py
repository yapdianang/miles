"""CPU tests for the sliding-window halo exchange under zigzag context parallelism.

Rows carry their global position, so a halo row fetched from the wrong rank or out of order is
visible. The attention checks compare a float64 MiMo SWA (window plus learnable sink) over whole
sequences with per-rank attention over [halo | chunk] under TE's bottom-right window, in values and
gradients. The all-to-all is replayed in Python, then run for real over gloo.
"""

from functools import partial
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from tests.ci.ci_register import register_cpu_ci
from tests.fast.dist_utils import init_gloo, run_multiprocess

import miles.backends.training_utils.data.context_parallel as cp_utils_mod
import miles.backends.training_utils.data.rollout as data_mod
from miles.backends.training_utils.data.rollout import get_batch
from miles_plugins.megatron_bridge.swa_context_parallel import exchange_swa_halo, plan_swa_halo

register_cpu_ci(est_time=30, suite="stage-a-cpu", labels=[])

# one long sequence, a pack with chunks shorter than the window, and a pack of short documents
SEQ_LENS = [[8192], [3000, 17, 1, 777], [64] * 5 + [2048]]


def _cu_seqlens(seq_lens, cp_size):
    """Global THD boundaries of sequences padded to a multiple of 2 * cp, as Miles packs them."""
    padded = [-(-n // (2 * cp_size)) * 2 * cp_size for n in seq_lens]
    return tuple(torch.tensor([0, *padded]).cumsum(0).tolist())


def _plans(cu_seqlens, cp_size, window):
    return [plan_swa_halo(cu_seqlens, cp_rank=r, cp_size=cp_size, window=window, device="cpu") for r in range(cp_size)]


def _local_positions(cu_seqlens, cp_rank, cp_size):
    """Global rows rank ``cp_rank`` holds: chunks r and 2 * cp - 1 - r of every sequence."""
    rows = []
    for start, end in zip(cu_seqlens[:-1], cu_seqlens[1:], strict=True):
        chunk = (end - start) // (2 * cp_size)
        for c in (cp_rank, 2 * cp_size - 1 - cp_rank):
            rows.append(torch.arange(start + c * chunk, start + (c + 1) * chunk))
    return torch.cat(rows)


def _exchange(local_rows, plans):
    """``exchange_swa_halo`` on every rank with ``all_to_all_single`` replayed in Python (differentiable)."""
    sent = [rows[plan.send_rows].split(list(plan.send_splits)) for rows, plan in zip(local_rows, plans, strict=True)]
    extended = []
    for dst, (rows, plan) in enumerate(zip(local_rows, plans, strict=True)):
        received = [sent[src][dst] for src in range(len(plans))]
        assert [r.shape[0] for r in received] == list(plan.recv_splits), "sender and receiver disagree"
        extended.append(torch.cat([rows, *received])[plan.kv_rows])
    return extended


def _segments(bounds):
    bounds = bounds.tolist()
    return list(zip(bounds[:-1], bounds[1:], strict=True))


@pytest.mark.parametrize("cp_size", [1, 2, 4, 8])
@pytest.mark.parametrize("window", [128, 5])
@pytest.mark.parametrize("seq_lens", SEQ_LENS)
def test_each_chunk_attends_exactly_its_window(cp_size, window, seq_lens):
    cu_seqlens = _cu_seqlens(seq_lens, cp_size)
    plans = _plans(cu_seqlens, cp_size, window)
    local = [_local_positions(cu_seqlens, r, cp_size) for r in range(cp_size)]
    starts = torch.tensor(cu_seqlens[:-1])

    for rank, (plan, keys) in enumerate(zip(plans, _exchange(local, plans), strict=True)):
        q_segments, kv_segments = _segments(plan.cu_seqlens_q), _segments(plan.cu_seqlens_kv)
        assert q_segments[-1][1] == local[rank].numel(), "query segments must tile the local rows"
        for (q_lo, q_hi), (kv_lo, kv_hi) in zip(q_segments, kv_segments, strict=True):
            queries = local[rank][q_lo:q_hi]
            seq_start = int(starts[starts <= queries[0]].max())
            first_key = max(seq_start, int(queries[0]) - window)
            assert torch.equal(queries, torch.arange(int(queries[0]), int(queries[0]) + q_hi - q_lo))
            assert torch.equal(keys[kv_lo:kv_hi], torch.arange(first_key, int(queries[-1]) + 1))
        assert plan.max_seqlen_q == max(hi - lo for lo, hi in q_segments)
        assert plan.max_seqlen_kv == max(hi - lo for lo, hi in kv_segments)


@pytest.mark.parametrize("cp_size", [2, 4, 8])
def test_traffic_is_bounded_by_the_window_not_the_sequence(cp_size):
    """A long sequence moves at most 2 * window rows into a rank, from its two zigzag neighbours."""
    window = 128
    for seq_len in (2 * cp_size * 1024, 2**20):
        for rank, plan in enumerate(_plans(_cu_seqlens([seq_len], cp_size), cp_size, window)):
            peers = {src for src, rows in enumerate(plan.recv_splits) if rows}
            assert peers <= {rank - 1, rank, rank + 1}
            assert sum(plan.recv_splits) == (window if rank == 0 else 2 * window)


def test_unpadded_sequences_are_rejected():
    with pytest.raises(ValueError, match="multiple of 2 \\* cp"):
        plan_swa_halo((0, 100), cp_rank=0, cp_size=4, window=128, device="cpu")


def _attend(q, k, v, sink, window):
    """MiMo SWA with TE's bottom-right window: query a attends keys [a + skv - sq - window, a + skv - sq]."""
    diagonal = torch.arange(q.shape[0]).unsqueeze(1) + k.shape[0] - q.shape[0]
    keys = torch.arange(k.shape[0]).unsqueeze(0)
    allowed = (keys <= diagonal) & (keys >= diagonal - window)
    groups = q.shape[1] // k.shape[1]
    k, v = k.repeat_interleave(groups, 1), v.repeat_interleave(groups, 1)
    scores = torch.einsum("qhd,khd->hqk", q, k) / q.shape[-1] ** 0.5
    scores = scores.masked_fill(~allowed, -torch.inf)
    # the learnable sink is one more logit per head in every softmax denominator
    logits = torch.cat([scores, sink.view(-1, 1, 1).expand(-1, q.shape[0], 1)], dim=-1)
    return torch.einsum("hqk,khd->qhd", logits.softmax(-1)[..., :-1], v)


@pytest.mark.parametrize("cp_size", [2, 4])
@pytest.mark.parametrize("seq_lens", [[512], [200, 9, 64]])
def test_cp_attention_matches_whole_sequence_attention_with_sink(cp_size, seq_lens):
    window, heads, groups, dk, dv = 16, 4, 2, 6, 4
    cu_seqlens = _cu_seqlens(seq_lens, cp_size)
    generator = torch.Generator().manual_seed(0)
    q, k = (torch.randn(cu_seqlens[-1], h, dk, generator=generator, dtype=torch.float64) for h in (heads, groups))
    v = torch.randn(cu_seqlens[-1], groups, dv, generator=generator, dtype=torch.float64)
    sink = torch.randn(heads, generator=generator, dtype=torch.float64)
    grad_out = torch.randn(cu_seqlens[-1], heads, dv, generator=generator, dtype=torch.float64)
    leaves = [t.requires_grad_() for t in (q, k, v, sink)]

    whole = torch.cat(
        [
            _attend(q[lo:hi], k[lo:hi], v[lo:hi], sink, window)
            for lo, hi in zip(cu_seqlens[:-1], cu_seqlens[1:], strict=True)
        ]
    )
    plans = _plans(cu_seqlens, cp_size, window)
    local = [_local_positions(cu_seqlens, r, cp_size) for r in range(cp_size)]
    ext_k, ext_v = _exchange([k[rows] for rows in local], plans), _exchange([v[rows] for rows in local], plans)
    split = torch.empty_like(whole)
    for rows, plan, keys, values in zip(local, plans, ext_k, ext_v, strict=True):
        queries = q[rows]
        split[rows] = torch.cat(
            [
                _attend(queries[q_lo:q_hi], keys[kv_lo:kv_hi], values[kv_lo:kv_hi], sink, window)
                for (q_lo, q_hi), (kv_lo, kv_hi) in zip(
                    _segments(plan.cu_seqlens_q), _segments(plan.cu_seqlens_kv), strict=True
                )
            ]
        )

    torch.testing.assert_close(split, whole, rtol=0, atol=1e-12)
    want = torch.autograd.grad((whole * grad_out).sum(), leaves)
    got = torch.autograd.grad((split * grad_out).sum(), leaves)
    for name, g, w in zip(("dq", "dk", "dv", "dsink"), got, want, strict=True):
        torch.testing.assert_close(g, w, rtol=0, atol=1e-12, msg=name)


def _gloo_exchange_worker(rank, world_size, port, seq_lens):
    init_gloo(rank, world_size, port=port)
    try:
        window = 32
        cu_seqlens = _cu_seqlens(seq_lens, world_size)
        plans = _plans(cu_seqlens, world_size, window)
        local = [_local_positions(cu_seqlens, r, world_size).double() for r in range(world_size)]
        # keys and values of different widths, every element tagged with its row's global position
        key = local[rank].view(-1, 1, 1).expand(-1, 2, 3).clone().requires_grad_()
        value = (local[rank].view(-1, 1, 1).expand(-1, 2, 2) + 0.5).clone().requires_grad_()

        ext_key, ext_value = exchange_swa_halo(key, value, plans[rank], dist.group.WORLD)
        expected = _exchange(local, plans)[rank]
        assert torch.equal(ext_key, expected.view(-1, 1, 1).expand(-1, 2, 3))
        assert torch.equal(ext_value, expected.view(-1, 1, 1).expand(-1, 2, 2) + 0.5)

        # each rank weighs its extended rows differently, so a gradient returned to the wrong owner shows
        def weights(r, rows):
            return (r + 1) * (torch.arange(rows, dtype=torch.float64) + 1)

        ((ext_key.sum((1, 2)) + 2 * ext_value.sum((1, 2))) * weights(rank, ext_key.shape[0])).sum().backward()
        tagged = [rows.clone().requires_grad_() for rows in local]
        reference = sum((ext * weights(r, ext.shape[0])).sum() for r, ext in enumerate(_exchange(tagged, plans)))
        row_grad = torch.autograd.grad(reference, tagged[rank])[0]
        torch.testing.assert_close(key.grad, row_grad.view(-1, 1, 1).expand(-1, 2, 3), rtol=0, atol=0)
        torch.testing.assert_close(value.grad, 2 * row_grad.view(-1, 1, 1).expand(-1, 2, 2), rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("world_size", [2, 4])
@pytest.mark.parametrize("seq_lens", [(1024,), (300, 7, 96)])
def test_gloo_exchange_delivers_halos_and_returns_their_gradients(world_size, seq_lens):
    run_multiprocess(partial(_gloo_exchange_worker, seq_lens=seq_lens), world_size=world_size)


def _rollout(tokens):
    return {
        "tokens": tokens,
        "loss_masks": [torch.ones(t.numel() - 1, dtype=torch.int) for t in tokens],
        "total_lengths": [t.numel() for t in tokens],
        "response_lengths": [t.numel() - 1 for t in tokens],
    }


class _FakeIterator:
    def __init__(self, batch: dict):
        self._batch = batch
        self.rollout_data = {}

    def get_next(self, keys):
        return {key: self._batch.get(key) for key in keys}


@pytest.mark.parametrize("cp_size", [2, 4])
def test_plan_matches_the_layout_get_batch_packs(monkeypatch, cp_size):
    """Rows of get_batch's THD stream (trailing pad included) sit where the plan expects them."""
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self, *args, **kwargs: self, raising=False)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: "cpu", raising=False)
    lengths = [301, 40, 7]
    # token values encode (sample, position) and start at 1, so get_batch's zero padding stays visible
    samples = [1000 * (i + 1) + torch.arange(n) for i, n in enumerate(lengths)]
    stream = None
    for cp_rank in range(cp_size):
        state = SimpleNamespace(cp=SimpleNamespace(rank=cp_rank, size=cp_size), tp=SimpleNamespace(size=1))
        monkeypatch.setattr(data_mod, "get_parallel_state", lambda state=state: state)
        monkeypatch.setattr(cp_utils_mod, "get_parallel_state", lambda state=state: state)
        batch = get_batch(_FakeIterator(_rollout(samples)), list(_rollout(samples)), pad_multiplier=128)
        cu_seqlens = batch["cu_seqlens_host"]
        if stream is None:
            stream = torch.zeros(cu_seqlens[-1], dtype=torch.long)
            for start, sample in zip(cu_seqlens, samples, strict=False):
                stream[start : start + sample.numel()] = sample
        assert torch.equal(batch["tokens"][0], stream[_local_positions(cu_seqlens, cp_rank, cp_size)])
        # the trailing pad segment is a whole multiple of 2 * cp as well
        plan_swa_halo(cu_seqlens, cp_rank=cp_rank, cp_size=cp_size, window=16, device="cpu")
