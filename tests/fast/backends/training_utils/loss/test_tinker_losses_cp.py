"""Tinker losses under zigzag CP=2 and CP=4 over gloo against CP=1 on the same datums.

Each rank sees its zigzag slice of every datum's log-probs. The client must still get whole-datum
log-probs and losses, and each rank's gradient must be its slice of the CP=1 gradient. Score centering
runs on zigzag-sliced logits against the dense oracle of test_tinker_sampling_supports, and every loss
runs under --tinker-fused-loss on zigzag-sliced hidden states against CP=1.
"""

import sys
from argparse import Namespace
from functools import partial
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from tests.ci.ci_register import register_cpu_ci
from tests.fast.backends.training_utils.loss import test_tinker_fused_loss as fused
from tests.fast.backends.training_utils.loss.test_tinker_sampling_supports import (
    TOKENS,
    VOCAB,
    _args,
    _reference_cross_entropy,
    _reference_logprobs,
    _reference_score_centering,
    _score_centering_batch,
)
from tests.fast.dist_utils import init_gloo, run_multiprocess

import miles.backends.training_utils.data.context_parallel as cp_utils_mod
from miles.backends.training_utils.data.context_parallel import slice_log_prob_with_cp, slice_with_cp
from miles.backends.training_utils.loss.hub import logit_processors, tinker_losses

register_cpu_ci(est_time=20, suite="stage-a-cpu", labels=[])

# Tinker targets may cover the prompt too: response_length can reach total_length - 1
TOTAL_LENGTHS, RESPONSE_LENGTHS = [17, 9, 30], [16, 5, 3]


def _use_state(cp_rank, cp_size, group):
    state = SimpleNamespace(
        cp=SimpleNamespace(rank=cp_rank, size=cp_size, group=group), tp=SimpleNamespace(rank=0, size=1, group=None)
    )
    for module in (tinker_losses, cp_utils_mod, logit_processors):
        module.get_parallel_state = lambda: state


def _batch(loss_fn):
    generator = torch.Generator().manual_seed(0)
    responses = [torch.randn(n, generator=generator, dtype=torch.float64) for n in RESPONSE_LENGTHS]
    return {
        "loss_fn": loss_fn,
        "loss_weights": [r.tolist() for r in responses],
        "advantages": [(2 * r).tolist() for r in responses],
        "rollout_log_probs": [-(r.abs() + 0.1) for r in responses],
        "loss_masks": [torch.tensor([i % 3 != 0 for i in range(n)], dtype=torch.int) for n in RESPONSE_LENGTHS],
        "total_lengths": TOTAL_LENGTHS,
        "response_lengths": RESPONSE_LENGTHS,
        "sample_indices": [3, 5, 8],
    }


def _run(loss_fn, batch, log_probs):
    leaves = [lp.clone().requires_grad_() for lp in log_probs]
    tinker_losses._target_logprobs = lambda _args, _batch, _logits: leaves
    loss, outputs = tinker_losses.TINKER_LOSS_FUNCTIONS[loss_fn](Namespace(qkv_format="thd"), batch, None, None)
    return loss, outputs["per_datum"], torch.autograd.grad(loss, leaves)


def _local(values, total_lengths=TOTAL_LENGTHS, response_lengths=RESPONSE_LENGTHS):
    return [slice_log_prob_with_cp(v, t, r) for v, t, r in zip(values, total_lengths, response_lengths, strict=True)]


def _worker(rank, world_size, port, loss_fn):
    init_gloo(rank, world_size, port=port)
    try:
        log_probs = [-(torch.arange(n, dtype=torch.float64) + 1) / 7 for n in RESPONSE_LENGTHS]
        batch = _batch(loss_fn)
        _use_state(0, 1, None)
        want_loss, want_outputs, want_grads = _run(loss_fn, batch, log_probs)

        _use_state(rank, world_size, dist.group.WORLD)
        # get_rollout_data has already sliced the sampler log-probs to the local layout
        local_batch = batch | {"rollout_log_probs": _local(batch["rollout_log_probs"])}
        loss, outputs, grads = _run(loss_fn, local_batch, _local(log_probs))

        loss = loss.detach().clone()
        dist.all_reduce(loss)
        torch.testing.assert_close(loss, want_loss.detach())
        for got, want in zip(outputs, want_outputs, strict=True):
            assert got["sample_index"] == want["sample_index"]
            torch.testing.assert_close(got["logprobs"], want["logprobs"])
            torch.testing.assert_close(got["loss"], want["loss"])
        for got, want in zip(grads, _local(want_grads), strict=True):
            torch.testing.assert_close(got, want)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("cp_size", [2, 4])
@pytest.mark.parametrize("loss_fn", sorted(set(tinker_losses.TINKER_LOSS_FUNCTIONS) - {"score_centering"}))
def test_cp_matches_cp1(loss_fn, cp_size):
    run_multiprocess(partial(_worker, loss_fn=loss_fn), world_size=cp_size)


def _valid(ids, log_probs):
    return [(row[row >= 0].tolist(), lp[row >= 0].tolist()) for row, lp in zip(ids, log_probs, strict=True)]


def test_support_head_of_zigzag_rows_concatenates_both_ranges():
    """Under CP a rank's target rows are two ranges; their supports must come back in that order."""
    offsets = torch.tensor([0, 2, 2, 5, 6, 9, 9, 10])
    batch = {
        "rollout_sampling_mask_ids": [torch.arange(10) + 100],
        "rollout_sampling_mask_offsets": [offsets],
        "rollout_sampling_mask_log_probs": [-torch.arange(10) / 10],
    }
    rows = [*range(1, 3), *range(4, 7)]
    zigzag = _valid(*tinker_losses._support_head(batch, 0, rows))
    whole = _valid(*tinker_losses._support_head(batch, 0, range(0, 7)))
    assert zigzag == [whole[row] for row in rows]


def _score_centering_worker(rank, world_size, port):
    init_gloo(rank, world_size, port=port)
    try:
        sys.modules["megatron.core.fusions.fused_cross_entropy"] = SimpleNamespace(
            fused_vocab_parallel_cross_entropy=_reference_cross_entropy
        )
        lengths = [len(tokens) for tokens in TOKENS]
        generator = torch.Generator().manual_seed(1)
        logits = torch.randn(1, sum(lengths), VOCAB, generator=generator, dtype=torch.float64, requires_grad=True)
        batch = _score_centering_batch("tis")
        want_loss = _reference_score_centering(logits, batch, 2.0)
        (want_grad,) = torch.autograd.grad(want_loss, logits)

        _use_state(rank, world_size, dist.group.WORLD)

        def zigzag(rows):
            return torch.cat([slice_with_cp(seq, 0.0, "thd") for seq in rows[0].split(lengths)]).unsqueeze(0)

        local_logits = zigzag(logits.detach()).float().requires_grad_()
        local_batch = batch | {
            "rollout_log_probs": _local(batch["rollout_log_probs"], batch["total_lengths"], batch["response_lengths"])
        }
        loss, outputs = tinker_losses.score_centering_loss_function(_args(-1), local_batch, local_logits, None)
        (grad,) = torch.autograd.grad(loss, local_logits)

        loss = loss.detach().double()
        dist.all_reduce(loss)
        torch.testing.assert_close(loss, want_loss.detach(), rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(grad.double(), zigzag(want_grad), rtol=1e-4, atol=1e-5)
        for output, reference in zip(outputs["per_datum"], _reference_logprobs(logits.detach().float()), strict=True):
            torch.testing.assert_close(output["logprobs"], reference)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("cp_size", [2, 4])
def test_score_centering_under_cp_matches_the_dense_oracle(cp_size):
    run_multiprocess(_score_centering_worker, world_size=cp_size)


def _fused_worker(rank, world_size, port, loss_fn):
    init_gloo(rank, world_size, port=port)
    try:
        sys.modules["megatron.core.fusions.fused_cross_entropy"] = SimpleNamespace(
            fused_vocab_parallel_cross_entropy=fused._reference_cross_entropy
        )
        hidden, weight = fused._inputs(torch.Generator().manual_seed(11))
        lengths = [len(tokens) for tokens in fused.TOKENS]
        batch = fused._batch() | {"loss_fn": loss_fn}

        def run(rows, run_batch):
            rows, columns = rows.clone().requires_grad_(), weight.clone().requires_grad_()
            loss_function = tinker_losses.TINKER_LOSS_FUNCTIONS[loss_fn]
            loss, outputs = loss_function(fused._args(2), run_batch | {"output_weight": columns}, rows, None)
            loss.backward()
            return loss.detach(), outputs["per_datum"], rows.grad, columns.grad

        def zigzag(rows):
            return torch.cat([slice_with_cp(seq, 0.0, "thd") for seq in rows[0].split(lengths)]).unsqueeze(0)

        _use_state(0, 1, None)
        want_loss, want_outputs, want_hidden_grad, want_weight_grad = run(hidden, batch)
        _use_state(rank, world_size, dist.group.WORLD)
        local_rollout = _local(batch["rollout_log_probs"], batch["total_lengths"], batch["response_lengths"])
        loss, outputs, hidden_grad, weight_grad = run(zigzag(hidden), batch | {"rollout_log_probs": local_rollout})

        dist.all_reduce(loss)
        dist.all_reduce(weight_grad)
        torch.testing.assert_close(loss, want_loss)
        torch.testing.assert_close(weight_grad, want_weight_grad)
        torch.testing.assert_close(hidden_grad, zigzag(want_hidden_grad))
        for got, want in zip(outputs, want_outputs, strict=True):
            torch.testing.assert_close(got["logprobs"], want["logprobs"])
            torch.testing.assert_close(got["loss"], want["loss"])
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("cp_size", [2, 4])
@pytest.mark.parametrize("loss_fn", sorted(tinker_losses.TINKER_LOSS_FUNCTIONS))
def test_fused_loss_under_cp_matches_cp1(loss_fn, cp_size):
    """Every loss, score centering included, scores top-k/top-p supports from zigzag hidden states."""
    run_multiprocess(partial(_fused_worker, loss_fn=loss_fn), world_size=cp_size)
