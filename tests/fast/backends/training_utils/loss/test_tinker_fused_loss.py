"""--tinker-fused-loss scores hidden states against the output weight and matches the logits path it replaces."""

import sys
from argparse import Namespace
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

from miles.backends.training_utils.loss.hub import logit_processors, tinker_losses
from miles.backends.training_utils.loss.hub.score_centering import (
    output_selected_log_probs_and_entropy,
    selected_log_probs_and_entropy,
)
from miles.backends.training_utils.parallel import GroupInfo, ParallelState, set_parallel_state
from miles.utils.sampling_mask import PartialSamplingMask

VOCAB, HIDDEN, TEMPERATURE = 12, 5, 0.7
# target i of a datum scores tokens[i + 1]; an empty support keeps the full vocabulary
TOKENS = [[3, 4, 5, 6, 7], [8, 9, 1], [2, 10, 0, 11]]
SUPPORTS = [[[], [5, 0, 11], [6], [7, 2]], [[], [1, 3, 4]], [[10, 9], [], [11, 0, 1, 2]]]


def _reference_cross_entropy(logits: torch.Tensor, tokens: torch.Tensor, _group: object) -> torch.Tensor:
    """Megatron's vocab-parallel cross entropy on one rank: -log_softmax(logits)[token]."""
    return torch.logsumexp(logits, dim=-1) - logits.gather(-1, tokens.unsqueeze(-1)).squeeze(-1)


@pytest.fixture
def single_rank(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "megatron.core.fusions.fused_cross_entropy",
        SimpleNamespace(fused_vocab_parallel_cross_entropy=_reference_cross_entropy),
    )
    parallel = SimpleNamespace(tp=SimpleNamespace(rank=0, size=1, group=None), cp=SimpleNamespace(rank=0, size=1))
    monkeypatch.setattr(logit_processors, "get_parallel_state", lambda: parallel)
    monkeypatch.setattr(tinker_losses, "get_parallel_state", lambda: parallel)


def _args(chunk_size: int) -> Namespace:
    return Namespace(
        qkv_format="thd",
        rollout_temperature=TEMPERATURE,
        true_on_policy_mode=False,
        log_probs_chunk_size=chunk_size,
        allgather_cp=False,
        debug_unified_grad_fused_logprob=False,
        vocab_size=VOCAB,
    )


def _batch(tokens=TOKENS, supports=SUPPORTS) -> dict:
    generator = torch.Generator().manual_seed(7)
    masks = [PartialSamplingMask.from_mask_list(rows) for rows in supports]
    sampler = [[torch.randn(len(row), generator=generator).log_softmax(0) for row in rows] for rows in supports]
    rollout_log_probs = [
        torch.tensor(
            [q[row.index(t)].item() if t in row else -1.1 for q, row, t in zip(qs, rows, ids[1:], strict=True)]
        )
        for qs, rows, ids in zip(sampler, supports, tokens, strict=True)
    ]
    return {
        "unconcat_tokens": [torch.tensor(ids) for ids in tokens],
        "target_tokens": [ids[1:] for ids in tokens],
        "total_lengths": [len(ids) for ids in tokens],
        "response_lengths": [len(ids) - 1 for ids in tokens],
        "rollout_sampling_mask_ids": [mask._as_tensors()[0] for mask in masks],
        "rollout_sampling_mask_offsets": [mask._as_tensors()[1] for mask in masks],
        "rollout_sampling_mask_log_probs": [torch.cat(qs) for qs in sampler],
        "rollout_log_probs": rollout_log_probs,
        "advantages": [torch.randn(len(ids) - 1, generator=generator) for ids in tokens],
        "loss_weights": [torch.randn(len(ids) - 1, generator=generator) for ids in tokens],
        "loss_masks": [torch.tensor([1.0] * (len(ids) - 2) + [0.0]) for ids in tokens],
        "sample_indices": list(range(len(tokens))),
        "loss_fn_config": {"importance_sampling": "tis"},
    }


def _inputs(generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    hidden = torch.randn(1, sum(map(len, TOKENS)), HIDDEN, generator=generator)
    return hidden, torch.randn(VOCAB, HIDDEN, generator=generator)


def _dense_target_logprobs(logits: torch.Tensor) -> list[torch.Tensor]:
    """log p(target) under softmax over the support, or over the vocabulary where the support is empty."""
    outputs, row = [], 0
    for tokens, supports in zip(TOKENS, SUPPORTS, strict=True):
        values = []
        for target, support in zip(tokens[1:], supports, strict=True):
            scores = logits[0, row] / TEMPERATURE
            values.append(scores[target] - torch.logsumexp(scores[support] if support else scores, dim=-1))
            row += 1
        row += 1  # the datum's last token predicts nothing
        outputs.append(torch.stack(values))
    return outputs


@pytest.mark.parametrize("chunk_size", [1, 3, 64])
@pytest.mark.parametrize("vocab_size", [VOCAB, VOCAB - 1])
def test_output_selected_log_probs_match_the_logits_primitive(chunk_size, vocab_size):
    generator = torch.Generator().manual_seed(3)
    hidden = torch.randn(7, HIDDEN, generator=generator, dtype=torch.float64)
    weight = torch.randn(VOCAB, HIDDEN, generator=generator, dtype=torch.float64)
    ids = torch.randint(-1, vocab_size, (7, 4), generator=generator)
    coefficients = torch.randn(7, 4, generator=generator, dtype=torch.float64)

    def score(fused: bool):
        rows, columns = hidden.clone().requires_grad_(), weight.clone().requires_grad_()
        kwargs = dict(vocab_size=vocab_size, temperature=TEMPERATURE, with_entropy=True)
        if fused:
            selected, entropy = output_selected_log_probs_and_entropy(
                rows, columns, ids, chunk_size=chunk_size, **kwargs
            )
        else:
            selected, entropy = selected_log_probs_and_entropy(F.linear(rows, columns), ids, **kwargs)
        ((selected * coefficients).sum() - 0.3 * entropy.sum()).backward()
        return selected, entropy, rows.grad, columns.grad

    for actual, expected in zip(score(fused=True), score(fused=False), strict=True):
        torch.testing.assert_close(actual, expected)


@pytest.mark.usefixtures("single_rank")
@pytest.mark.parametrize("chunk_size", [2, 5])
@pytest.mark.parametrize("loss_fn", sorted(tinker_losses.TINKER_LOSS_FUNCTIONS))
def test_every_tinker_loss_matches_the_logits_path(loss_fn, chunk_size):
    hidden, weight = _inputs(torch.Generator().manual_seed(11))

    def run(fused: bool):
        rows, columns = hidden.clone().requires_grad_(), weight.clone().requires_grad_()
        batch = _batch() | {"loss_fn": loss_fn}
        if fused:
            loss, outputs = tinker_losses.TINKER_LOSS_FUNCTIONS[loss_fn](
                _args(chunk_size), batch | {"output_weight": columns}, rows, None
            )
        else:
            logits = F.linear(rows, columns)
            loss, outputs = tinker_losses.TINKER_LOSS_FUNCTIONS[loss_fn](_args(chunk_size), batch, logits, None)
        loss.backward()
        return loss, [output["logprobs"] for output in outputs["per_datum"]], rows.grad, columns.grad

    (loss, logprobs, hidden_grad, weight_grad), expected = run(fused=True), run(fused=False)
    torch.testing.assert_close(loss, expected[0])
    torch.testing.assert_close(logprobs, expected[1])
    torch.testing.assert_close(hidden_grad, expected[2])
    torch.testing.assert_close(weight_grad, expected[3])
    torch.testing.assert_close(logprobs, [lp.detach() for lp in _dense_target_logprobs(F.linear(hidden, weight))])


@pytest.mark.usefixtures("single_rank")
def test_a_target_outside_its_support_scores_minus_infinity_in_both_paths():
    hidden, weight = torch.randn(1, 3, HIDDEN), torch.randn(VOCAB, HIDDEN)
    batch = _batch(tokens=[[3, 4, 5]], supports=[[[1, 2], []]])
    fused = tinker_losses._target_logprobs(_args(2), batch | {"output_weight": weight}, hidden)
    unfused = tinker_losses._target_logprobs(_args(2), batch, F.linear(hidden, weight))
    assert fused[0][0].item() == -torch.inf
    torch.testing.assert_close(fused, unfused)


def _check_tensor_parallel(tp: GroupInfo) -> None:
    hidden, weight = _inputs(torch.Generator().manual_seed(23))
    shard = VOCAB // tp.size
    local = weight[tp.rank * shard : (tp.rank + 1) * shard].clone().requires_grad_()
    full = weight.clone().requires_grad_()
    hidden_local, hidden_full = hidden.clone().requires_grad_(), hidden.clone().requires_grad_()
    coefficients = torch.randn(len(TOKENS), 4, generator=torch.Generator().manual_seed(29))

    # the target log-probs, renormalized within the supports, and their gradients
    actual = tinker_losses._target_logprobs(_args(3), _batch() | {"output_weight": local}, hidden_local)
    expected = _dense_target_logprobs(F.linear(hidden_full, full))
    for actual_datum, expected_datum in zip(actual, expected, strict=True):
        torch.testing.assert_close(actual_datum, expected_datum)
    sum(lp @ c[: len(lp)] for lp, c in zip(actual, coefficients, strict=True)).backward()
    sum(lp @ c[: len(lp)] for lp, c in zip(expected, coefficients, strict=True)).backward()
    # each rank holds its vocabulary shard's hidden-state gradient; the output layer's region op sums them
    dist.all_reduce(hidden_local.grad, group=tp.group)
    torch.testing.assert_close(hidden_local.grad, hidden_full.grad)
    torch.testing.assert_close(local.grad, full.grad[tp.rank * shard : (tp.rank + 1) * shard])

    # score centering through hidden states against the vocabulary-sharded logits path
    gradients = []
    for fused in (True, False):
        rows, columns = hidden.clone().requires_grad_(), local.detach().clone().requires_grad_()
        batch = _batch() | {"loss_fn": "score_centering"}
        if fused:
            loss, outputs = tinker_losses.score_centering_loss_function(
                _args(2), batch | {"output_weight": columns}, rows, None
            )
        else:
            loss, outputs = tinker_losses.score_centering_loss_function(_args(2), batch, F.linear(rows, columns), None)
        loss.backward()
        dist.all_reduce(rows.grad, group=tp.group)
        gradients.append((loss, [output["logprobs"] for output in outputs["per_datum"]], rows.grad, columns.grad))
    for actual_value, expected_value in zip(*gradients, strict=True):
        torch.testing.assert_close(actual_value, expected_value)


def _worker(rank: int, world_size: int, rendezvous: str) -> None:
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", init_method=rendezvous, rank=rank, world_size=world_size, timeout=timedelta(seconds=120)
    )
    try:
        tp = GroupInfo(rank=rank, size=world_size, group=dist.group.WORLD)
        singleton = GroupInfo(rank=0, size=1, group=None)
        state = {name: singleton for name in ("intra_dp", "intra_dp_cp", "cp", "pp", "ep", "etp", "indep_dp")}
        set_parallel_state(ParallelState(**state, tp=tp))
        _check_tensor_parallel(tp)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("tp_size", [2, 4])
def test_vocabulary_parallel_shards_match_the_full_vocabulary(tmp_path: Path, tp_size: int) -> None:
    mp.spawn(_worker, args=(tp_size, (tmp_path / "rendezvous").as_uri()), nprocs=tp_size, join=True)
