"""Tinker target log-probs renormalize within the engine's sampling support where a datum position has one."""

import sys
from argparse import Namespace
from types import SimpleNamespace

import pytest
import torch

from miles.backends.training_utils.data.sampling_mask import build_local_sampling_mask
from miles.backends.training_utils.loss.hub import logit_processors, tinker_losses
from miles.utils.sampling_mask import PartialSamplingMask, RolloutSamplingMask

VOCAB = 12
# two datums: target i of a datum scores tokens[i + 1]; an empty support keeps the full vocabulary
TOKENS = [[3, 4, 5, 6, 7], [8, 9, 1]]
SUPPORTS = [[[], [5, 0, 11], [6], [7, 2]], [[], [1, 3, 4]]]


def _reference_cross_entropy(logits: torch.Tensor, tokens: torch.Tensor, _group: object) -> torch.Tensor:
    """Megatron's vocab-parallel cross entropy on one rank: -log_softmax(logits)[token]."""
    return torch.logsumexp(logits, dim=-1) - logits.gather(-1, tokens.unsqueeze(-1)).squeeze(-1)


@pytest.fixture(autouse=True)
def single_rank(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "megatron.core.fusions.fused_cross_entropy",
        SimpleNamespace(fused_vocab_parallel_cross_entropy=_reference_cross_entropy),
    )
    parallel = SimpleNamespace(tp=SimpleNamespace(rank=0, size=1, group=None), cp=SimpleNamespace(rank=0, size=1))
    monkeypatch.setattr(logit_processors, "get_parallel_state", lambda: parallel)


def _args(chunk_size: int) -> Namespace:
    return Namespace(
        qkv_format="thd",
        rollout_temperature=1.0,
        true_on_policy_mode=False,
        log_probs_chunk_size=chunk_size,
        allgather_cp=False,
        debug_unified_grad_fused_logprob=False,
    )


def _batch(with_supports: bool) -> dict:
    batch = {
        "unconcat_tokens": [torch.tensor(tokens) for tokens in TOKENS],
        "target_tokens": [tokens[1:] for tokens in TOKENS],
        "total_lengths": [len(tokens) for tokens in TOKENS],
        "response_lengths": [len(tokens) - 1 for tokens in TOKENS],
    }
    if with_supports:
        masks = [PartialSamplingMask.from_mask_list(supports) for supports in SUPPORTS]
        batch["rollout_sampling_mask_ids"] = [mask._as_tensors()[0] for mask in masks]
        batch["rollout_sampling_mask_offsets"] = [mask._as_tensors()[1] for mask in masks]
    return batch


def _reference_logprobs(logits: torch.Tensor) -> list[torch.Tensor]:
    """log p(target) under softmax over the support, or over the vocabulary where the support is empty."""
    outputs, row = [], 0
    for tokens, supports in zip(TOKENS, SUPPORTS, strict=True):
        values = []
        for target, support in zip(tokens[1:], supports, strict=True):
            scores = logits[0, row]
            allowed = scores[support] if support else scores
            values.append(scores[target] - torch.logsumexp(allowed, dim=-1))
            row += 1
        row += 1  # the datum's last token predicts nothing
        outputs.append(torch.stack(values))
    return outputs


@pytest.mark.parametrize("chunk_size", [-1, 2])
def test_target_logprobs_and_gradients_match_support_renormalization(chunk_size):
    logits = torch.randn(1, sum(map(len, TOKENS)), VOCAB, requires_grad=True)
    weights = [torch.randn(len(tokens) - 1) for tokens in TOKENS]

    actual = tinker_losses._target_logprobs(_args(chunk_size), _batch(with_supports=True), logits)
    (gradient,) = torch.autograd.grad(sum((w * lp).sum() for w, lp in zip(weights, actual, strict=True)), logits)
    expected = _reference_logprobs(logits)
    (expected_gradient,) = torch.autograd.grad(
        sum((w * lp).sum() for w, lp in zip(weights, expected, strict=True)), logits
    )

    for actual_logprobs, expected_logprobs in zip(actual, expected, strict=True):
        torch.testing.assert_close(actual_logprobs, expected_logprobs)
    torch.testing.assert_close(gradient, expected_gradient)
    # outside the support the renormalized log-prob has no gradient
    assert gradient[0, 1, [1, 2, 3, 4, 6, 7, 8, 9, 10]].abs().max() == 0


def test_a_batch_without_supports_keeps_full_vocabulary_logprobs():
    logits = torch.randn(1, sum(map(len, TOKENS)), VOCAB)
    actual = tinker_losses._target_logprobs(_args(-1), _batch(with_supports=False), logits)
    full = torch.log_softmax(logits[0], dim=-1)
    torch.testing.assert_close(actual[0], full[[0, 1, 2, 3], [4, 5, 6, 7]])
    torch.testing.assert_close(actual[1], full[[5, 6], [9, 1]])


def test_cross_entropy_weights_differentiate_the_renormalized_logprobs():
    """forward_backward_custom sends -dL/dlogprob as cross_entropy weights; the gradient must be of the support log-probs."""
    logits = torch.randn(1, sum(map(len, TOKENS)), VOCAB, requires_grad=True)
    batch = _batch(with_supports=True) | {
        "loss_weights": [[1.0, 2.0, -1.0, 0.5], [0.0, 3.0]],
        "loss_masks": [torch.ones(4), torch.ones(2)],
        "sample_indices": [0, 1],
    }
    loss, outputs = tinker_losses.cross_entropy_loss_function(_args(-1), batch, logits, None)
    (gradient,) = torch.autograd.grad(loss, logits)
    expected = _reference_logprobs(logits)
    expected_loss = -sum((torch.tensor(w) * lp).sum() for w, lp in zip(batch["loss_weights"], expected, strict=True))
    (expected_gradient,) = torch.autograd.grad(expected_loss, logits)
    torch.testing.assert_close(loss, expected_loss)
    torch.testing.assert_close(gradient, expected_gradient)
    torch.testing.assert_close(outputs["per_datum"][1]["logprobs"], expected[1].detach())


def test_an_empty_support_row_is_unrestricted_on_every_vocabulary_shard():
    mask = build_local_sampling_mask(
        torch.zeros(3, 4),
        sampling_mask=PartialSamplingMask.from_mask_list([[1, 5], [], [7]]),
        response_indices=range(3),
        tp_rank=1,
    )
    expected = torch.tensor([[False, True, False, False], [True, True, True, True], [False, False, False, True]])
    torch.testing.assert_close(mask, expected)


def test_only_the_partial_mask_accepts_empty_rows():
    assert len(PartialSamplingMask.from_mask_list([[], [2], []])) == 3
    with pytest.raises(ValueError, match="non-empty sampling mask"):
        RolloutSamplingMask.from_mask_list([[], [2]])
    with pytest.raises(ValueError, match="non-decreasing"):
        PartialSamplingMask(ids=[1, 2], offsets=[0, 2, 1, 2])
