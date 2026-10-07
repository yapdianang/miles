"""Sum Tinker losses over datums without server-side normalization.

Client loss inputs carry normalization; the trainer accumulates raw sums.
The SDK represents custom-loss gradients as `weights = -dL/dlogprob` with `cross_entropy`.
"""

from argparse import Namespace
from collections.abc import Callable

import torch
import torch.distributed as dist

from miles.backends.training_utils.data.context_parallel import all_gather_with_cp, slice_log_prob_with_cp
from miles.backends.training_utils.loss.hub.logit_processors import _iter_response_chunks, get_log_probs_and_entropy
from miles.backends.training_utils.loss.hub.score_centering import (
    ScoreCenteringInputs,
    output_selected_log_probs_and_entropy,
    score_centering_loss,
    selected_log_probs,
)
from miles.backends.training_utils.parallel import get_parallel_state
from miles.utils.sampling_mask import PartialSamplingMask
from miles.utils.types import RolloutBatch

PPO_DEFAULTS = {"clip_low_threshold": 0.8, "clip_high_threshold": 1.2}
CISPO_DEFAULTS = {"clip_low_threshold": 0.0, "clip_high_threshold": 4.0}
DRO_DEFAULTS = {"beta": 0.05}
SCORE_CENTERING_DEFAULTS = {"importance_sampling": "none", "tis_clip": 2.0, "mis_low": 0.5, "mis_high": 5.0}


def _label_tokens(batch: RolloutBatch) -> list[torch.Tensor]:
    # Tinker targets are explicit labels: splice them over the response region of the gather sequence
    return [
        torch.cat([tokens[: len(tokens) - len(targets)], _as_tensor_like(targets, tokens)])
        for tokens, targets in zip(batch["unconcat_tokens"], batch["target_tokens"], strict=True)
    ]


def _target_logprobs(args: Namespace, batch: RolloutBatch, logits: torch.Tensor) -> list[torch.Tensor]:
    if "output_weight" in batch:
        return _fused_target_logprobs(args, batch, logits)
    outputs = get_log_probs_and_entropy(
        logits,
        args=args,
        unconcat_tokens=_label_tokens(batch),
        total_lengths=batch["total_lengths"],
        response_lengths=batch["response_lengths"],
        with_entropy=False,
        max_seq_lens=batch.get("max_seq_lens", None),
        rollout_sampling_mask=_sampling_supports(batch),
    )
    return outputs["log_probs"]


def _sampling_supports(batch: RolloutBatch) -> list[PartialSamplingMask] | None:
    """The engine's top-k/top-p supports: a position with one renormalizes within it, as the sampler did."""
    if batch.get("rollout_sampling_mask_ids") is None:
        return None
    return [
        PartialSamplingMask(ids=torch.as_tensor(ids), offsets=torch.as_tensor(offsets))
        for ids, offsets in zip(
            batch["rollout_sampling_mask_ids"], batch["rollout_sampling_mask_offsets"], strict=True
        )
    ]


def _support_head(batch: RolloutBatch, index: int, rows: range | list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    """Recorded supports of these target rows as ``[rows, width]`` ids (-1 padding) and sampler log-probs."""
    head_log_probs = _padded_supports(batch, "rollout_sampling_mask_log_probs", index, rows, -torch.inf)
    return _support_ids(batch, index, rows), head_log_probs.float()


def _support_ids(batch: RolloutBatch, index: int, rows: range | list[int]) -> torch.Tensor:
    return _padded_supports(batch, "rollout_sampling_mask_ids", index, rows, -1).long()


def _padded_supports(batch: RolloutBatch, key: str, index: int, rows: range | list[int], fill: float) -> torch.Tensor:
    """One CSR field of these target rows' recorded supports as ``[rows, width]``, padded with ``fill``.

    ``rows`` is a range, or under context parallelism this rank's two zigzag ranges as one list.
    """
    if batch.get("rollout_sampling_mask_ids") is None:
        return torch.full((len(rows), 0), fill)
    rows = torch.arange(rows.start, rows.stop) if isinstance(rows, range) else torch.tensor(rows, dtype=torch.long)
    offsets = torch.as_tensor(batch["rollout_sampling_mask_offsets"][index])
    starts = offsets[rows]
    lengths = offsets[rows + 1] - starts
    row = torch.repeat_interleave(torch.arange(len(rows)), lengths)
    # entry k of the flattened heads sits at starts[row] + k - (entries of earlier rows)
    flat = torch.repeat_interleave(starts - (lengths.cumsum(0) - lengths), lengths) + torch.arange(int(lengths.sum()))
    column = flat - starts[row]
    width = int(lengths.max()) if len(rows) else 0
    values = torch.as_tensor(batch[key][index])[flat]
    padded = torch.full((len(rows), width), fill, dtype=values.dtype)
    padded[row, column] = values
    return padded


def _fused_target_logprobs(args: Namespace, batch: RolloutBatch, hidden: torch.Tensor) -> list[torch.Tensor]:
    """``_target_logprobs`` from hidden states: each recorded support renormalizes as the masked logits do."""
    log_probs = []
    for index, (chunk, labels, rows) in enumerate(_response_chunks(args, batch, hidden)):
        head_ids = _support_ids(batch, index, rows).to(labels.device)
        labels = labels.unsqueeze(-1).long()
        # padded vocabulary entries stay in the normalizer, as in Megatron's vocab-parallel cross entropy
        selected = _selected_log_probs(args, batch, chunk, torch.cat([labels, head_ids], dim=-1), vocab_size=None)
        head = head_ids >= 0
        has_head = head.any(-1)
        # a masked target logit is -inf, so a target outside its recorded support has log-prob -inf
        target = selected[:, 0].masked_fill(has_head & ~(head_ids == labels).any(-1), -torch.inf)
        log_probs.append(target - torch.where(has_head, _logsumexp_where(selected[:, 1:], head).squeeze(-1), 0.0))
    return log_probs


def _response_chunks(args: Namespace, batch: RolloutBatch, logits: torch.Tensor):
    return _iter_response_chunks(
        logits,
        args=args,
        unconcat_tokens=_label_tokens(batch),
        total_lengths=batch["total_lengths"],
        response_lengths=batch["response_lengths"],
        max_seq_lens=batch.get("max_seq_lens", None),
        include_response_indices=True,
    )


def _selected_log_probs(
    args: Namespace, batch: RolloutBatch, chunk: torch.Tensor, token_ids: torch.Tensor, vocab_size: int | None
) -> torch.Tensor:
    """Log-probs ``[rows, K]`` at global ids from response logits, or from hidden states under --tinker-fused-loss."""
    parallel = get_parallel_state()
    kwargs = dict(
        group=parallel.tp.group if parallel.tp.size > 1 else None,
        vocab_size=vocab_size,
        temperature=1.0 if args.true_on_policy_mode else args.rollout_temperature,
        chunk_size=args.log_probs_chunk_size,
    )
    if "output_weight" in batch:
        return output_selected_log_probs_and_entropy(chunk, batch["output_weight"], token_ids, **kwargs)[0]
    return selected_log_probs(chunk, token_ids, **kwargs)


def _logsumexp_where(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return torch.logsumexp(values.masked_fill(~mask, -torch.inf), dim=-1, keepdim=True)


def _as_tensor_like(values, reference: torch.Tensor) -> torch.Tensor:
    return torch.as_tensor(values, dtype=reference.dtype, device=reference.device)


def _local(args: Namespace, batch: RolloutBatch, key: str) -> list:
    """Full-length per-datum response vectors in this CP rank's log-prob layout."""
    if get_parallel_state().cp.size == 1:
        return batch[key]
    max_seq_lens = batch.get("max_seq_lens") or [None] * len(batch[key])
    return [
        slice_log_prob_with_cp(torch.as_tensor(values), total_length, response_length, args.qkv_format, max_seq_len)
        for values, total_length, response_length, max_seq_len in zip(
            batch[key], batch["total_lengths"], batch["response_lengths"], max_seq_lens, strict=True
        )
    ]


def _response_masks(args: Namespace, batch: RolloutBatch, log_probs: list[torch.Tensor]) -> list[torch.Tensor]:
    """Per-datum loss masks; a DP-padding datum is all zeros and must not reach the objective."""
    return [
        _as_tensor_like(mask, log_prob)
        for mask, log_prob in zip(_local(args, batch, "loss_masks"), log_probs, strict=True)
    ]


def _sum_loss_and_outputs(
    args: Namespace,
    batch: RolloutBatch,
    logits: torch.Tensor,
    log_probs: list[torch.Tensor],
    per_datum_losses: list[torch.Tensor],
) -> tuple[torch.Tensor, dict]:
    if per_datum_losses:
        loss = torch.stack(per_datum_losses).sum()
    else:
        # a microbatch with no supervised tokens still needs the graph alive; fp32 sum avoids fp16 inf -> nan
        loss = logits.sum(dtype=torch.float32) * 0
    datum_log_probs, datum_losses = log_probs, per_datum_losses
    parallel = get_parallel_state()
    if parallel.cp.size > 1 and per_datum_losses:
        # each CP rank holds part of every datum; the client gets whole-datum log-probs and losses
        with torch.no_grad():
            max_seq_lens = batch.get("max_seq_lens") or [None] * len(log_probs)
            datum_log_probs = [
                all_gather_with_cp(log_prob.detach(), total_length, response_length, args.qkv_format, max_seq_len)
                for log_prob, total_length, response_length, max_seq_len in zip(
                    log_probs, batch["total_lengths"], batch["response_lengths"], max_seq_lens, strict=True
                )
            ]
            summed = torch.stack(per_datum_losses).detach()
            dist.all_reduce(summed, group=parallel.cp.group)
            datum_losses = list(summed.unbind())
    per_datum = [
        {"sample_index": index, "logprobs": log_prob.detach().cpu(), "loss": sample_loss.detach().cpu()}
        for index, log_prob, sample_loss in zip(batch["sample_indices"], datum_log_probs, datum_losses, strict=True)
    ]
    return loss, {"loss": loss.detach(), "per_datum": per_datum}


def cross_entropy_loss_function(
    args: Namespace,
    batch: RolloutBatch,
    logits: torch.Tensor,
    sum_of_sample_mean: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, dict]:
    log_probs = _target_logprobs(args, batch, logits)
    per_datum_losses = [
        -(_as_tensor_like(weights, log_prob) * log_prob * mask).sum()
        for log_prob, weights, mask in zip(
            log_probs, _local(args, batch, "loss_weights"), _response_masks(args, batch, log_probs), strict=True
        )
    ]
    return _sum_loss_and_outputs(args, batch, logits, log_probs, per_datum_losses)


def importance_sampling_loss_function(
    args: Namespace,
    batch: RolloutBatch,
    logits: torch.Tensor,
    sum_of_sample_mean: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, dict]:
    log_probs = _target_logprobs(args, batch, logits)
    per_datum_losses = []
    for log_prob, sampling_log_prob, advantage, mask in zip(
        log_probs,
        batch["rollout_log_probs"],
        _local(args, batch, "advantages"),
        _response_masks(args, batch, log_probs),
        strict=True,
    ):
        ratio = torch.exp(log_prob - _as_tensor_like(sampling_log_prob, log_prob))
        per_datum_losses.append(-(ratio * _as_tensor_like(advantage, log_prob) * mask).sum())
    return _sum_loss_and_outputs(args, batch, logits, log_probs, per_datum_losses)


def ppo_loss_function(
    args: Namespace,
    batch: RolloutBatch,
    logits: torch.Tensor,
    sum_of_sample_mean: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, dict]:
    config = batch.get("loss_fn_config") or {}
    clip_low = config.get("clip_low_threshold", PPO_DEFAULTS["clip_low_threshold"])
    clip_high = config.get("clip_high_threshold", PPO_DEFAULTS["clip_high_threshold"])
    log_probs = _target_logprobs(args, batch, logits)
    per_datum_losses = []
    for log_prob, sampling_log_prob, advantage, mask in zip(
        log_probs,
        batch["rollout_log_probs"],
        _local(args, batch, "advantages"),
        _response_masks(args, batch, log_probs),
        strict=True,
    ):
        ratio = torch.exp(log_prob - _as_tensor_like(sampling_log_prob, log_prob))
        advantages = _as_tensor_like(advantage, log_prob)
        objective = torch.minimum(ratio * advantages, torch.clamp(ratio, clip_low, clip_high) * advantages)
        per_datum_losses.append(-(objective * mask).sum())
    return _sum_loss_and_outputs(args, batch, logits, log_probs, per_datum_losses)


def cispo_loss_function(
    args: Namespace,
    batch: RolloutBatch,
    logits: torch.Tensor,
    sum_of_sample_mean: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, dict]:
    config = batch.get("loss_fn_config") or {}
    clip_low = config.get("clip_low_threshold", CISPO_DEFAULTS["clip_low_threshold"])
    clip_high = config.get("clip_high_threshold", CISPO_DEFAULTS["clip_high_threshold"])
    log_probs = _target_logprobs(args, batch, logits)
    per_datum_losses = []
    for log_prob, sampling_log_prob, advantage, mask in zip(
        log_probs,
        batch["rollout_log_probs"],
        _local(args, batch, "advantages"),
        _response_masks(args, batch, log_probs),
        strict=True,
    ):
        ratio = torch.exp(log_prob - _as_tensor_like(sampling_log_prob, log_prob))
        coefficient = torch.clamp(ratio, clip_low, clip_high).detach()
        per_datum_losses.append(-(coefficient * log_prob * _as_tensor_like(advantage, log_prob) * mask).sum())
    return _sum_loss_and_outputs(args, batch, logits, log_probs, per_datum_losses)


def dro_loss_function(
    args: Namespace,
    batch: RolloutBatch,
    logits: torch.Tensor,
    sum_of_sample_mean: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, dict]:
    config = batch.get("loss_fn_config") or {}
    beta = config.get("beta", DRO_DEFAULTS["beta"])
    log_probs = _target_logprobs(args, batch, logits)
    per_datum_losses = []
    for log_prob, sampling_log_prob, advantage, mask in zip(
        log_probs,
        batch["rollout_log_probs"],
        _local(args, batch, "advantages"),
        _response_masks(args, batch, log_probs),
        strict=True,
    ):
        divergence = log_prob - _as_tensor_like(sampling_log_prob, log_prob)
        objective = log_prob * _as_tensor_like(advantage, log_prob) - 0.5 * beta * divergence**2
        per_datum_losses.append(-(objective * mask).sum())
    return _sum_loss_and_outputs(args, batch, logits, log_probs, per_datum_losses)


def score_centering_loss_function(
    args: Namespace,
    batch: RolloutBatch,
    logits: torch.Tensor,
    sum_of_sample_mean: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, dict]:
    """Score centering (arXiv:2609.20807) against the sampler's recorded support of each target.

    The trainer renormalizes over the support as the sampler did, so the head is the whole sampling
    distribution. A target without a recorded support has an empty head: no correction term.
    """
    config = SCORE_CENTERING_DEFAULTS | (batch.get("loss_fn_config") or {})
    masks, advantages = _local(args, batch, "loss_masks"), _local(args, batch, "advantages")
    log_probs, per_datum_losses = [], []
    for index, (chunk, labels, rows) in enumerate(_response_chunks(args, batch, logits)):
        head_ids, head_log_probs = _support_head(batch, index, rows)
        head_ids = head_ids.to(labels.device)
        selected = _selected_log_probs(
            args,
            batch,
            chunk,
            torch.cat([labels.unsqueeze(-1).long(), head_ids], dim=-1),
            vocab_size=getattr(args, "vocab_size", None),
        )
        head = head_ids >= 0
        has_head = head.any(-1, keepdim=True)
        # A recorded support leaves no tail: both distributions are renormalized on it in float64 so that
        # the estimator's tail-mass ratio stays exactly one instead of amplifying float32 rounding.
        selected = selected.double()
        selected = selected - torch.where(has_head, _logsumexp_where(selected[:, 1:], head), 0.0)
        rollout_head = head_log_probs.to(selected)
        rollout_head = rollout_head - torch.where(has_head, _logsumexp_where(rollout_head, head), 0.0)
        mask = _as_tensor_like(masks[index], selected)
        token_losses, _ = score_centering_loss(
            ScoreCenteringInputs(
                train_log_probs=selected[:, 0],
                train_head_log_probs=selected[:, 1:],
                rollout_log_probs=_as_tensor_like(batch["rollout_log_probs"][index], selected),
                rollout_head_log_probs=rollout_head,
                head_mask=head & mask.bool().unsqueeze(-1),
                advantages=_as_tensor_like(advantages[index], selected) * mask,
                mode=config["importance_sampling"],
                tis_clip=config["tis_clip"],
                mis_low=config["mis_low"],
                mis_high=config["mis_high"],
            )
        )
        log_probs.append(selected[:, 0].float())
        per_datum_losses.append((token_losses * mask).sum().float())
    return _sum_loss_and_outputs(args, batch, logits, log_probs, per_datum_losses)


TINKER_LOSS_FUNCTIONS = {
    "cross_entropy": cross_entropy_loss_function,
    "importance_sampling": importance_sampling_loss_function,
    "ppo": ppo_loss_function,
    "cispo": cispo_loss_function,
    "dro": dro_loss_function,
    "score_centering": score_centering_loss_function,
}
