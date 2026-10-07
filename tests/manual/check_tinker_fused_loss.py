"""Compare --tinker-fused-loss with the logits path at the MiMo-V2.6-Flash output layer, one GPU per TP rank.

torchrun --nproc-per-node 4 -m tests.manual.check_tinker_fused_loss [--lengths 131072 262144]

Each rank holds a sequence-parallel shard of the final hidden states and a Megatron output layer with a frozen
[vocab/TP, hidden] bf16 weight, as the attention-LoRA trainer does. Per length, the logits path runs the output
layer and the current Tinker loss; the fused path runs the trainer's output processor and the same loss from the
hidden states. Rank 0 prints each path's peak memory above its inputs, its forward+backward time, and the
fused-minus-logits differences; the exit code is 1 when a difference exceeds its bound.
"""

import argparse
import os
import sys
import time
from argparse import Namespace

import torch
import torch.distributed as dist
from megatron.core import parallel_state
from megatron.core.tensor_parallel import ColumnParallelLinear
from megatron.core.transformer.transformer_config import TransformerConfig

from miles.backends.megatron_utils.model import _output_layer_input
from miles.backends.training_utils.loss.hub.tinker_losses import TINKER_LOSS_FUNCTIONS
from miles.backends.training_utils.parallel import GroupInfo, ParallelState, set_parallel_state
from miles.utils.sampling_mask import PartialSamplingMask

# scripts/models/mimo-v2.6-flash.py
HIDDEN, VOCAB = 4096, 152576
GiB = 2**30


def _batch(length: int, support: int, loss_fn: str, generator: torch.Generator) -> dict:
    device = generator.device
    tokens = torch.randint(0, VOCAB, (length,), generator=generator, device=device)
    targets = tokens[1:].cpu()
    # supports cover the second half of the targets, as engine samples follow a prompt; ids are unique per row
    sampled = torch.arange(length - 1) >= (length - 1) // 2
    others = (targets[sampled, None] + 7919 * torch.arange(1, support)) % VOCAB
    ids = torch.cat([targets[sampled, None], others], dim=-1)
    offsets = torch.cat([torch.zeros(1, dtype=torch.long), (sampled.long() * support).cumsum(0)])
    sampler = torch.randn(ids.shape, generator=generator, device=device).log_softmax(-1)
    rollout_log_probs = torch.full((length - 1,), -1.0, device=device)
    rollout_log_probs[sampled.to(device)] = sampler[:, 0]
    mask = PartialSamplingMask(ids=ids.flatten(), offsets=offsets)
    return {
        "loss_fn": loss_fn,
        "unconcat_tokens": [tokens],
        "target_tokens": [targets.tolist()],
        "total_lengths": [length],
        "response_lengths": [length - 1],
        "rollout_sampling_mask_ids": [mask._as_tensors()[0]],
        "rollout_sampling_mask_offsets": [mask._as_tensors()[1]],
        "rollout_sampling_mask_log_probs": [sampler.flatten().cpu()],
        "rollout_log_probs": [rollout_log_probs],
        "advantages": [torch.randn(length - 1, generator=generator, device=device)],
        "loss_weights": [torch.randn(length - 1, generator=generator, device=device)],
        "loss_masks": [torch.ones(length - 1, device=device)],
        "sample_indices": [0],
    }


def _args(chunk_size: int) -> Namespace:
    return Namespace(
        qkv_format="thd",
        rollout_temperature=1.0,
        true_on_policy_mode=False,
        log_probs_chunk_size=chunk_size,
        allgather_cp=False,
        debug_unified_grad_fused_logprob=False,
        vocab_size=VOCAB,
    )


def _run(fused: bool, hidden: torch.Tensor, layer: ColumnParallelLinear, batch: dict, chunk_size: int) -> dict:
    loss_function = TINKER_LOSS_FUNCTIONS[batch["loss_fn"]]
    rows = hidden.detach().requires_grad_()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    start = time.perf_counter()
    if fused:
        batch = dict(batch)
        inputs = _output_layer_input(
            hidden_states=rows, output_layer=layer, output_weight=None, context=batch, config=layer.config
        )
        loss, outputs = loss_function(_args(chunk_size), batch, inputs, None)
    else:
        # the launcher's current logits path: the output layer, then the unchunked loss
        logits, _ = layer(rows)
        loss, outputs = loss_function(_args(-1), batch, logits.transpose(0, 1), None)
    loss.backward()
    torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    peak = torch.cuda.max_memory_allocated() - base
    return {
        "loss": loss.detach().double(),
        "logprobs": outputs["per_datum"][0]["logprobs"].double(),
        "grad": rows.grad,
        "peak_gib": peak / GiB,
        "ms": seconds * 1e3,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lengths", type=int, nargs="+", default=[131072, 262144])
    parser.add_argument("--loss-fn", default="importance_sampling", choices=sorted(TINKER_LOSS_FUNCTIONS))
    parser.add_argument("--chunk-size", type=int, default=4096)
    parser.add_argument("--support", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=2, help="runs per path; the last one is reported")
    options = parser.parse_args()

    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    rank, world_size = dist.get_rank(), dist.get_world_size()
    parallel_state.initialize_model_parallel(tensor_model_parallel_size=world_size)
    config = TransformerConfig(
        num_layers=1,
        hidden_size=HIDDEN,
        num_attention_heads=64,
        tensor_model_parallel_size=world_size,
        sequence_parallel=world_size > 1,
        bf16=True,
        params_dtype=torch.bfloat16,
        perform_initialization=False,
    )
    layer = ColumnParallelLinear(HIDDEN, VOCAB, config=config, init_method=config.init_method, bias=False)
    weight_generator = torch.Generator(device="cuda").manual_seed(1000 + rank)
    layer.weight.requires_grad_(False).copy_(
        torch.randn(layer.weight.shape, generator=weight_generator, device="cuda") * 0.04
    )
    singleton = GroupInfo(rank=0, size=1, group=None)
    state = {name: singleton for name in ("intra_dp", "intra_dp_cp", "cp", "pp", "ep", "etp", "indep_dp")}
    set_parallel_state(ParallelState(**state, tp=GroupInfo(rank=rank, size=world_size, group=layer.tp_group)))

    failed = False
    for length in options.lengths:
        # this rank's sequence-parallel shard of the final hidden states, [length / TP, 1, hidden]
        hidden_generator = torch.Generator(device="cuda").manual_seed(length * world_size + rank)
        hidden = torch.randn(length // world_size, 1, HIDDEN, generator=hidden_generator, device="cuda").bfloat16()
        batch = _batch(length, options.support, options.loss_fn, torch.Generator(device="cuda").manual_seed(length))
        results = {}
        for fused in (False, True):
            for _ in range(options.repeats):
                results[fused] = None
                torch.cuda.empty_cache()
                results[fused] = _run(fused, hidden, layer, batch, options.chunk_size)
        logits, fused = results[False], results[True]
        logprob_error = (fused["logprobs"] - logits["logprobs"]).abs()
        finite = logits["logprobs"].isfinite()
        # each rank holds the hidden-state gradient of its sequence shard
        squares = torch.stack(
            [(fused["grad"].float() - logits["grad"].float()).square().sum(), logits["grad"].float().square().sum()]
        )
        dist.all_reduce(squares)
        grad_error = (squares[0] / squares[1]).sqrt()
        checks = {
            "loss relative error": ((fused["loss"] - logits["loss"]).abs() / logits["loss"].abs()).item(),
            "logprob max abs error": logprob_error[finite].max().item(),
            "hidden-grad relative L2 error": grad_error.item(),
        }
        bounds = {"loss relative error": 1e-4, "logprob max abs error": 1e-3, "hidden-grad relative L2 error": 1e-2}
        if rank == 0:
            print(f"length {length}, TP{world_size}, {options.loss_fn}, fused chunk {options.chunk_size}")
            for name, result in (("logits", logits), ("fused", fused)):
                print(f"  {name:6s} peak {result['peak_gib']:7.2f} GiB  fwd+bwd {result['ms']:8.1f} ms")
            print(f"  logprob mean abs error {logprob_error[finite].mean().item():.3e}")
            for name, value in checks.items():
                ok = value <= bounds[name]
                failed |= not ok
                print(f"  {name} {value:.3e} (bound {bounds[name]:.0e}) {'ok' if ok else 'FAIL'}")
            sys.stdout.flush()
        del results, logits, fused
    dist.destroy_process_group()
    return int(failed)


if __name__ == "__main__":
    sys.exit(main())
