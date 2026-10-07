"""Check MiMo-V2 context parallelism on GPUs: CP=k against CP=1, halo traffic and peak memory.

``run`` (under torchrun, one GPU per TP x CP rank) builds MiMo-V2 from a BF16 conversion with
Megatron-Bridge, packs one THD micro-batch as Miles' get_batch does, and runs forward and backward on
the summed target log-probs, scored from hidden states as under --tinker-fused-loss. It writes the
whole-sequence log-probs, the gradients of the attention and router weights, the CP=1 routes, the
per-layer traffic of SWA halos and the global-layer ring, timing and peak memory. ``--replay-routes``
replays a CP=1 run's routes (R3), so ``compare`` isolates attention.

    P4=<4-layer BF16 partial (mimo26-p4-bf16)>; FULL=/data/model-cache/mimo-v26/MiMo-V2.6-Flash-RL-bf16
    OUT=/tmp/cp-check; SEQS=65536,30000,17
    torchrun --nproc-per-node 2 tools/validate_mimo_v2_cp.py run --hf-path $P4 --tp 2 --cp 1 --ep 1 \
        --seq-lens $SEQS --out $OUT/cp1
    torchrun --nproc-per-node 4 tools/validate_mimo_v2_cp.py run --hf-path $P4 --tp 2 --cp 2 --ep 1 \
        --seq-lens $SEQS --replay-routes $OUT/cp1 --out $OUT/cp2
    torchrun --nproc-per-node 8 tools/validate_mimo_v2_cp.py run --hf-path $P4 --tp 2 --cp 4 --ep 1 \
        --seq-lens $SEQS --replay-routes $OUT/cp1 --out $OUT/cp4
    python tools/validate_mimo_v2_cp.py compare $OUT/cp1 $OUT/cp2 $OUT/cp4

    # memory: the full model with full recompute and nothing saved; CP 8 runs at TP 1
    torchrun --nproc-per-node 8 tools/validate_mimo_v2_cp.py run --hf-path $FULL --tp 2 --cp 4 \
        --seq-lens 1048576 --recompute --no-save --out $OUT/mem-1m-cp4
"""

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
from megatron.bridge import AutoBridge
from megatron.core import parallel_state as mpu
from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.moe.router_replay import RouterReplay, RouterReplayAction

import miles_plugins.megatron_bridge  # noqa: F401  registers the MiMo-V2 bridge
from miles.backends.megatron_utils.model import _output_layer_input
from miles.backends.megatron_utils.parallel import PackedSeqParamsWithHostCuSeqlens
from miles.backends.training_utils.loss.hub.score_centering import output_selected_log_probs_and_entropy
from miles.backends.training_utils.data.context_parallel import slice_with_cp
from miles_plugins.megatron_bridge.swa_context_parallel import plan_swa_halo

# Miles' default --data-pad-size-multiplier: get_batch pads a micro-batch to tp * 128 tokens
_PAD_MULTIPLIER = 128
_GRAD_PARAMS = ("linear_qkv.weight", "linear_proj.weight", "core_attention.softmax_offset", "router.weight")


def _build_model(args):
    bridge = AutoBridge.from_hf_pretrained(args.hf_path, trust_remote_code=True)
    provider = bridge.to_megatron_provider(load_weights=True)
    world = dist.get_world_size()
    assert world == args.tp * args.cp, "one datum per run: the world must be exactly tp * cp (no DP)"
    provider.tensor_model_parallel_size = args.tp
    provider.context_parallel_size = args.cp
    provider.pipeline_model_parallel_size = 1
    provider.expert_model_parallel_size = args.ep or world
    provider.expert_tensor_parallel_size = 1
    provider.sequence_parallel = args.tp > 1
    provider.attention_backend = AttnBackend.fused
    provider.gradient_accumulation_fusion = False
    provider.variable_seq_lengths = True
    provider.moe_enable_routing_replay = True
    if args.recompute:
        provider.recompute_granularity, provider.recompute_method, provider.recompute_num_layers = "full", "uniform", 1
    provider.finalize()
    provider.initialize_model_parallel(seed=1234)
    [model] = provider.provide_distributed_model(wrap_with_ddp=False, bf16=True)
    model.train()
    return provider, model


def _sequences(args, vocab_size):
    if args.tokens_file:
        return [torch.tensor(seq, dtype=torch.long) for seq in json.loads(Path(args.tokens_file).read_text())]
    generator = torch.Generator().manual_seed(args.seed)
    return [torch.randint(vocab_size, (int(n),), generator=generator) for n in args.seq_lens.split(",")]


def _local_positions(cu_seqlens, cp_rank, cp_size):
    """Global packed rows this rank holds: chunks r and 2 * cp - 1 - r of every sequence."""
    if cp_size == 1:
        return torch.arange(cu_seqlens[-1])
    rows = []
    for start, end in zip(cu_seqlens[:-1], cu_seqlens[1:], strict=True):
        chunk = (end - start) // (2 * cp_size)
        rows += [
            torch.arange(start + c * chunk, start + (c + 1) * chunk) for c in (cp_rank, 2 * cp_size - 1 - cp_rank)
        ]
    return torch.cat(rows)


def _pack(sequences, fill_rows, cp_rank, cp_size, tp_size):
    """get_batch's THD layout: per-sequence zigzag slices, then padding to tp * 128 rows."""
    state = argparse.Namespace(cp=argparse.Namespace(rank=cp_rank, size=cp_size))
    local = [slice_with_cp(seq, fill_rows, "thd", parallel_state=state) for seq in sequences]
    lengths = [t.shape[0] for t in local]
    stream = torch.cat(local)
    pad = -stream.shape[0] % (tp_size * _PAD_MULTIPLIER)
    if pad:
        stream = fill_rows(stream, pad)
        lengths.append(pad)
    return stream, tuple(int(b) * cp_size for b in torch.tensor([0, *lengths]).cumsum(0))


def _batch(sequences, cp_rank, cp_size, tp_size):
    labels = [torch.cat([seq[1:], seq.new_zeros(1)]) for seq in sequences]
    weights = [torch.cat([torch.ones(len(seq) - 1), torch.zeros(1)]) for seq in sequences]

    def zeros(rows, pad):
        return torch.cat([rows, rows.new_zeros(pad, *rows.shape[1:])])

    tokens, cu_seqlens = _pack(sequences, zeros, cp_rank, cp_size, tp_size)
    cu = torch.tensor(cu_seqlens, dtype=torch.int32, device="cuda")
    packed = PackedSeqParamsWithHostCuSeqlens(
        qkv_format="thd",
        cu_seqlens_q=cu,
        cu_seqlens_kv=cu,
        max_seqlen_q=int((cu[1:] - cu[:-1]).max()),
        max_seqlen_kv=int((cu[1:] - cu[:-1]).max()),
        cu_seqlens_host=cu_seqlens,
    )
    return {
        "tokens": tokens.cuda().unsqueeze(0),
        "labels": _pack(labels, zeros, cp_rank, cp_size, tp_size)[0].cuda(),
        "weights": _pack(weights, zeros, cp_rank, cp_size, tp_size)[0].cuda(),
        "packed": packed,
    }


def _routes_to_replay(routes, sequences, cp_rank, cp_size, tp_rank, tp_size):
    """A CP=1 run's per-sequence routes in this rank's layout; padding rows route to experts 0..topk-1."""

    def arange_rows(rows, pad):
        filler = torch.arange(rows.shape[-1], dtype=rows.dtype).expand(pad, *rows.shape[1:])
        return torch.cat([rows, filler])

    assert [r.shape[0] for r in routes] == [len(seq) for seq in sequences], "routes are for another datum"
    stream, _ = _pack(routes, arange_rows, cp_rank, cp_size, tp_size)
    # sequence parallelism hands each TP rank a contiguous slice of the local rows
    stream = stream.chunk(tp_size)[tp_rank].cuda().long()
    return [stream[:, layer].contiguous() for layer in range(stream.shape[1])]


def _target_log_probs(model, batch, chunk_size):
    """Label log-probs by --tinker-fused-loss's path: hidden states times the output weight, per row chunk."""
    context = {}
    hidden = model(
        input_ids=batch["tokens"],
        position_ids=None,
        attention_mask=None,
        packed_seq_params=batch["packed"],
        output_processor=_output_layer_input,
        output_processor_context=context,
        fp32_output=False,
    )
    tp_group = mpu.get_tensor_model_parallel_group() if mpu.get_tensor_model_parallel_world_size() > 1 else None
    selected, _ = output_selected_log_probs_and_entropy(
        hidden[0], context["output_weight"], batch["labels"].unsqueeze(-1), group=tp_group, chunk_size=chunk_size
    )
    return selected[:, 0]


def _gather_local(values, group):
    gathered = [torch.empty_like(values) for _ in range(dist.get_world_size(group))]
    dist.all_gather(gathered, values.contiguous(), group=group)
    return gathered


def _whole_sequences(local, cu_seqlens, sequences, cp_size):
    """Each rank's local rows of a per-row tensor -> one tensor per sequence in token order."""
    stream = torch.empty(cu_seqlens[-1], *local.shape[1:], dtype=local.dtype)
    for rank, rows in enumerate(_gather_local(local, mpu.get_context_parallel_group())):
        stream[_local_positions(cu_seqlens, rank, cp_size)] = rows.cpu()
    return [stream[start : start + len(seq)] for start, seq in zip(cu_seqlens, sequences, strict=False)]


def _recorded_routes(cu_seqlens, sequences, cp_size):
    layers = [r.get_recorded_indices() for r in RouterReplay.global_router_replay_instances]
    local = torch.stack(layers, dim=1)  # [rows on this TP rank, layers, topk]
    local = torch.cat(_gather_local(local, mpu.get_tensor_model_parallel_group()))
    return [r.to(torch.int16) for r in _whole_sequences(local, cu_seqlens, sequences, cp_size)]


def _reduced_grads(model):
    grads = {}
    for name, param in model.named_parameters():
        if not name.endswith(_GRAD_PARAMS) or param.grad is None:
            continue
        grad = param.grad.float()
        dist.all_reduce(grad, group=mpu.get_context_parallel_group())
        if getattr(param, "sequence_parallel", False):
            dist.all_reduce(grad, group=mpu.get_tensor_model_parallel_group())
        grads[name] = grad.to(torch.bfloat16).cpu()
    return grads


def _traffic(provider, cu_seqlens, cp_rank, cp_size, tp_size):
    """Forward bytes per layer into this rank: SWA halos from the plan, global layers from TE's p2p ring."""
    if cp_size == 1:
        return {"swa_halo": 0, "swa_if_ring": 0, "global_ring": 0}
    row_bytes = (provider.kv_channels + provider.v_head_dim) * 2  # one bf16 key + value head
    local_rows = cu_seqlens[-1] // cp_size
    plan = plan_swa_halo(cu_seqlens, cp_rank=cp_rank, cp_size=cp_size, window=provider.window_size[0], device="cpu")
    halo_rows = sum(plan.recv_splits) - plan.recv_splits[cp_rank]
    swa_heads, full_heads = provider.swa_num_query_groups // tp_size, provider.full_attn_num_query_groups // tp_size
    return {
        "swa_halo": halo_rows * swa_heads * row_bytes,
        "swa_if_ring": (cp_size - 1) * local_rows * swa_heads * row_bytes,
        "global_ring": (cp_size - 1) * local_rows * full_heads * row_bytes,
    }


def run(args):
    dist.init_process_group("nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    provider, model = _build_model(args)
    cp_rank, tp_rank = mpu.get_context_parallel_rank(), mpu.get_tensor_model_parallel_rank()
    sequences = _sequences(args, provider.vocab_size)
    batch = _batch(sequences, cp_rank, args.cp, args.tp)
    cu_seqlens = batch["packed"].cu_seqlens_host
    record = not args.replay_routes and not args.no_save
    if args.replay_routes:
        routes = torch.load(Path(args.replay_routes) / "routes.pt")
        RouterReplay.set_replay_data(_routes_to_replay(routes, sequences, cp_rank, args.cp, tp_rank, args.tp))
        RouterReplay.set_global_router_replay_action(RouterReplayAction.REPLAY_FORWARD)
    elif record:
        RouterReplay.set_global_router_replay_action(RouterReplayAction.RECORD)

    weights_gb = torch.cuda.memory_allocated() / 2**30
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.perf_counter()
    log_probs = _target_log_probs(model, batch, args.log_prob_chunk)
    num_targets = sum(len(seq) - 1 for seq in sequences)
    loss = -(log_probs * batch["weights"]).sum() / num_targets
    recorded = _recorded_routes(cu_seqlens, sequences, args.cp) if record else None
    if args.replay_routes:
        RouterReplay.set_global_router_replay_action(RouterReplayAction.REPLAY_BACKWARD)
    else:
        RouterReplay.clear_global_router_replay_action()
    loss.backward()
    torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    peak = torch.tensor(torch.cuda.max_memory_allocated() / 2**30, device="cuda")
    dist.all_reduce(peak, op=dist.ReduceOp.MAX)

    traffic = _traffic(provider, cu_seqlens, cp_rank, args.cp, args.tp)
    worst = {key: torch.tensor(value, device="cuda") for key, value in traffic.items()}
    for value in worst.values():
        dist.all_reduce(value, op=dist.ReduceOp.MAX)
    if not args.no_save:
        whole_log_probs = _whole_sequences(log_probs.detach().float(), cu_seqlens, sequences, args.cp)
    grads = None if args.no_save else _reduced_grads(model)

    out = Path(args.out)
    if dist.get_rank() == 0:
        out.mkdir(parents=True, exist_ok=True)
        summary = {
            "tp": args.tp,
            "cp": args.cp,
            "seq_lens": [len(seq) for seq in sequences],
            "local_rows": cu_seqlens[-1] // args.cp,
            "first_forward_backward_s": seconds,
            "weights_gb": weights_gb,
            "peak_gb": peak.item(),
            "forward_bytes_per_layer_max_rank": {key: int(value.item()) for key, value in worst.items()},
            "swa_layers": sum(provider.hybrid_attention_pattern),
            "global_layers": len(provider.hybrid_attention_pattern) - sum(provider.hybrid_attention_pattern),
            "loss": loss.item(),
        }
        (out / "summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2), flush=True)
        if not args.no_save:
            torch.save([lp[:-1] for lp in whole_log_probs], out / "log_probs.pt")
            if recorded is not None:
                torch.save(recorded, out / "routes.pt")
    if grads is not None and cp_rank == 0:
        out.mkdir(parents=True, exist_ok=True)
        torch.save(grads, out / f"grads_tp{tp_rank}.pt")
    dist.barrier()
    dist.destroy_process_group()


def _halo_rows(lengths, cp_size, window):
    """Target rows whose window reaches into the previous zigzag chunk, the rows a halo bug would corrupt."""
    rows = []
    for n in lengths:
        chunk = -(-(n + 1) // (2 * cp_size))
        position = torch.arange(n)
        rows.append((position >= chunk) & (position % chunk < window))
    return torch.cat(rows)


def compare(args):
    reference = Path(args.reference)
    want_lp = torch.load(reference / "log_probs.pt")
    failed = False
    for candidate in map(Path, args.candidates):
        cp_size = json.loads((candidate / "summary.json").read_text())["cp"]
        diff = (torch.cat(torch.load(candidate / "log_probs.pt")) - torch.cat(want_lp)).abs()
        halo = _halo_rows([len(lp) for lp in want_lp], cp_size, args.window)
        report = {
            "logprob_mean_abs": diff.mean().item(),
            "logprob_mean_abs_halo_rows": diff[halo].mean().item(),
            "logprob_max_abs": diff.max().item(),
        }
        worst_cos, worst_rel = 1.0, 0.0
        for grads_file in sorted(reference.glob("grads_tp*.pt")):
            want, got = torch.load(grads_file), torch.load(candidate / grads_file.name)
            assert want.keys() == got.keys(), f"{grads_file.name}: different parameters"
            for name in want:
                w, g = want[name].float().flatten(), got[name].float().flatten()
                worst_cos = min(worst_cos, torch.nn.functional.cosine_similarity(w, g, dim=0).item())
                worst_rel = max(worst_rel, ((g - w).norm() / w.norm().clamp_min(1e-30)).item())
        report |= {"grad_min_cosine": worst_cos, "grad_max_rel_err": worst_rel}
        ok = (
            max(report["logprob_mean_abs"], report["logprob_mean_abs_halo_rows"]) <= args.max_mean_abs
            and worst_cos >= args.min_cosine
        )
        failed |= not ok
        print(f"{candidate.name}: {'PASS' if ok else 'FAIL'} {json.dumps(report)}")
        print(f"  {json.loads((candidate / 'summary.json').read_text())['forward_bytes_per_layer_max_rank']}")
    raise SystemExit(1 if failed else 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    run_parser = commands.add_parser("run")
    run_parser.add_argument("--hf-path", required=True, help="BF16 conversion from tools/convert_mimo_v2_to_bf16.py")
    run_parser.add_argument("--tp", type=int, default=1)
    run_parser.add_argument("--cp", type=int, default=1)
    run_parser.add_argument("--ep", type=int, default=0, help="expert parallel size; 0 spans every GPU")
    run_parser.add_argument("--seq-lens", default="65536", help="comma-separated lengths of synthetic sequences")
    run_parser.add_argument("--tokens-file", help="JSON list of token-id lists, used instead of --seq-lens")
    run_parser.add_argument("--seed", type=int, default=0)
    run_parser.add_argument("--replay-routes", help="a CP=1 run directory whose routes to replay")
    run_parser.add_argument("--recompute", action="store_true", help="full activation recompute, as training")
    run_parser.add_argument("--log-prob-chunk", type=int, default=4096, help="the launcher's --log-probs-chunk-size")
    run_parser.add_argument("--no-save", action="store_true", help="report timing and memory only")
    run_parser.add_argument("--out", required=True)
    compare_parser = commands.add_parser("compare")
    compare_parser.add_argument("reference")
    compare_parser.add_argument("candidates", nargs="+")
    compare_parser.add_argument("--max-mean-abs", type=float, default=0.02)
    compare_parser.add_argument("--min-cosine", type=float, default=0.99)
    compare_parser.add_argument("--window", type=int, default=128, help="the SWA layers' sliding window")
    args = parser.parse_args()
    run(args) if args.command == "run" else compare(args)


if __name__ == "__main__":
    main()
