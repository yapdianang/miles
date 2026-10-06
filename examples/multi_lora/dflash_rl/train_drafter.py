"""Stages (c) and (e): fine-tune the DFlash drafter on hidden-state shards and write a servable drafter directory.

Starts from ``--drafter`` (the checkpoint's dflash/), keeps the target embedding and LM head frozen, and trains
every drafter weight and the mask embedding. Each rank reads one shard per step; its blocks start at every sampled
token with a sampled successor (``num_anchors`` at most, sampled uniformly), and slot k's cross entropy against the
rollout's sampled token has weight exp(-(k - 1) / loss_decay_gamma), from the drafter's config. The loss is the
weighted mean over every slot of the global batch. After each epoch, rank 0 writes ``--out``: the weights,
mask_embedding.pt, and the config and code of ``--drafter``, for --speculative-draft-model-path.

torchrun --nproc-per-node 8 -m examples.multi_lora.dflash_rl.train_drafter \\
    --target-checkpoint /data/model-cache/mimo-v26/MiMo-V2.6-Flash-RL-w4a16-linear \\
    --drafter /data/model-cache/mimo-v26/MiMo-V2.6-Flash-RL-w4a16-linear/dflash \\
    --data /data/dflash-rl/hidden/train --out /data/dflash-rl/drafter
"""

import argparse
import json
import math
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.distributed as dist
from examples.multi_lora.dflash_rl.data import load_shard, make_chunk, valid_anchors
from examples.multi_lora.dflash_rl.drafter import (
    block_loss,
    decay_weights,
    load_drafter,
    load_target_head,
    save_drafter,
)
from safetensors import safe_open
from torch.nn.parallel import DistributedDataParallel


def _init_distributed() -> tuple[int, int, torch.device]:
    if "RANK" not in os.environ:
        return 0, 1, torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo")
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
        torch.cuda.set_device(device)
    return dist.get_rank(), dist.get_world_size(), device


def _all_reduce(tensor: torch.Tensor, world: int) -> torch.Tensor:
    if world > 1:
        dist.all_reduce(tensor)
    return tensor


def trainable_shards(data: Path) -> list[Path]:
    """Shards with at least one block start, read without loading hidden states."""
    shards = []
    for path in sorted(data.glob("*.safetensors")):
        with safe_open(path, framework="pt") as handle:
            shard = {name: handle.get_tensor(name) for name in ("positions", "loss_mask")}
        if len(valid_anchors({**shard, "loss_mask": shard["loss_mask"].bool()})):
            shards.append(path)
    return shards


def lr_scale(step: int, *, warmup: int, total: int) -> float:
    if step < warmup:
        return (step + 1) / warmup
    return 0.5 * (1 + math.cos(math.pi * min(1.0, (step - warmup) / max(1, total - warmup))))


def train_step(model, shard: dict, args, *, embed, lm_head, generator: torch.Generator, world: int) -> torch.Tensor:
    """Accumulate one shard's gradient; return [loss sum, weight sum, accepted sum, blocks] of this rank."""
    config = model.module.config if world > 1 else model.config
    device = embed.device
    anchors = valid_anchors(shard)
    if len(anchors) > args.max_anchors:
        keep = torch.randperm(len(anchors), generator=generator)[: args.max_anchors].to(anchors.device)
        anchors = anchors[keep].sort().values
    chunks = [
        make_chunk(
            shard, group, block_size=args.block_size, window=config.sliding_window, mask_token_id=config.mask_token_id
        )
        for group in anchors.split(args.anchor_chunk)
    ]
    decay = decay_weights(args.block_size, config.loss_decay_gamma, device)
    weight = sum((chunk.label_mask * decay).sum() for chunk in chunks)
    global_weight = _all_reduce(weight.detach().clone(), world)
    stats = torch.zeros(4, device=device)
    for index, chunk in enumerate(chunks):
        last = index == len(chunks) - 1
        with model.no_sync() if world > 1 and not last else nullcontext():
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                hidden = model(
                    chunk.block_ids,
                    chunk.block_positions,
                    chunk.ctx_hidden,
                    chunk.ctx_positions,
                    chunk.ctx_visible,
                    embed,
                )
                loss_sum, weight_sum, accepted = block_loss(
                    hidden, chunk.labels, chunk.label_mask, lm_head, config.loss_decay_gamma
                )
            (loss_sum * (world / global_weight)).backward()
        blocks = torch.full((), len(accepted), dtype=torch.float32, device=device)
        stats += torch.stack([loss_sum.detach(), weight_sum, accepted.sum().float(), blocks])
    return stats


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--target-checkpoint", type=Path, required=True, help="has embed_tokens and lm_head")
    parser.add_argument("--drafter", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--block-size", type=int, default=None, help="default: the drafter's block_size")
    parser.add_argument("--max-anchors", type=int, default=None, help="blocks per shard; default: num_anchors")
    parser.add_argument("--anchor-chunk", type=int, default=128, help="blocks per forward")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    rank, world, device = _init_distributed()
    drafter = load_drafter(args.drafter).to(device)
    args.block_size = args.block_size or drafter.config.block_size
    args.max_anchors = args.max_anchors or drafter.config.num_anchors
    head_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    embed, lm_head = load_target_head(args.target_checkpoint, device, head_dtype)
    shards = trainable_shards(args.data)
    steps_per_epoch = len(shards) // world
    if steps_per_epoch == 0:
        raise ValueError(f"{len(shards)} trainable shards in {args.data} for {world} ranks")
    total_steps = steps_per_epoch * args.epochs
    model = drafter
    if world > 1:
        model = DistributedDataParallel(drafter, device_ids=[device.index] if device.type == "cuda" else None)
    optimizer = torch.optim.AdamW(
        drafter.parameters(), lr=args.lr, weight_decay=args.weight_decay, fused=device.type == "cuda"
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: lr_scale(step, warmup=args.warmup_steps, total=total_steps)
    )
    generator = torch.Generator().manual_seed(args.seed + rank)
    step, start, totals = 0, time.time(), torch.zeros(4, device=device)
    for epoch in range(args.epochs):
        order = random.Random(args.seed + epoch).sample(shards, len(shards))
        mine = order[rank::world][:steps_per_epoch]
        with ThreadPoolExecutor(1) as loader:
            pending = loader.submit(load_shard, mine[0])
            for index in range(steps_per_epoch):
                shard = {name: tensor.to(device) for name, tensor in pending.result().items()}
                if index + 1 < steps_per_epoch:
                    pending = loader.submit(load_shard, mine[index + 1])
                totals += train_step(
                    model, shard, args, embed=embed, lm_head=lm_head, generator=generator, world=world
                )
                torch.nn.utils.clip_grad_norm_(drafter.parameters(), args.grad_clip)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                if step % args.log_every == 0 or step == total_steps:
                    loss, weight, accepted, blocks = _all_reduce(totals, world).tolist()
                    if rank == 0:
                        log = {"epoch": epoch, "step": step, "of": total_steps, "loss": loss / weight}
                        log |= {"accept_length": 1 + accepted / blocks, "blocks": int(blocks)}
                        log |= {"lr": scheduler.get_last_lr()[0], "elapsed_s": round(time.time() - start)}
                        print(json.dumps(log), flush=True)
                    totals = torch.zeros(4, device=device)
        if rank == 0:
            save_drafter(drafter, source=args.drafter, out=args.out)
        if world > 1:
            dist.barrier()
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
