"""Stage (d), offline: accept length of drafters on held-out hidden-state shards.

For each drafter and block size, drafts every block start of ``--data`` greedily, as SGLang's DFlash draft sampler
does, against the rollouts' sampled tokens. Reports the decay-weighted loss, the mean accept length over block
starts, and the walk accept length: decode each turn block by block from its first sampled token (tokens after the
first / verify steps), bench_engine.py's accept_length_after_first. With sampled verification, matching a sampled
token is accepted with the target's probability of the draft, so both estimate engine acceptance on the same
contexts.

python -m examples.multi_lora.dflash_rl.eval_offline \\
    --target-checkpoint /data/model-cache/mimo-v26/MiMo-V2.6-Flash-RL-w4a16-linear \\
    --data /data/dflash-rl/hidden/heldout --block-sizes 6 8 \\
    --drafter shipped=/data/model-cache/mimo-v26/MiMo-V2.6-Flash-RL-w4a16-linear/dflash \\
    --drafter rl=/data/dflash-rl/drafter
"""

import argparse
import json
from pathlib import Path

import torch

from examples.multi_lora.dflash_rl.data import load_shard, make_chunk, valid_anchors, walk_accept
from examples.multi_lora.dflash_rl.drafter import DFlashDrafter, block_loss, load_drafter, load_target_head


@torch.no_grad()
def evaluate(
    drafter: DFlashDrafter, shards: list[Path], *, embed, lm_head, block_size: int, anchor_chunk: int
) -> dict:
    config, device = drafter.config, embed.device
    totals = dict.fromkeys(("loss", "weight", "accepted", "blocks", "walk_tokens", "walk_steps"), 0.0)
    for path in shards:
        shard = load_shard(path, str(device))
        anchors = valid_anchors(shard)
        accepted = []
        for group in anchors.split(anchor_chunk):
            chunk = make_chunk(
                shard, group, block_size=block_size, window=config.sliding_window, mask_token_id=config.mask_token_id
            )
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                hidden = drafter(
                    chunk.block_ids,
                    chunk.block_positions,
                    chunk.ctx_hidden,
                    chunk.ctx_positions,
                    chunk.ctx_visible,
                    embed,
                )
                loss, weight, group_accepted = block_loss(
                    hidden, chunk.labels, chunk.label_mask, lm_head, config.loss_decay_gamma
                )
            totals["loss"] += loss.item()
            totals["weight"] += weight.item()
            accepted += group_accepted.tolist()
        walk_tokens, walk_steps = walk_accept(shard, anchors.tolist(), accepted)
        totals["accepted"] += sum(accepted)
        totals["blocks"] += len(accepted)
        totals["walk_tokens"] += walk_tokens
        totals["walk_steps"] += walk_steps
    return {
        "blocks": int(totals["blocks"]),
        "loss": totals["loss"] / totals["weight"],
        "mean_accept_length": 1 + totals["accepted"] / totals["blocks"],
        "walk_accept_length": totals["walk_tokens"] / totals["walk_steps"],
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--target-checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--drafter", action="append", required=True, help="name=path, repeatable")
    parser.add_argument("--block-sizes", type=int, nargs="+", default=[6, 8])
    parser.add_argument("--anchor-chunk", type=int, default=256)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embed, lm_head = load_target_head(
        args.target_checkpoint, device, torch.bfloat16 if device.type == "cuda" else torch.float32
    )
    shards = sorted(args.data.glob("*.safetensors"))
    results = []
    for spec in args.drafter:
        name, path = spec.split("=", 1)
        drafter = load_drafter(Path(path)).to(device).eval()
        for block_size in args.block_sizes:
            metrics = evaluate(
                drafter, shards, embed=embed, lm_head=lm_head, block_size=block_size, anchor_chunk=args.anchor_chunk
            )
            results.append({"drafter": name, "block_size": block_size} | metrics)
            print(json.dumps(results[-1]), flush=True)
        del drafter
    if args.output is not None:
        args.output.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
