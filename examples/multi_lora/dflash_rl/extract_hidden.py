"""Stage (b): save the DFlash target hidden states of exported rollouts with the serving engine.

Starts ``--num-engines`` TP4 SGLang servers with the serve_mimo_v26_flash_tinker.py flags, DFLASH with
``--drafter`` (so the target captures the drafter's layers), no radix cache, and capture_hook.py as a forward hook.
Each rollout is prefilled with ``max_new_tokens=1``; the hook saves the rows the drafter needs and this script
joins them into ``<out>/<key>.safetensors`` (format in data.py). Rerunning skips finished rollouts. With
``--lora-path`` the target applies that policy adapter, as the rollout engine does.

python -m examples.multi_lora.dflash_rl.extract_hidden \\
    --hf-checkpoint /data/model-cache/mimo-v26/MiMo-V2.6-Flash-RL-w4a16-linear \\
    --rollouts /data/dflash-rl/rollouts/train.jsonl --out /data/dflash-rl/hidden/train
"""

import argparse
import asyncio
import json
import time
from pathlib import Path

import httpx
import torch

from examples.multi_lora.dflash_rl.data import assemble_shard, context_spans, save_shard
from examples.multi_lora.dflash_rl.drafter import DrafterConfig
from examples.multi_lora.dflash_rl.engine import LORA_NAME, running_servers, server_argv


def _hook_flags(capture_dir: Path) -> tuple[str, ...]:
    hook = {
        "name": "dflash_capture",
        "target_modules": ["logits_processor"],
        "hook_factory": "examples.multi_lora.dflash_rl.capture_hook:make_hook",
        "config": {"dir": str(capture_dir)},
    }
    return ("--disable-radix-cache", "--forward-hooks", json.dumps([hook]))


def _collect(rollout: dict, rid: str, spans: list, args, config: DrafterConfig) -> None:
    chunk_paths = sorted(args.capture_dir.glob(f"{rid}.*.pt"))
    chunks = [torch.load(path, weights_only=True) for path in chunk_paths]
    shard = assemble_shard(rollout["tokens"], rollout["loss_mask"], spans, chunks, config.target_hidden_size)
    save_shard(shard, args.out / f"{rollout['key']}.safetensors")
    for path in [*chunk_paths, args.capture_dir / f"{rid}.spans.json"]:
        path.unlink()


async def _extract(client: httpx.AsyncClient, url: str, rollout: dict, args, config: DrafterConfig) -> None:
    rid = f"dflash-{rollout['key']}"
    # Chunk boundaries depend on the batch, so a failed earlier attempt's files would not line up.
    for path in args.capture_dir.glob(f"{rid}.*.pt"):
        path.unlink()
    spans = context_spans(rollout["loss_mask"], config.sliding_window)
    (args.capture_dir / f"{rid}.spans.json").write_text(json.dumps(spans))
    payload = {"input_ids": rollout["tokens"], "rid": rid, "sampling_params": {"max_new_tokens": 1, "temperature": 0}}
    if args.lora_path:
        payload["lora_path"] = LORA_NAME
    response = await client.post(f"{url}/generate", json=payload)
    response.raise_for_status()
    await asyncio.to_thread(_collect, rollout, rid, spans, args, config)


async def _run(urls: list[str], rollouts: list[dict], args, config: DrafterConfig) -> None:
    queue = list(reversed(rollouts))
    done, tokens, start = 0, 0, time.time()

    async def worker(client: httpx.AsyncClient, url: str) -> None:
        nonlocal done, tokens
        while queue:
            rollout = queue.pop()
            await _extract(client, url, rollout, args, config)
            done, tokens = done + 1, tokens + len(rollout["tokens"])
            if done % 20 == 0 or done == len(rollouts):
                elapsed = time.time() - start
                print(json.dumps({"done": done, "of": len(rollouts), "prefill_tok_per_s": round(tokens / elapsed)}))

    async with httpx.AsyncClient(timeout=args.timeout) as client:
        await asyncio.gather(*(worker(client, url) for url in urls for _ in range(args.concurrency)))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hf-checkpoint", required=True)
    parser.add_argument("--drafter", type=Path, default=None, help="default: <hf-checkpoint>/dflash")
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--capture-dir", type=Path, default=Path("/tmp/dflash_capture"))
    parser.add_argument("--lora-path", default=None)
    parser.add_argument("--num-engines", type=int, default=2)
    parser.add_argument("--concurrency", type=int, default=4, help="requests in flight per engine")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--timeout", type=float, default=3600)
    args = parser.parse_args(argv)

    drafter = args.drafter or Path(args.hf_checkpoint) / "dflash"
    config = DrafterConfig.from_dir(drafter)
    args.out.mkdir(parents=True, exist_ok=True)
    args.capture_dir.mkdir(parents=True, exist_ok=True)
    with open(args.rollouts) as file:
        rollouts = [json.loads(line) for line in file]
    todo = [rollout for rollout in rollouts if not (args.out / f"{rollout['key']}.safetensors").exists()]
    print(json.dumps({"rollouts": len(rollouts), "todo": len(todo)}))
    ports = [args.port + index for index in range(args.num_engines)]
    argvs = [
        server_argv(
            hf_checkpoint=args.hf_checkpoint,
            drafter=str(drafter),
            block_size=config.block_size,
            port=port,
            lora_path=args.lora_path,
            extra=_hook_flags(args.capture_dir),
        )
        for port in ports
    ]
    with running_servers(argvs, ports=ports, log_dir=args.out) as urls:
        asyncio.run(_run(urls, todo, args, config))


if __name__ == "__main__":
    main()
