"""Stage (d), on the engine: replay held-out turns through DFLASH servers and report accept length and throughput.

Runs every ``--drafter`` x ``--block-sizes`` x ``--draft-precisions`` configuration on a TP4 server with the
serve_mimo_v26_flash_tinker.py flags, ``--num-engines`` configurations at a time. Each replays ``--replay``
(export_rollouts.py's heldout_replay.json): the turns of a trajectory in order, each generating exactly its recorded
token count with ignore_eos, trajectories ``--concurrency`` at a time. Reports accept length (completion tokens /
verify steps) and output tok/s.

python -m examples.multi_lora.dflash_rl.bench_engine \\
    --hf-checkpoint /data/model-cache/mimo-v26/MiMo-V2.6-Flash-RL-w4a16-linear \\
    --replay /data/dflash-rl/rollouts/heldout_replay.json --out /data/dflash-rl/bench \\
    --drafter shipped=/data/model-cache/mimo-v26/MiMo-V2.6-Flash-RL-w4a16-linear/dflash \\
    --drafter rl=/data/dflash-rl/drafter --block-sizes 6 8 --draft-precisions bf16 fp8
"""

import argparse
import asyncio
import itertools
import json
import time
from pathlib import Path

import httpx

from examples.multi_lora.dflash_rl.engine import LORA_NAME, running_servers, server_argv


async def replay(url: str, workload: list[list[dict]], args) -> dict:
    semaphore = asyncio.Semaphore(args.concurrency)
    sampling = {"temperature": args.temperature, "top_p": args.top_p, "top_k": args.top_k, "ignore_eos": True}

    async def trajectory(client: httpx.AsyncClient, turns: list[dict]) -> tuple[int, int]:
        tokens = steps = 0
        async with semaphore:
            for turn in turns:
                payload = {
                    "input_ids": turn["prompt"],
                    "sampling_params": sampling | {"max_new_tokens": turn["output_len"]},
                }
                if args.lora_path:
                    payload["lora_path"] = LORA_NAME
                response = await client.post(f"{url}/generate", json=payload)
                response.raise_for_status()
                meta = response.json()["meta_info"]
                tokens += meta["completion_tokens"]
                steps += meta["spec_verify_ct"]
        return tokens, steps

    start = time.time()
    async with httpx.AsyncClient(timeout=args.timeout) as client:
        results = await asyncio.gather(*(trajectory(client, turns) for turns in workload))
    wall = time.time() - start
    tokens, steps = map(sum, zip(*results, strict=True))
    return {
        "accept_length": tokens / steps,
        "output_tok_per_s": tokens / wall,
        "completion_tokens": tokens,
        "wall_s": wall,
    }


async def _gather(*coroutines):
    return await asyncio.gather(*coroutines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hf-checkpoint", required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--drafter", action="append", required=True, help="name=path, repeatable")
    parser.add_argument("--block-sizes", type=int, nargs="+", default=[6, 8])
    parser.add_argument("--draft-precisions", nargs="+", choices=["bf16", "fp8"], default=["bf16", "fp8"])
    parser.add_argument("--lora-path", default=None)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.97)
    parser.add_argument("--top-k", type=int, default=1024)
    parser.add_argument("--concurrency", type=int, default=32, help="trajectories in flight per engine")
    parser.add_argument("--num-engines", type=int, default=2)
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--timeout", type=float, default=3600)
    args = parser.parse_args(argv)

    workload = json.loads(args.replay.read_text())
    drafters = [spec.split("=", 1) for spec in args.drafter]
    configs = list(itertools.product(drafters, args.block_sizes, args.draft_precisions))
    args.out.mkdir(parents=True, exist_ok=True)
    results = []
    for wave_start in range(0, len(configs), args.num_engines):
        wave = configs[wave_start : wave_start + args.num_engines]
        ports = [args.port + index for index in range(len(wave))]
        argvs = [
            server_argv(
                hf_checkpoint=args.hf_checkpoint,
                drafter=path,
                block_size=block_size,
                port=port,
                draft_quantization="fp8" if precision == "fp8" else None,
                lora_path=args.lora_path,
            )
            for ((_, path), block_size, precision), port in zip(wave, ports, strict=True)
        ]
        log_dir = args.out / f"wave{wave_start // args.num_engines}"
        log_dir.mkdir(exist_ok=True)
        with running_servers(argvs, ports=ports, log_dir=log_dir) as urls:
            metrics = asyncio.run(_gather(*(replay(url, workload, args) for url in urls)))
        for ((name, _), block_size, precision), metric in zip(wave, metrics, strict=True):
            results.append({"drafter": name, "block_size": block_size, "draft_precision": precision} | metric)
            print(json.dumps(results[-1]), flush=True)
        (args.out / "bench.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
