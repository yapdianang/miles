"""Per-turn engine-to-gateway traffic with and without engine-kept rollout records (MiMo-V2.6 section 6.4).

Replays Tau3 trajectories through the gateway's sampling path (MilesBackend.sample) against the engines' router.
Each turn samples the recorded output length and the next turn's context appends the sampled tokens and the
recorded environment tokens, so turns build on their own samples as in a rollout. Arms, at each concurrency:

  per_turn  every turn returns its route deltas and sampling supports (today's path)
  keep      the engine keeps them; each batch of final datums collects them once, as forward_backward does

Reported per arm: engine-to-gateway response bytes per turn, turn latency, and per collection batch its latency
and bytes. Engines need --sglang-rollout-record-cache-gb (launcher --rollout-record-cache-gb).

--check compares the two paths on the same sampled tokens. Start the engines with
SGLANG_ROLLOUT_RECORDS_ALSO_RETURN=1 so that every kept turn also returns its record: a per-turn store records
that response exactly as today, and every datum's collected routes and supports must equal the per-turn ones.

python examples/multi_lora/bench_rollout_records.py --router-url http://HOST:PORT --replay replay.json \\
    --concurrency 64 128 256
"""

import argparse
import asyncio
import contextvars
import json
import math
import random
import statistics
import time
from collections import Counter

import httpx
import numpy as np

import miles.tinker.runtime as runtime
from miles.tinker.runtime import MilesBackend, _build_train_data
from miles.tinker.sampler_records import SamplerRecordStore, prefix_hashes
from miles.utils.http_utils import router_worker_base_urls

# response bytes by endpoint
_bytes: Counter = Counter()
# --check: the response a keep-arm call received, replayed to the per-turn store
_captured: contextvars.ContextVar[dict | None] = contextvars.ContextVar("captured", default=None)


async def _post(client: httpx.AsyncClient, url: str, payload: dict) -> dict:
    captured = _captured.get()
    if captured is not None and "response" in captured:
        return captured["response"]
    response = await client.post(url, json=payload)
    response.raise_for_status()
    _bytes[url.rsplit("/", 1)[-1]] += len(response.content)
    if captured is not None and url.endswith("/generate"):
        captured["response"] = response.json()
        return captured["response"]
    return response.json()


async def _engine_urls(client: httpx.AsyncClient, router_url: str) -> list[str]:
    response = await client.get(f"{router_url}/workers")
    if response.status_code == 404:
        response = await client.get(f"{router_url}/list_workers")
        return router_worker_base_urls(response.json()["urls"])
    return router_worker_base_urls([worker["url"] for worker in response.json()["workers"]])


def _backend(args, client: httpx.AsyncClient, *, keep: bool) -> MilesBackend:
    store = SamplerRecordStore(64 << 30, supports=True, routes=True, collect=keep)
    return MilesBackend(
        None,
        args.router_url,
        num_layers=args.num_layers,
        sampler_records=store,
        engine_urls=lambda: _engine_urls(client, args.router_url),
    )


async def _run_trajectory(args, backends, label: str, prefix: list[int], turns: list[dict], latencies: list) -> dict:
    """Sample every turn on its own earlier samples; return the final datum, weighting its sampled targets."""
    context, weights = prefix + turns[0]["prompt"], []
    for index, turn in enumerate(turns):
        payload = {
            "prompt_tokens": context,
            "sampling_params": {
                "max_tokens": turn["output_len"],
                "temperature": args.temperature,
                "top_p": args.top_p,
                "top_k": args.top_k,
                "stop": [],
            },
            "num_samples": 1,
            "prompt_logprobs": False,
            "topk_prompt_logprobs": 0,
        }
        sequence_id = f"{label}-{index}"
        _captured.set({} if len(backends) > 1 else None)
        start = time.perf_counter()
        results = [await backend.sample(payload, args.lora_name, sequence_ids=[sequence_id]) for backend in backends]
        latencies.append(time.perf_counter() - start)
        for result in results:
            assert "error" not in result, result
        sampled = results[0]["sequences"][0]["tokens"]
        assert all(result["sequences"][0]["tokens"] == sampled for result in results)
        environment = []
        if index + 1 < len(turns):
            environment = turns[index + 1]["prompt"][len(turn["prompt"]) + turn["output_len"] :]
        weights += [0.0] * (len(context) - 1 - len(weights)) + [1.0] * len(sampled)
        context = context + sampled + environment
    weights += [0.0] * (len(context) - 1 - len(weights))
    return {"tokens": context, "target_len": len(context) - 1, "target_tokens": context[1:], "weights": weights}


async def _sample_arm(args, workload, backends, label: str, concurrency: int) -> tuple[list[dict], list[float]]:
    copies = max(1, math.ceil(concurrency / len(workload)))
    rng = random.Random(label)
    jobs = [([rng.randrange(1000, 100_000) for _ in range(16)], turns) for _ in range(copies) for turns in workload]
    semaphore, latencies = asyncio.Semaphore(concurrency), []

    async def run(index: int, prefix: list[int], turns: list[dict]) -> dict:
        async with semaphore:
            return await _run_trajectory(args, backends, f"{label}-{index}", prefix, turns, latencies)

    datums = await asyncio.gather(*(run(index, prefix, turns) for index, (prefix, turns) in enumerate(jobs)))
    return datums, latencies


async def _collect(backend: MilesBackend, datums: list[dict], batch_size: int) -> list[float]:
    """Collection latency of each forward_backward-sized batch of datums."""
    latencies = []
    for start in range(0, len(datums), batch_size):
        begin = time.perf_counter()
        await backend._collect_rollout_records(datums[start : start + batch_size])
        latencies.append(time.perf_counter() - begin)
    return latencies


def _coverage(backend: MilesBackend, datums: list[dict]) -> tuple[int, int]:
    """Datums with complete routes, and sampled positions without a support."""
    records = backend.sampler_records
    routed = unsupported = 0
    for datum in datums:
        hashes = prefix_hashes(np.asarray(datum["tokens"][:-1], dtype=np.int64))
        routed += records.datum_routes(datum, hashes) is not None
        _, offsets, _ = records.datum_supports(datum, hashes)
        unsupported += int(((np.diff(offsets) == 0) & (np.asarray(datum["weights"]) > 0)).sum())
    return routed, unsupported


def _summary(values: list[float]) -> str:
    values = sorted(values)
    return f"mean {statistics.fmean(values):.3f}s p50 {values[len(values) // 2]:.3f}s p90 {values[int(len(values) * 0.9)]:.3f}s"


async def _bench(args, workload, client: httpx.AsyncClient) -> None:
    for concurrency in args.concurrency:
        for arm in ("per_turn", "keep"):
            backend = _backend(args, client, keep=arm == "keep")
            _bytes.clear()
            datums, latencies = await _sample_arm(args, workload, [backend], f"{arm}-{concurrency}", concurrency)
            turn_bytes = _bytes["generate"]
            collect_latencies = await _collect(backend, datums, args.collect_batch) if arm == "keep" else []
            routed, unsupported = _coverage(backend, datums)
            records = backend.sampler_records
            line = (
                f"conc {concurrency} {arm}: {len(latencies)} turns, {turn_bytes / len(latencies) / 1e3:.1f} KB/turn "
                f"engine->gateway, turn latency {_summary(latencies)}"
            )
            if collect_latencies:
                line += (
                    f"; collection {_bytes['collect_rollout_records'] / len(datums) / 1e6:.2f} MB/datum, "
                    f"{_bytes['collect_rollout_records'] / len(latencies) / 1e3:.1f} KB/turn, latency "
                    f"{_summary(collect_latencies)} per {args.collect_batch}-datum batch; records "
                    f"{records.num_collected} collected, {records.num_missing} missing, "
                    f"{records.num_recomputed} datums recomputed"
                )
            print(
                f"{line}; {routed}/{len(datums)} datums routed, {unsupported} sampled positions unsupported",
                flush=True,
            )


async def _check(args, workload, client: httpx.AsyncClient) -> None:
    keep, per_turn = _backend(args, client, keep=True), _backend(args, client, keep=False)
    datums, _ = await _sample_arm(args, workload, [keep, per_turn], "check", args.concurrency[0])
    await _collect(keep, datums, args.collect_batch)
    mismatches = Counter()
    for start in range(0, len(datums), args.collect_batch):
        batch = [(0, datum) for datum in datums[start : start + args.collect_batch]]
        collected = _build_train_data(batch, sampler_records=keep.sampler_records, loss_fn="score_centering")
        expected = _build_train_data(batch, sampler_records=per_turn.sampler_records, loss_fn="score_centering")
        for key in (
            "rollout_routed_experts",
            "rollout_sampling_mask_ids",
            "rollout_sampling_mask_offsets",
            "rollout_sampling_mask_log_probs",
        ):
            if (key in collected) != (key in expected):
                mismatches[f"{key} present"] += len(batch)
                continue
            for left, right in zip(collected.get(key, []), expected.get(key, []), strict=True):
                if not np.array_equal(np.asarray(left), np.asarray(right)):
                    mismatches[key] += 1
    records = keep.sampler_records
    routed, unsupported = _coverage(keep, datums)
    print(
        f"check: {len(datums)} datums, {records.num_collected} records collected, {records.num_missing} missing, "
        f"{records.num_recomputed} recomputed, {routed} routed, {unsupported} sampled positions unsupported; "
        f"mismatching datums: {dict(mismatches) or 'none'}",
        flush=True,
    )


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--router-url", required=True)
    parser.add_argument("--replay", required=True, help="JSON list of trajectories of {prompt, output_len} turns")
    parser.add_argument("--concurrency", type=int, nargs="+", default=[64, 128, 256])
    parser.add_argument("--trajectories", type=int, default=None, help="use the first N trajectories")
    parser.add_argument("--collect-batch", type=int, default=64, help="datums per forward_backward")
    parser.add_argument("--num-layers", type=int, default=48)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.97)
    parser.add_argument("--top-k", type=int, default=1024)
    parser.add_argument("--lora-name", default=None)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    workload = json.load(open(args.replay))[: args.trajectories]
    async with httpx.AsyncClient(timeout=httpx.Timeout(None), limits=httpx.Limits(max_connections=1024)) as client:
        runtime.post = lambda url, payload, **_: _post(client, url, payload)
        await (_check if args.check else _bench)(args, workload, client)


if __name__ == "__main__":
    asyncio.run(main())
