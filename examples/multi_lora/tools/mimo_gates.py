"""Numerics and throughput gates for MiMo-V2.6-Flash-RL on a Miles Tinker gateway.

Each gate prints one JSON summary on stdout and one summary line on stderr. A gate with a bar adds
``passed`` and ``failures`` to the summary and exits 1 when it fails.

For a sampled token, d = target - reference log-probability (trainer - sampler unless named otherwise);
k3 = mean(exp(d) - d - 1) estimates KL(reference || target), with mean |d| and mean d beside it.

Gates with ``--workload`` read a replay workload: a JSON (or ``.json.gz``) list of trajectories, each a
list of turns ``{"prompt": [token ids], "output_len": n}``, where each turn's prompt extends the previous
turn's context.

python examples/multi_lora/tools/mimo_gates.py parity --base-url http://localhost:10613
python examples/multi_lora/tools/mimo_gates.py kl-decompose --base-url http://localhost:10613 --workload replay.json.gz
"""

from __future__ import annotations

import argparse
import asyncio
import gzip
import itertools
import json
import math
import random
import re
import statistics
import sys
import time
from pathlib import Path

from transformers import AutoTokenizer

import tinker
from tinker import types

MODEL = "XiaomiMiMo/MiMo-V2.6-Flash-RL"
REVISION = "5711b268169967567844e1e560e8a3966da959b1"
PARITY_PROMPTS = (
    "Book the cheapest one-way flight from SFO to JFK on May 5 for one adult and explain each step.",
    "A customer wants to return two items from order #W123 and exchange a third for a larger size. "
    "Walk through the policy checks you would run before acting.",
    "Write a Python function that merges overlapping intervals and explain its complexity.",
    "My phone plan was charged twice this month. What information do you need to investigate?",
)
LENGTH_BUCKETS = ((0, 6_000), (6_000, 16_000), (16_000, 32_000), (32_000, 64_000), (64_000, 200_000))
# (reference, target): engine noise, decode vs prefill, engine vs trainer, and the end-to-end gap.
KL_PAIRS = (("pre1", "pre2"), ("dec", "pre1"), ("pre1", "tr"), ("dec", "tr"))
REPLAY_PREFIX_TOKENS = 16

_PREFILL = re.compile(
    r"TP0 EP0\] Prefill batch, #new-seq: (\d+), #new-token: (\d+), #cached-token: (\d+), full token usage: ([\d.]+)"
)
_DECODE = re.compile(
    r"TP0 EP0\] Decode batch, #running-req: (\d+).*gen throughput \(token/s\): ([\d.]+), #queue-req: (\d+)"
)
_ACCEPT = re.compile(r"accept len: ([\d.]+)")


def gap(target: list[float], reference: list[float]) -> dict:
    """k3, mean |d| and mean d of d = target - reference over tokens."""
    d = [t - r for t, r in zip(target, reference, strict=True)]
    return {
        "tokens": len(d),
        "k3": sum(math.exp(v) - v - 1 for v in d) / len(d),
        "mean_abs": sum(abs(v) for v in d) / len(d),
        "mean_d": sum(d) / len(d),
    }


def build_datum(prompt: list[int], tokens: list[int], logprobs: list[float], advantage: float) -> types.Datum:
    """An importance-sampling datum whose loss inputs are zero over the prompt."""
    full = prompt + tokens
    pad = [0.0] * (len(prompt) - 1)
    return types.Datum(
        model_input=types.ModelInput.from_ints(full[:-1]),
        loss_fn_inputs={
            "target_tokens": full[1:],
            "logprobs": pad + logprobs,
            "advantages": pad + [advantage] * len(tokens),
        },
    )


def trainer_logprobs(output: dict, prompt_length: int) -> list[float]:
    return output["logprobs"].tolist()[prompt_length - 1 :]


def load_workload(path: Path) -> list[list[dict]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as file:
        return json.load(file)


def kl_contexts(workload: list[list[dict]], count: int, min_length: int, max_length: int) -> list[list[int]]:
    prompts = (turn["prompt"] for trajectory in workload for turn in trajectory)
    return [prompt for prompt in prompts if min_length <= len(prompt) < max_length][:count]


def length_buckets(workload: list[list[dict]], per_bucket: int) -> list[tuple[str, list[list[int]]]]:
    prompts = [turn["prompt"] for trajectory in workload for turn in trajectory]
    buckets = []
    for low, high in LENGTH_BUCKETS:
        contexts = [prompt for prompt in prompts if low <= len(prompt) < high][:per_bucket]
        if contexts:
            buckets.append((f"{low // 1000}-{high // 1000}K", contexts))
    return buckets


def longest_contexts(workload: list[list[dict]], count: int) -> list[list[int]]:
    """The last-turn prompts of the `count` longest trajectories."""
    return sorted((trajectory[-1]["prompt"] for trajectory in workload), key=len, reverse=True)[:count]


def long_context(workload: list[list[dict]], tokens: int) -> list[int]:
    """The last-turn prompts, longest first and repeated as needed, concatenated and cut to `tokens` tokens."""
    context = []
    for prompt in itertools.cycle(longest_contexts(workload, len(workload))):
        context += prompt
        if len(context) >= tokens:
            return context[:tokens]


def replay_jobs(workload: list[list[dict]], copies: int, rng: random.Random) -> list[tuple[list[int], list[dict]]]:
    """Each copy of a trajectory gets its own token prefix, so copies share no prefix cache."""
    jobs = [
        ([rng.randrange(1000, 100_000) for _ in range(REPLAY_PREFIX_TOKENS)], turns)
        for _ in range(copies)
        for turns in workload
    ]
    rng.shuffle(jobs)
    return jobs


def engine_stats(lines) -> dict:
    """Prefix-cache hit rate and decode throughput from SGLang TP0 batch log lines."""
    seqs = new = cached = 0
    full_usage = 0.0
    running, throughput, queue, accept = [], [], [], []
    for line in lines:
        if match := _PREFILL.search(line):
            seqs += int(match[1])
            new += int(match[2])
            cached += int(match[3])
            full_usage = max(full_usage, float(match[4]))
        elif match := _DECODE.search(line):
            running.append(int(match[1]))
            throughput.append(float(match[2]))
            queue.append(int(match[3]))
            if accepted := _ACCEPT.search(line):
                accept.append(float(accepted[1]))
    # Logs from idle intervals report near-zero throughput.
    steady = [value for value in throughput if value > 1.0]
    return {
        "prefill_seqs": seqs,
        "new_tokens": new,
        "cached_tokens": cached,
        "cache_hit_rate": cached / (new + cached) if new + cached else None,
        "max_full_token_usage": full_usage,
        "decode_logs": len(throughput),
        "mean_gen_tokens_per_second": statistics.fmean(steady) if steady else None,
        "max_gen_tokens_per_second": max(throughput, default=None),
        "mean_running": statistics.fmean(running) if running else None,
        "mean_queue": statistics.fmean(queue) if queue else None,
        "mean_accept_length": statistics.fmean(accept) if accept else None,
    }


def _service(args) -> tinker.ServiceClient:
    return tinker.ServiceClient(base_url=args.base_url, api_key="tml-dummy")


def _training_client(args):
    return _service(args).create_lora_training_client(args.model, rank=args.rank)


def _sample(sampler, prompts: list[list[int]], params: types.SamplingParams, num_samples: int = 1) -> list[tuple]:
    """(prompt, sampled sequence) pairs, with all requests in flight at once."""
    futures = [
        sampler.sample(types.ModelInput.from_ints(prompt), num_samples=num_samples, sampling_params=params)
        for prompt in prompts
    ]
    return [
        (prompt, sequence)
        for prompt, future in zip(prompts, futures, strict=True)
        for sequence in future.result().sequences
    ]


def _datums(samples: list[tuple], advantages: list[float]) -> list[types.Datum]:
    return [
        build_datum(prompt, list(sequence.tokens), list(sequence.logprobs), advantage)
        for (prompt, sequence), advantage in zip(samples, advantages, strict=True)
    ]


def _trainer_gap(samples: list[tuple], result) -> dict:
    trainer, sampler = [], []
    for (prompt, sequence), output in zip(samples, result.loss_fn_outputs, strict=True):
        trainer += trainer_logprobs(output, len(prompt))
        sampler += list(sequence.logprobs)
    return gap(trainer, sampler)


def _timed(seconds: dict, label: str, call):
    start = time.time()
    result = call()
    seconds[label] = time.time() - start
    return result


def _with_bar(summary: dict, failures: list[str]) -> dict:
    return summary | {"passed": not failures, "failures": failures}


def run_parity(args) -> tuple[dict, str]:
    """A wrong adapter export keeps base parity but breaks it after the update."""
    training = _training_client(args)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, revision=args.revision)
    prompts = [
        # Render, then encode: some transformers return a BatchEncoding from tokenize=True.
        tokenizer.encode(
            tokenizer.apply_chat_template(
                [{"role": "user", "content": text}], add_generation_prompt=True, tokenize=False
            ),
            add_special_tokens=False,
        )
        for text in PARITY_PROMPTS
    ]
    params = types.SamplingParams(max_tokens=args.max_tokens, temperature=1.0, top_p=args.top_p, top_k=args.top_k)
    seconds = {}
    sampler = _timed(seconds, "save_s0", lambda: training.save_weights_and_get_sampling_client("s0"))
    samples = _timed(seconds, "sample_s0", lambda: _sample(sampler, prompts, params, num_samples=2))
    # One sample per prompt is pushed up, the other down.
    advantages = [1.0 if index % 2 == 0 else -1.0 for index in range(len(samples))]
    result = _timed(
        seconds,
        "forward_backward",
        lambda: training.forward_backward(_datums(samples, advantages), "importance_sampling").result(),
    )
    base = _trainer_gap(samples, result)
    _timed(seconds, "optim_step", lambda: training.optim_step(types.AdamParams(learning_rate=args.lr)).result())
    updated = _timed(seconds, "save_s1", lambda: training.save_weights_and_get_sampling_client("s1"))
    new_samples = _timed(seconds, "sample_s1", lambda: _sample(updated, prompts, params, num_samples=2))
    result = _timed(
        seconds,
        "forward_s1",
        lambda: training.forward(_datums(new_samples, [0.0] * len(new_samples)), "importance_sampling").result(),
    )
    after = _trainer_gap(new_samples, result)
    rescored = [
        updated.compute_logprobs(types.ModelInput.from_ints(prompt + list(sequence.tokens)))
        for prompt, sequence in samples
    ]
    moves = [
        {"advantage": advantage, "before": sum(sequence.logprobs), "after": sum(future.result()[len(prompt) :])}
        for (prompt, sequence), advantage, future in zip(samples, advantages, rescored, strict=True)
    ]
    failures = [
        f"{name} k3 {stats['k3']:.5f} > {args.max_k3}"
        for name, stats in (("base", base), ("updated", after))
        if stats["k3"] > args.max_k3
    ]
    summary = _with_bar(
        {
            "base": base,
            "updated": after,
            "logprob_sums": moves,
            "sample_text": tokenizer.decode(samples[0][1].tokens)[:200],
            "seconds": seconds,
        },
        failures,
    )
    line = (
        f"parity top-p {args.top_p}: base k3 {base['k3']:.5f} |d| {base['mean_abs']:.4f}, "
        f"updated k3 {after['k3']:.5f} |d| {after['mean_abs']:.4f} ({'pass' if not failures else 'FAIL'})"
    )
    return summary, line


def run_kl_decompose(args) -> tuple[dict, str]:
    """At top-p 1.0, splits the sampler/trainer gap into engine noise, decode vs prefill, and engine vs trainer."""
    prompts = kl_contexts(load_workload(args.workload), args.contexts, args.min_prompt, args.max_prompt)
    training = _training_client(args)
    sampler = training.save_weights_and_get_sampling_client("decompose")
    samples = _sample(sampler, prompts, types.SamplingParams(max_tokens=args.max_tokens, temperature=1.0))
    series = {"dec": [], "pre1": [], "pre2": [], "tr": []}
    for prompt, sequence in samples:
        full = types.ModelInput.from_ints(prompt + list(sequence.tokens))
        series["dec"] += list(sequence.logprobs)
        series["pre1"] += sampler.compute_logprobs(full).result()[len(prompt) :]
        series["pre2"] += sampler.compute_logprobs(full).result()[len(prompt) :]
    result = training.forward(_datums(samples, [0.0] * len(samples)), "importance_sampling").result()
    for (prompt, _), output in zip(samples, result.loss_fn_outputs, strict=True):
        series["tr"] += trainer_logprobs(output, len(prompt))
    pairs = {f"{reference}->{target}": gap(series[target], series[reference]) for reference, target in KL_PAIRS}
    k3 = pairs["dec->tr"]["k3"]
    failures = [f"dec->tr k3 {k3:.5f} > {args.max_k3}"] if k3 > args.max_k3 else []
    summary = _with_bar(
        {
            "turns": len(samples),
            "tokens": len(series["dec"]),
            "entropy_proxy": -statistics.fmean(series["dec"]),
            "pairs": pairs,
        },
        failures,
    )
    line = ", ".join(f"{name} k3 {stats['k3']:.5f}" for name, stats in pairs.items())
    return summary, f"{len(samples)} turns: {line} ({'pass' if not failures else 'FAIL'})"


def run_parity_by_length(args) -> tuple[dict, str]:
    buckets = length_buckets(load_workload(args.workload), args.contexts)
    training = _training_client(args)
    sampler = training.save_weights_and_get_sampling_client("parity")
    params = types.SamplingParams(max_tokens=args.max_tokens, temperature=1.0, top_p=args.top_p, top_k=args.top_k)
    rows = []
    for label, contexts in buckets:
        samples = _sample(sampler, contexts, params)
        result = training.forward(_datums(samples, [0.0] * len(samples)), "importance_sampling").result()
        rows.append({"bucket": label, "contexts": len(contexts), **_trainer_gap(samples, result)})
    line = ", ".join(f"{row['bucket']} k3 {row['k3']:.5f}" for row in rows)
    return {"buckets": rows}, f"parity by prompt length: {line}"


def run_long_train(args) -> tuple[dict, str]:
    """Samples after the longest contexts, then runs forward and forward_backward on those datums."""
    workload = load_workload(args.workload)
    if args.context_tokens:
        contexts = [long_context(workload, args.context_tokens)]
    else:
        contexts = longest_contexts(workload, args.contexts)
    training = _training_client(args)
    sampler = training.save_weights_and_get_sampling_client("repro")
    params = types.SamplingParams(max_tokens=args.max_tokens, temperature=1.0, top_p=args.top_p, top_k=args.top_k)
    seconds = {}
    samples = _timed(seconds, "sample", lambda: _sample(sampler, contexts, params))
    datums = _datums(samples, [1.0] * len(samples))
    _timed(seconds, "forward", lambda: training.forward(datums, "importance_sampling").result())
    result = _timed(
        seconds, "forward_backward", lambda: training.forward_backward(datums, "importance_sampling").result()
    )
    summary = {
        "datum_lengths": [datum.model_input.length for datum in datums],
        "seconds": seconds,
        "forward_backward_metrics": result.metrics,
    }
    line = f"{len(datums)} datums up to {max(summary['datum_lengths'])} tokens: " + ", ".join(
        f"{name} {value:.0f}s" for name, value in seconds.items()
    )
    return summary, line


async def _replay_trajectory(sampler, prefix, turns, delay, rng, latencies, semaphore) -> int:
    generated = 0
    async with semaphore:
        for index, turn in enumerate(turns):
            if index:
                await asyncio.sleep(delay * rng.uniform(0.5, 1.5))
            start = time.time()
            response = await sampler.sample_async(
                prompt=types.ModelInput.from_ints(prefix + turn["prompt"]),
                num_samples=1,
                sampling_params=types.SamplingParams(max_tokens=turn["output_len"], temperature=1.0, stop=[]),
            )
            latencies.append(time.time() - start)
            generated += len(response.sequences[0].tokens)
    return generated


async def _replay(args) -> dict:
    rng = random.Random(0)
    jobs = replay_jobs(load_workload(args.workload), args.copies, rng)
    concurrency = args.concurrency or len(jobs)
    training = await _service(args).create_lora_training_client_async(args.model, rank=args.rank)
    sampler = await training.save_weights_and_get_sampling_client_async("replay")
    log_offset = args.engine_log.stat().st_size if args.engine_log else 0
    latencies: list[float] = []
    semaphore = asyncio.Semaphore(concurrency)
    start = time.time()
    generated = sum(
        await asyncio.gather(
            *(
                _replay_trajectory(sampler, prefix, turns, args.env_delay, rng, latencies, semaphore)
                for prefix, turns in jobs
            )
        )
    )
    wall = time.time() - start
    summary = {
        "trajectories": len(jobs),
        "concurrency": concurrency,
        "env_delay": args.env_delay,
        "wall_seconds": wall,
        "trajectories_per_minute": len(jobs) / wall * 60,
        "generated_tokens": generated,
        "generated_tokens_per_second": generated / wall,
        "prompt_tokens": sum(len(prefix) + len(turn["prompt"]) for prefix, turns in jobs for turn in turns),
        "turns": len(latencies),
        "mean_turn_latency": statistics.fmean(latencies),
    }
    if args.engine_log:
        with args.engine_log.open("rb") as log:
            log.seek(log_offset)
            summary["engine"] = engine_stats(log.read().decode(errors="replace").splitlines())
    return summary


def run_replay(args) -> tuple[dict, str]:
    """Turns of a trajectory run in order, each generating its recorded output length."""
    summary = asyncio.run(_replay(args))
    line = (
        f"{summary['trajectories']} trajectories x conc {summary['concurrency']}, delay {args.env_delay:.0f}s: "
        f"wall {summary['wall_seconds']:.0f}s, {summary['trajectories_per_minute']:.1f} trajectories/min, "
        f"{summary['generated_tokens_per_second']:.0f} generated tok/s, "
        f"mean turn latency {summary['mean_turn_latency']:.2f}s"
    )
    if "engine" in summary and summary["engine"]["cache_hit_rate"] is not None:
        line += f", cache hit {summary['engine']['cache_hit_rate']:.2f}"
    return summary, line


def run_engine_stats(args) -> tuple[dict, str]:
    if args.logs:
        lines = [line for path in args.logs for line in path.read_text(errors="replace").splitlines()]
    else:
        lines = sys.stdin
    stats = engine_stats(lines)
    hit = "none" if stats["cache_hit_rate"] is None else f"{stats['cache_hit_rate']:.3f}"
    gen = "none" if stats["mean_gen_tokens_per_second"] is None else f"{stats['mean_gen_tokens_per_second']:.0f}"
    line = (
        f"prefill {stats['new_tokens']} new + {stats['cached_tokens']} cached tokens (hit {hit}), mean gen {gen} tok/s"
    )
    return stats, line


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    gates = parser.add_subparsers(dest="gate", required=True)

    def client_gate(name: str, run, help: str, workload: bool = False) -> argparse.ArgumentParser:
        gate = gates.add_parser(name, help=help, description=run.__doc__)
        gate.set_defaults(run=run)
        gate.add_argument("--base-url", required=True, help="Tinker gateway URL")
        gate.add_argument("--model", default=MODEL)
        gate.add_argument("--rank", type=int, default=32)
        if workload:
            gate.add_argument("--workload", type=Path, required=True, help="replay workload (.json or .json.gz)")
        return gate

    def sampling(gate: argparse.ArgumentParser, max_tokens: int) -> None:
        gate.add_argument("--max-tokens", type=int, default=max_tokens)
        gate.add_argument("--top-p", type=float, default=0.97)
        gate.add_argument("--top-k", type=int, default=1024)

    parity = client_gate("parity", run_parity, "sampler/trainer gap before and after one LoRA update")
    sampling(parity, max_tokens=1024)
    parity.add_argument("--tokenizer", default=MODEL, help="tokenizer name or path for the chat prompts")
    parity.add_argument("--revision", default=REVISION)
    parity.add_argument("--lr", type=float, default=1e-3)
    parity.add_argument("--max-k3", type=float, default=0.001)

    decompose = client_gate(
        "kl-decompose", run_kl_decompose, "decode, prefill and trainer log-probs at top-p 1.0", True
    )
    decompose.add_argument("--contexts", type=int, default=24)
    decompose.add_argument("--min-prompt", type=int, default=4_000)
    decompose.add_argument("--max-prompt", type=int, default=32_000)
    decompose.add_argument("--max-tokens", type=int, default=256)
    decompose.add_argument("--max-k3", type=float, default=0.0015, help="bar on the dec->tr k3")

    by_length = client_gate("parity-by-length", run_parity_by_length, "sampler/trainer gap per prompt length", True)
    sampling(by_length, max_tokens=256)
    by_length.add_argument("--contexts", type=int, default=6, help="contexts per length bucket")

    long_train = client_gate(
        "long-train", run_long_train, "forward and forward_backward on the longest contexts", True
    )
    sampling(long_train, max_tokens=256)
    long_train.add_argument("--contexts", type=int, default=5)
    long_train.add_argument(
        "--context-tokens", type=int, default=0, help="instead, one context of this many tokens cut from the workload"
    )

    replay = client_gate("replay", run_replay, "replay sampling throughput", True)
    replay.add_argument("--copies", type=int, default=1, help="copies of each trajectory")
    replay.add_argument("--concurrency", type=int, default=0, help="active trajectories; 0 runs all at once")
    replay.add_argument("--env-delay", type=float, default=0.0, help="mean seconds of environment time between turns")
    replay.add_argument("--engine-log", type=Path, default=None, help="service log to read engine stats from")

    stats = gates.add_parser("engine-stats", help="engine stats from SGLang batch log lines")
    stats.set_defaults(run=run_engine_stats)
    stats.add_argument("logs", type=Path, nargs="*", help="log files; stdin if none")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    summary, line = args.run(args)
    print(json.dumps(summary))
    print(line, file=sys.stderr)
    if summary.get("passed") is False:
        sys.exit(1)


if __name__ == "__main__":
    main()
