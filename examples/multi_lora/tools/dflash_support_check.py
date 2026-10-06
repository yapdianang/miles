"""GPU check: DFlash-committed tokens come with the top-k/top-p set they were sampled from.

Run inside the Tinker service pod, with the engine on DFlash and the gateway on
``--tinker-sampling-support-replay``:

    python dflash_support_check.py --base-url http://127.0.0.1:10613 --router-url http://<router> \\
        --tokenizer-path "$MIMO_CHECKPOINT" --engine-log /path/to/engine.log

Engine probe (``--router-url``): sends the gateway's request (``return_sampling_mask``,
``sampling_logprobs_mode=support``) plus ``top_logprobs_num``. Per output token it checks that the
set contains the token, has no duplicates and at most ``top_k`` ids, that the set's log-probabilities
sum to one, and, when every set member is among the top log-probabilities, that the renormalized
log-probability equals ``log p_y - logsumexp_{v in set} log p_v``.

Gateway probe (``--base-url``): samples through the Tinker API, then scores the same tokens with
the trainer (``forward``, which renormalizes within the gateway's recorded sets) and with
teacher-forced full-vocabulary prompt log-probabilities. Sampler and trainer log-probabilities
must be finite and non-positive; the renormalization gap (sampler minus full vocabulary) is
reported, as is the DFlash accept length from ``/server_info`` and the engine log.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import re
import statistics
import sys
from pathlib import Path

import httpx
from transformers import AutoTokenizer

import tinker
from tinker import types

PROMPTS = (
    "Write a Python function that merges overlapping intervals, then explain its complexity.",
    "A train leaves at 3:40 pm and travels 210 km at 84 km/h. When does it arrive? Show your steps.",
    "Summarize the causes of the 1929 stock market crash in five bullet points.",
    "Draft a polite email asking a landlord to fix a leaking kitchen faucet.",
    "List three ways to reduce memory usage in a PyTorch training loop and why they work.",
    "Explain the difference between TCP and UDP to a high-school student.",
    "Write a haiku about debugging at midnight, then a limerick about the same thing.",
    "Given the SQL table orders(id, customer, total, created_at), write a query for monthly revenue.",
)
ACCEPT_LEN = re.compile(r"accept len: ([0-9.]+)")


def _logsumexp(values: list[float]) -> float:
    peak = max(values)
    return peak + math.log(sum(math.exp(value - peak) for value in values))


def _summary(values: list[float]) -> dict:
    if not values:
        return {"count": 0}
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "mean": statistics.fmean(ordered),
        "p50": ordered[len(ordered) // 2],
        "p99": ordered[min(len(ordered) - 1, round(0.99 * (len(ordered) - 1)))],
        "max": ordered[-1],
        "min": ordered[0],
    }


def _prompts(tokenizer, count: int) -> list[list[int]]:
    return [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": PROMPTS[index % len(PROMPTS)]}], add_generation_prompt=True, tokenize=True
        )
        for index in range(count)
    ]


def check_engine_response(meta: dict, top_k: int, tolerance: float) -> tuple[list[str], dict]:
    """Per-token checks of one SGLang /generate response in support mode."""
    failures = []
    tokens = [int(entry[1]) for entry in meta["output_token_logprobs"]]
    masks = meta.get("output_token_sampling_mask")
    rows = meta.get("output_token_sampling_logprobs")
    tops = meta.get("output_top_logprobs") or []
    if masks is None or rows is None:
        return ["the engine returned no sampling supports"], {}
    if not len(masks) == len(rows) == len(tokens):
        return [f"{len(masks)} supports and {len(rows)} log-prob rows for {len(tokens)} tokens"], {}
    sizes, mass_errors, renorm_errors, reconstructed = [], [], [], 0
    for position, (token, mask, row) in enumerate(zip(tokens, masks, rows, strict=True)):
        where = f"token {position} ({token})"
        if token not in mask:
            failures.append(f"{where} is not in its set of {len(mask)}")
            continue
        if len(set(mask)) != len(mask) or len(mask) > top_k or len(row) != len(mask):
            failures.append(f"{where}: set of {len(mask)} ids ({len(set(mask))} distinct, {len(row)} log-probs)")
            continue
        sizes.append(len(mask))
        mass_errors.append(abs(_logsumexp([float(value) for value in row])))
        renormalized = float(row[mask.index(token)])
        if position < len(tops) and tops[position]:
            full = {int(entry[1]): float(entry[0]) for entry in tops[position]}
            if all(member in full for member in mask):
                reconstructed += 1
                expected = full[token] - _logsumexp([full[member] for member in mask])
                renorm_errors.append(abs(renormalized - expected))
                if abs(renormalized - expected) > tolerance:
                    failures.append(f"{where}: renormalized {renormalized:.6f} != {expected:.6f}")
    if mass_errors and max(mass_errors) > tolerance:
        failures.append(f"set log-probabilities sum to 1 only within {max(mass_errors):.2e}")
    return failures, {
        "tokens": len(tokens),
        "set_sizes": sizes,
        "mass_errors": mass_errors,
        "renorm_errors": renorm_errors,
        "reconstructed": reconstructed,
    }


def engine_probe(args, prompts: list[list[int]]) -> dict:
    failures, sizes, mass, renorm, tokens, reconstructed = [], [], [], [], 0, 0
    with httpx.Client(timeout=args.timeout_seconds) as client:
        for index, prompt in enumerate(prompts):
            request = {
                "input_ids": prompt,
                "sampling_params": {
                    "max_new_tokens": args.max_tokens,
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "top_k": args.top_k,
                },
                "return_logprob": True,
                "return_sampling_mask": True,
                "sampling_logprobs_mode": "support",
                "top_logprobs_num": args.top_logprobs,
            }
            response = client.post(f"{args.router_url}/generate", json=request)
            if response.status_code != 200:
                failures.append(f"prompt {index}: HTTP {response.status_code} {response.text[:300]}")
                continue
            meta = response.json()["meta_info"]
            if meta["finish_reason"]["type"] == "abort":
                failures.append(f"prompt {index}: aborted: {meta['finish_reason'].get('message')}")
                continue
            prompt_failures, stats = check_engine_response(meta, args.top_k, args.tolerance)
            failures.extend(f"prompt {index}: {failure}" for failure in prompt_failures[:5])
            tokens += stats.get("tokens", 0)
            reconstructed += stats.get("reconstructed", 0)
            sizes += stats.get("set_sizes", [])
            mass += stats.get("mass_errors", [])
            renorm += stats.get("renorm_errors", [])
    return {
        "failures": failures,
        "tokens": tokens,
        "reconstructed_tokens": reconstructed,
        "set_size": _summary([float(size) for size in sizes]),
        "set_mass_abs_log_error": _summary(mass),
        "renormalized_logprob_abs_error": _summary(renorm),
    }


async def _result(future, timeout_seconds: float):
    return await asyncio.wait_for(future.result_async(), timeout=timeout_seconds)


async def gateway_probe(args, prompts: list[list[int]]) -> dict:
    service = tinker.ServiceClient(base_url=args.base_url, api_key=args.api_key)
    training = await asyncio.wait_for(
        service.create_lora_training_client_async(
            base_model=args.base_model, rank=args.lora_rank, train_attn=True, train_mlp=False, train_unembed=False
        ),
        timeout=args.timeout_seconds,
    )
    saved = await _result(
        await training.save_weights_for_sampler_async(name="dflash-support-check"), args.timeout_seconds
    )
    sampler = await asyncio.wait_for(service.create_sampling_client_async(model_path=saved.path), args.timeout_seconds)
    params = types.SamplingParams(
        max_tokens=args.max_tokens, temperature=args.temperature, top_p=args.top_p, top_k=args.top_k
    )
    results = await asyncio.wait_for(
        asyncio.gather(
            *[
                sampler.sample_async(prompt=types.ModelInput.from_ints(prompt), num_samples=1, sampling_params=params)
                for prompt in prompts
            ]
        ),
        timeout=args.timeout_seconds,
    )
    datums, sampled = [], []
    for prompt, result in zip(prompts, results, strict=True):
        sequence = result.sequences[0]
        tokens = list(prompt) + [int(token) for token in sequence.tokens]
        weights = [float(position >= len(prompt)) for position in range(1, len(tokens))]
        datums.append(
            types.Datum(
                model_input=types.ModelInput.from_ints(tokens[:-1]),
                loss_fn_inputs={"target_tokens": tokens[1:], "weights": weights},
            )
        )
        sampled.append((tokens, len(prompt), [float(value) for value in sequence.logprobs]))
    forward = await _result(await training.forward_async(datums, loss_fn="cross_entropy"), args.timeout_seconds)
    full_params = types.SamplingParams(max_tokens=1, temperature=0.0)
    full = await asyncio.gather(
        *[
            sampler.sample_async(
                prompt=types.ModelInput.from_ints(tokens),
                num_samples=1,
                sampling_params=full_params,
                include_prompt_logprobs=True,
            )
            for tokens, _, _ in sampled
        ]
    )
    failures, train_gap, renorm_gap, trainer_renorm_gap = [], [], [], []
    for index, ((_, prompt_len, sampler_lp), output, scored) in enumerate(
        zip(sampled, forward.loss_fn_outputs, full, strict=True)
    ):
        trainer_lp = [float(value) for value in output["logprobs"].tolist()][prompt_len - 1 :]
        full_lp = [float(value) for value in scored.prompt_logprobs[prompt_len:]]
        if not len(trainer_lp) == len(full_lp) == len(sampler_lp):
            failures.append(
                f"prompt {index}: {len(sampler_lp)} sampled, {len(trainer_lp)} trained, {len(full_lp)} scored"
            )
            continue
        for position, (sample, train, vocabulary) in enumerate(zip(sampler_lp, trainer_lp, full_lp, strict=True)):
            if not (math.isfinite(sample) and math.isfinite(train)) or sample > 1e-6 or train > 1e-6:
                failures.append(f"prompt {index} token {position}: sampler {sample}, trainer {train}")
                continue
            train_gap.append(abs(sample - train))
            renorm_gap.append(sample - vocabulary)
            trainer_renorm_gap.append(train - vocabulary)
    return {
        "failures": failures[:20],
        "sampler_path": saved.path,
        "sampler_vs_trainer_abs": _summary(train_gap),
        "sampler_minus_full_vocab": _summary(renorm_gap),
        "trainer_minus_full_vocab": _summary(trainer_renorm_gap),
    }


def server_state(args) -> dict:
    """Speculative algorithm and cumulative accept length from each engine's /server_info."""
    urls = [args.router_url]
    try:
        urls = httpx.get(f"{args.router_url}/list_workers", timeout=30).json()["urls"] or urls
    except (httpx.HTTPError, KeyError, ValueError):
        pass
    states = {}
    for url in urls:
        try:
            info = httpx.get(f"{url}/server_info", timeout=30).json()
        except (httpx.HTTPError, ValueError) as error:
            states[url] = {"error": str(error)}
            continue
        accepts = [state.get("avg_spec_accept_length") for state in info.get("internal_states", [])]
        states[url] = {"speculative_algorithm": info.get("speculative_algorithm"), "avg_spec_accept_length": accepts}
    return states


def engine_log_accept_lengths(path: Path | None, offset: int) -> dict:
    if path is None or not path.exists():
        return {"count": 0}
    with path.open(errors="replace") as log:
        log.seek(offset)
        return _summary([float(match) for match in ACCEPT_LEN.findall(log.read())])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default=None, help="Tinker gateway; skip the gateway probe if unset")
    parser.add_argument("--router-url", default=None, help="SGLang router; skip the engine probe if unset")
    parser.add_argument("--api-key", default="tml-local")
    parser.add_argument("--base-model", default="XiaomiMiMo/MiMo-V2.6-Flash-RL")
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--num-prompts", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.97)
    parser.add_argument("--top-k", type=int, default=1024)
    parser.add_argument("--top-logprobs", type=int, default=1024, help="engine probe: enough to cover each set")
    parser.add_argument("--tolerance", type=float, default=1e-4)
    parser.add_argument("--engine-log", type=Path, default=None, help="engine log to read 'accept len' lines from")
    parser.add_argument("--timeout-seconds", type=float, default=1800.0)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, local_files_only=True, trust_remote_code=True)
    prompts = _prompts(tokenizer, args.num_prompts)
    log_offset = args.engine_log.stat().st_size if args.engine_log is not None and args.engine_log.exists() else 0
    report: dict = {"top_p": args.top_p, "top_k": args.top_k, "prompts": len(prompts)}
    if args.router_url is not None:
        report["engines_before"] = server_state(args)
        report["engine_probe"] = engine_probe(args, prompts)
    if args.base_url is not None:
        report["gateway_probe"] = asyncio.run(gateway_probe(args, prompts))
    if args.router_url is not None:
        report["engines_after"] = server_state(args)
    report["engine_log_accept_len"] = engine_log_accept_lengths(args.engine_log, log_offset)

    failures = [
        f"{probe}: {failure}"
        for probe in ("engine_probe", "gateway_probe")
        for failure in report.get(probe, {}).get("failures", [])
    ]
    for url, state in report.get("engines_after", {}).items():
        if state.get("speculative_algorithm") not in (None, "DFLASH"):
            failures.append(f"{url} runs {state['speculative_algorithm']}, not DFLASH")
    report["passed"] = not failures
    print(json.dumps(report, indent=2))
    if args.output is not None:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    if failures:
        print("FAILED:\n" + "\n".join(failures[:40]), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
