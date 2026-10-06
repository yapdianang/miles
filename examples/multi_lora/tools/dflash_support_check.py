"""Probe an SGLang engine (DFLASH, sglang#34201) for per-token sampling supports.

Runs against one SGLang ``/generate`` endpoint with DFLASH speculative decoding:

1. Cases, ``--case-requests`` each, checked token by token:
   (a) T=1, top_p 0.97, top_k 1024, support mode; (b) the same, selected mode; (c) T=0;
   (d) top_k 50, top_p 1, support mode; (e) case (a) with ``--stop-text``'s token as a stop token,
   so generation stops inside a verify block.
2. A sweep of ``--sweep-requests`` requests over cases (a), (b) and (d), counting INVALID and
   OVERFLOW aborts.
3. ``--throughput-requests`` requests of case (a) at ``--concurrency`` with ``ignore_eos``, masks off
   and then on: decode tok/s, and the DFLASH accept length from each response's ``spec_verify_ct``.

Token checks: mask count == output_token_logprobs count == completion tokens; every token is in
its mask; masks have no duplicates and at most top_k ids; support-mode rows sum to one within
``--tolerance``; selected-mode values are at least the full-vocabulary log-probability; greedy rows
are singletons with log-probability 0. Exits non-zero if any check fails or the accept lengths
with and without masks differ by more than ``--accept-z`` standard errors.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

import httpx
from transformers import AutoTokenizer

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
ABORT_KINDS = (
    ("OVERFLOW", "exceeds --sampling-mask-max-tokens"),
    ("INVALID", "outside its captured sampling support"),
    ("INVALID", "did not return captured sampling support"),
)


def cases(args, stop_token: int) -> dict[str, dict]:
    filtered = {"temperature": 1.0, "top_p": args.top_p, "top_k": args.top_k}
    return {
        "a_support": {"params": filtered, "mode": "support", "top_k": args.top_k},
        "b_selected": {"params": filtered, "mode": "selected", "top_k": args.top_k},
        "c_greedy": {"params": {"temperature": 0.0}, "mode": "selected", "top_k": 1, "greedy": True},
        "d_top_k": {"params": {"temperature": 1.0, "top_p": 1.0, "top_k": 50}, "mode": "support", "top_k": 50},
        "e_stop": {
            "params": filtered | {"stop_token_ids": [stop_token]},
            "mode": "support",
            "top_k": args.top_k,
            "stop_token": stop_token,
        },
    }


def _logsumexp(values: list[float]) -> float:
    peak = max(values)
    return peak + math.log(sum(math.exp(value - peak) for value in values))


def abort_kind(message: str) -> str:
    return next((kind for kind, text in ABORT_KINDS if text in message), "OTHER")


def check_response(meta: dict, case: dict, tolerance: float) -> list[str]:
    """Token-level checks of one finished /generate response with return_sampling_mask."""
    logprobs = meta["output_token_logprobs"]
    masks = meta.get("output_token_sampling_mask")
    values = meta.get("output_token_sampling_logprobs")
    if masks is None or values is None:
        return ["no sampling supports in the response"]
    if not len(masks) == len(values) == len(logprobs) == meta["completion_tokens"]:
        return [
            f"{len(masks)} masks, {len(values)} sampling log-probs, {len(logprobs)} output log-probs, "
            f"{meta['completion_tokens']} completion tokens"
        ]
    failures = []
    for position, ((full, token, *_), mask, value) in enumerate(zip(logprobs, masks, values, strict=True)):
        where = f"token {position} ({token})"
        if token not in mask:
            failures.append(f"{where} is not in its mask of {len(mask)}")
            continue
        if len(set(mask)) != len(mask) or len(mask) > case["top_k"]:
            failures.append(f"{where}: mask of {len(mask)} ids, {len(set(mask))} distinct, top_k {case['top_k']}")
        if case["mode"] == "support":
            if len(value) != len(mask) or abs(_logsumexp([float(entry) for entry in value])) > tolerance:
                failures.append(f"{where}: support log-probs do not sum to one over the mask")
                continue
            selected = float(value[mask.index(token)])
        else:
            selected = float(value)
        if case.get("greedy"):
            if mask != [token] or selected != 0.0:
                failures.append(f"{where}: greedy mask {mask[:4]} with log-prob {selected}")
        elif not math.isfinite(selected) or selected > 1e-6 or selected < float(full) - 1e-4:
            failures.append(f"{where}: renormalized log-prob {selected} vs full-vocabulary {full}")
    if "stop_token" in case and meta["finish_reason"].get("matched") == case["stop_token"]:
        if logprobs[-1][1] != case["stop_token"]:
            failures.append("stopped on the stop token, which is not the last output token")
    return failures


async def generate(client, url: str, prompt: list[int], case: dict, args, masks: bool = True, **extra) -> dict:
    request = {
        "input_ids": prompt,
        "sampling_params": {"max_new_tokens": args.max_tokens} | case["params"] | extra,
        "return_logprob": True,
    }
    if masks:
        request["return_sampling_mask"] = True
        if case["mode"] == "support":
            request["sampling_logprobs_mode"] = "support"
    response = await client.post(f"{url}/generate", json=request)
    if response.status_code != 200:
        return {"error": response.text[:500]}
    return response.json()


async def run_cases(client, args, prompts: list[list[int]], requests: list[tuple[str, dict]]) -> dict:
    """Send (case name, case) requests over the prompts at the configured concurrency and check each."""
    semaphore = asyncio.Semaphore(args.concurrency)
    report = {"requests": Counter(), "aborts": Counter(), "failures": [], "stopped_on_stop_token": 0}

    async def one(index: int, name: str, case: dict) -> None:
        async with semaphore:
            result = await generate(client, args.url, prompts[index % len(prompts)], case, args)
        report["requests"][name] += 1
        meta = result.get("meta_info")
        if meta is None or meta["finish_reason"].get("type") == "abort":
            message = str(result.get("error") if meta is None else meta["finish_reason"].get("message", ""))
            report["aborts"][abort_kind(message)] += 1
            report["failures"].append(f"{name} #{index}: aborted: {message[:200]}")
            return
        report["stopped_on_stop_token"] += meta["finish_reason"].get("matched") == case.get("stop_token", object())
        report["failures"].extend(
            f"{name} #{index}: {failure}" for failure in check_response(meta, case, args.tolerance)[:3]
        )

    await asyncio.gather(*[one(index, name, case) for index, (name, case) in enumerate(requests)])
    return report


async def run_throughput(client, args, prompts: list[list[int]], case: dict, masks: bool, requests: int) -> dict:
    semaphore = asyncio.Semaphore(args.concurrency)
    accept_lengths, tokens, verify_steps = [], 0, 0

    async def one(index: int) -> None:
        nonlocal tokens, verify_steps
        async with semaphore:
            result = await generate(
                client, args.url, prompts[index % len(prompts)], case, args, masks=masks, ignore_eos=True
            )
        meta = result.get("meta_info") or {}
        tokens += meta.get("completion_tokens", 0)
        if meta.get("spec_verify_ct"):
            verify_steps += meta["spec_verify_ct"]
            accept_lengths.append(meta["completion_tokens"] / meta["spec_verify_ct"])

    start = time.perf_counter()
    await asyncio.gather(*[one(index) for index in range(requests)])
    elapsed = time.perf_counter() - start
    return {
        "masks": masks,
        "completion_tokens": tokens,
        "seconds": elapsed,
        "decode_tokens_per_second": tokens / elapsed,
        "accept_length": tokens / verify_steps if verify_steps else None,
        "per_request_accept_lengths": accept_lengths,
    }


def compare_accept_lengths(off: list[float], on: list[float]) -> dict:
    if len(off) < 2 or len(on) < 2:
        return {"z": None}
    error = math.sqrt(statistics.variance(off) / len(off) + statistics.variance(on) / len(on))
    difference = statistics.fmean(on) - statistics.fmean(off)
    z = difference / error if error else (0.0 if difference == 0 else math.copysign(math.inf, difference))
    return {"mean_off": statistics.fmean(off), "mean_on": statistics.fmean(on), "z": z}


async def run(args) -> dict:
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, local_files_only=True, trust_remote_code=True)
    prompts = [
        # Render, then encode: some transformers return a BatchEncoding from tokenize=True.
        tokenizer.encode(
            tokenizer.apply_chat_template(
                [{"role": "user", "content": text}], add_generation_prompt=True, tokenize=False
            ),
            add_special_tokens=False,
        )
        for text in PROMPTS
    ]
    stop_ids = tokenizer.encode(args.stop_text, add_special_tokens=False)
    if len(stop_ids) != 1:
        raise SystemExit(f"--stop-text {args.stop_text!r} is {len(stop_ids)} tokens; pick a single-token text")
    probe_cases = cases(args, stop_ids[0])
    async with httpx.AsyncClient(timeout=args.timeout_seconds) as client:
        try:
            info = (await client.get(f"{args.url}/server_info")).json()
        except (httpx.HTTPError, ValueError):
            info = {}  # not an engine endpoint; the DFLASH check below fails
        report = {
            "speculative_algorithm": info.get("speculative_algorithm"),
            "speculative_num_draft_tokens": info.get("speculative_num_draft_tokens"),
            "sampling_mask_max_tokens": info.get("sampling_mask_max_tokens"),
        }
        report["cases"] = await run_cases(
            client,
            args,
            prompts,
            [(name, case) for name, case in probe_cases.items() for _ in range(args.case_requests)],
        )
        sweep = [("a_support", "b_selected", "d_top_k")[index % 3] for index in range(args.sweep_requests)]
        report["sweep"] = await run_cases(client, args, prompts, [(name, probe_cases[name]) for name in sweep])
        throughput_case = probe_cases["a_support"]
        await run_throughput(client, args, prompts, throughput_case, masks=False, requests=args.concurrency)  # warm up
        off = await run_throughput(
            client, args, prompts, throughput_case, masks=False, requests=args.throughput_requests
        )
        on = await run_throughput(
            client, args, prompts, throughput_case, masks=True, requests=args.throughput_requests
        )
    report["throughput"] = {
        "off": {key: value for key, value in off.items() if key != "per_request_accept_lengths"},
        "on": {key: value for key, value in on.items() if key != "per_request_accept_lengths"},
        "masks_on_over_off": on["decode_tokens_per_second"] / off["decode_tokens_per_second"],
        "accept_length": compare_accept_lengths(off["per_request_accept_lengths"], on["per_request_accept_lengths"]),
    }
    return report


def failures_of(report: dict, args) -> list[str]:
    failures = []
    if report["speculative_algorithm"] != "DFLASH":
        failures.append(f"engine runs {report['speculative_algorithm']}, not DFLASH")
    for phase in ("cases", "sweep"):
        failures += [f"{phase}: {failure}" for failure in report[phase]["failures"]]
    if report["cases"]["stopped_on_stop_token"] == 0:
        failures.append(f"no case (e) request stopped on {args.stop_text!r}; pick a more frequent --stop-text")
    z = report["throughput"]["accept_length"]["z"]
    if z is None or abs(z) > args.accept_z:
        failures.append(f"accept length with masks differs from without (z={z})")
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", required=True, help="SGLang engine or router base URL serving /generate")
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--top-p", type=float, default=0.97)
    parser.add_argument("--top-k", type=int, default=1024)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--stop-text", default=" the", help="single-token text whose token is case (e)'s stop token")
    parser.add_argument("--case-requests", type=int, default=16)
    parser.add_argument("--sweep-requests", type=int, default=1024)
    parser.add_argument("--throughput-requests", type=int, default=128)
    parser.add_argument("--concurrency", type=int, default=64)
    parser.add_argument("--tolerance", type=float, default=1e-5)
    parser.add_argument("--accept-z", type=float, default=3.0)
    parser.add_argument("--timeout-seconds", type=float, default=3600.0)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    report = asyncio.run(run(args))
    failures = failures_of(report, args)
    report["passed"] = not failures
    report["failures"] = failures[:200]
    for phase in ("cases", "sweep"):
        report[phase]["failures"] = report[phase]["failures"][:50]
    text = json.dumps(report, indent=2)
    print(text)
    if args.output is not None:
        args.output.write_text(text + "\n")
    if failures:
        print(f"FAILED: {len(failures)} problems; first: {failures[0]}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
