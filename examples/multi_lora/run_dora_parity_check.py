"""Train a DoRA adapter a few Muown steps, then compare engine and trainer logprobs on engine samples.

Start the gateway with ``serve_mimo_v26_flash_tinker.py serve --optimizer muown`` (DoRA magnitudes, SGLang
--enable-lora-dora). The k3 estimate of KL(engine || trainer) on sampled tokens must stay at plain LoRA's
level (about 1e-3) while the steps move the trainer by much more.

python examples/multi_lora/run_dora_parity_check.py --base-url http://<head>:10613
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
from pathlib import Path

import tinker
from tinker import types

PROMPTS = (
    "Explain why the sky appears blue during the day in two sentences.",
    "Write a Python function that returns the n-th Fibonacci number.",
    "List three differences between TCP and UDP.",
    "Summarize the plot of Hamlet in one paragraph.",
)


def _k3(logprobs: list[float], reference: list[float]) -> float:
    """Mean of exp(r) - 1 - r, r = logprobs - reference: KL(reference || logprobs) on tokens drawn from reference."""
    ratios = [a - b for a, b in zip(logprobs, reference, strict=True)]
    return sum(math.expm1(r) - r for r in ratios) / len(ratios)


async def _sample(sampler, tokenizer, args) -> list[tuple[list[int], list[int], list[float]]]:
    """-> (prompt tokens, completion tokens, engine logprobs) per sample."""
    samples = []
    for prompt in PROMPTS:
        prompt_tokens = tokenizer.encode(prompt)
        result = await sampler.sample_async(
            prompt=types.ModelInput.from_ints(prompt_tokens),
            num_samples=args.samples_per_prompt,
            sampling_params=types.SamplingParams(max_tokens=args.completion_tokens, temperature=1.0),
        )
        assert all(seq.logprobs is not None for seq in result.sequences), "the engine must return sampled logprobs"
        samples += [(prompt_tokens, list(seq.tokens), list(seq.logprobs)) for seq in result.sequences]
    return samples


def _datum(prompt_tokens: list[int], completion_tokens: list[int]) -> types.Datum:
    tokens = prompt_tokens + completion_tokens
    weights = [float(position >= len(prompt_tokens)) for position in range(1, len(tokens))]
    return types.Datum(
        model_input=types.ModelInput.from_ints(tokens[:-1]),
        loss_fn_inputs={"target_tokens": tokens[1:], "weights": weights},
    )


async def _trainer_logprobs(training, samples) -> list[float]:
    """The trainer's logprobs of every sampled completion token, in sample order."""
    result = await (
        await training.forward_async([_datum(p, c) for p, c, _ in samples], loss_fn="cross_entropy")
    ).result_async()
    logprobs = []
    for (prompt, completion, _), output in zip(samples, result.loss_fn_outputs, strict=True):
        completion_logprobs = output["logprobs"].tolist()[len(prompt) - 1 :]
        assert len(completion_logprobs) == len(completion)
        logprobs += completion_logprobs
    return logprobs


def _engine_logprobs(samples) -> list[float]:
    return [logprob for _, _, logprobs in samples for logprob in logprobs]


async def _publish(service, training, name: str):
    saved = await (await training.save_weights_for_sampler_async(name=name)).result_async()
    return await service.create_sampling_client_async(model_path=saved.path)


async def run(args: argparse.Namespace) -> dict:
    service = tinker.ServiceClient(base_url=args.base_url, api_key=args.api_key)
    training = await service.create_lora_training_client_async(
        base_model=args.base_model, rank=args.lora_rank, train_attn=True, train_mlp=False, train_unembed=False
    )
    tokenizer = training.get_tokenizer()
    # MiMo-V2.6 section 5.1's Adam settings with a larger lr, so a few steps move the magnitudes measurably
    adam = types.AdamParams(
        learning_rate=args.learning_rate, beta1=0.95, beta2=0.95, eps=1e-8, weight_decay=0.0, grad_clip_norm=1.0
    )

    samples_0 = await _sample(await _publish(service, training, "dora-parity-step-0"), tokenizer, args)
    trainer_0 = await _trainer_logprobs(training, samples_0)
    steps = []
    for step in range(1, args.steps + 1):
        data = [_datum(prompt, completion) for prompt, completion, _ in samples_0]
        fb = await (await training.forward_backward_async(data, loss_fn="cross_entropy")).result_async()
        optim = await (await training.optim_step_async(adam)).result_async()
        steps.append({"step": step, "forward_backward": fb.metrics, "optim": optim.metrics})
        print(f"step {step}: {fb.metrics} {optim.metrics}", flush=True)

    samples = await _sample(await _publish(service, training, f"dora-parity-step-{args.steps}"), tokenizer, args)
    return {
        "tokens": len(_engine_logprobs(samples)),
        "k3_engine_trainer_step_0": _k3(trainer_0, _engine_logprobs(samples_0)),
        "k3_engine_trainer": _k3(await _trainer_logprobs(training, samples), _engine_logprobs(samples)),
        "k3_trainer_movement": _k3(await _trainer_logprobs(training, samples_0), trainer_0),
        "steps": steps,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--base-model", default="XiaomiMiMo/MiMo-V2.6-Flash-RL")
    parser.add_argument("--api-key", default="tml-dora-parity")
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--samples-per-prompt", type=int, default=4)
    parser.add_argument("--completion-tokens", type=int, default=256)
    parser.add_argument("--max-k3", type=float, default=2e-3)
    parser.add_argument("--min-movement-k3", type=float, default=2e-2)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    report = asyncio.run(run(args))
    print(json.dumps({key: value for key, value in report.items() if key != "steps"}, indent=2))
    if args.output is not None:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    assert report["k3_engine_trainer"] <= args.max_k3, f"engine/trainer k3 {report['k3_engine_trainer']:.2e}"
    assert report["k3_trainer_movement"] >= args.min_movement_k3, f"movement k3 {report['k3_trainer_movement']:.2e}"
