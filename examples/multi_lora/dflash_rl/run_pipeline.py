"""Run the DFlash RL drafter pipeline on one 8-GPU node, parity gate first, and write ``<work>/summary.json``.

1. Gate: extract the held-out hidden states, then measure the shipped drafter's accept length offline (blocks 8 and
   6) and on the engine over the same held-out replay (BF16 draft, blocks 8 and 6). Stop if the block-8 offline
   walk accept length and the engine's accept length after the first token differ by more than
   ``--gate-tolerance`` (relative): drafter.py would not reproduce SGLang's draft forward.
2. Extract the training hidden states, fine-tune for ``--epochs``, and evaluate the result offline.
3. Bench the remaining engine configurations: RL drafter x block 6/8 x BF16/FP8, shipped x block 6/8 x FP8.

Each stage is a subprocess logging to ``<work>/logs``; a stage whose output exists is skipped, so a rerun resumes.
``--rollouts`` holds export_rollouts.py's train.jsonl, heldout.jsonl and summary.json.

python -m examples.multi_lora.dflash_rl.run_pipeline --rollouts /mnt/shared-volume/dflash-rl/rollouts \\
    --work /mnt/shared-volume/dflash-rl
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from examples.multi_lora import serve_mimo_v26_flash_tinker as launcher
from examples.multi_lora.dflash_rl.engine import MILES_ROOT

_BLOCK_SIZES = (8, 6)


def _module(name: str, *launcher_args: str, **flags) -> list[str]:
    """``python -m`` argv of a pipeline module, with keyword flags as ``--flag value...``."""
    argv = [sys.executable, *launcher_args, "-m", f"examples.multi_lora.dflash_rl.{name}"]
    for flag, value in flags.items():
        argv += [f"--{flag.replace('_', '-')}", *map(str, value if isinstance(value, list | tuple) else [value])]
    return argv


def _rows(path: Path) -> list[dict]:
    return json.loads(path.read_text()) if path.exists() else []


def _train_logs(work: Path) -> list[dict]:
    log = work / "logs" / "train.log"
    lines = log.read_text().splitlines() if log.exists() else []
    return [json.loads(line) for line in lines if line.startswith('{"epoch"')]


class _Stages:
    def __init__(self, args) -> None:
        self.args = args
        self.work = args.work
        self.shipped = args.drafter or args.hf_checkpoint / "dflash"
        self.rl = args.work / "drafter"
        self.elapsed: dict[str, int] = {}

    def run(self, name: str, command: list[str]) -> None:
        start = time.time()
        print(f"{time.strftime('%H:%M:%S')} {name}", flush=True)
        with open(self.work / "logs" / f"{name}.log", "a") as log:
            result = subprocess.run(command, cwd=MILES_ROOT, stdout=log, stderr=subprocess.STDOUT)
        self.elapsed[name] = round(time.time() - start)
        if result.returncode:
            raise RuntimeError(f"stage {name} failed with code {result.returncode}; see {self.work}/logs/{name}.log")

    def extract(self, split: str) -> None:
        hidden = self.work / "hidden" / split
        rollouts = self.args.rollouts / f"{split}.jsonl"
        flags = dict(hf_checkpoint=self.args.hf_checkpoint, drafter=self.shipped, rollouts=rollouts, out=hidden)
        self.run(f"extract_{split}", _module("extract_hidden", **flags, capture_dir=self.work / "capture"))

    def evaluate(self, name: str, drafter: Path) -> None:
        output = self.work / f"eval_{name}.json"
        if not output.exists():
            flags = dict(target_checkpoint=self.args.hf_checkpoint, data=self.work / "hidden" / "heldout")
            flags |= dict(drafter=f"{name}={drafter}", block_sizes=_BLOCK_SIZES, output=output)
            self.run(f"eval_{name}", _module("eval_offline", **flags))

    def bench(self, name: str, drafter: str, precisions: list[str]) -> None:
        out = self.work / f"bench_{name}"
        if len(_rows(out / "bench.json")) < len(_BLOCK_SIZES) * len(precisions):
            flags = dict(hf_checkpoint=self.args.hf_checkpoint, rollouts=self.args.rollouts / "heldout.jsonl")
            flags |= dict(out=out, drafter=drafter, block_sizes=_BLOCK_SIZES, draft_precisions=precisions)
            self.run(f"bench_{name}", _module("bench_engine", **flags))

    def train(self) -> None:
        if not any(log["step"] == log["of"] for log in _train_logs(self.work)):
            flags = dict(target_checkpoint=self.args.hf_checkpoint, drafter=self.shipped, out=self.rl)
            flags |= dict(data=self.work / "hidden" / "train", epochs=self.args.epochs)
            torchrun = ("-m", "torch.distributed.run", "--nproc-per-node", "8")
            self.run("train", _module("train_drafter", *torchrun, **flags))


def _gate(work: Path, tolerance: float) -> dict:
    offline = next(row for row in _rows(work / "eval_shipped.json") if row["block_size"] == 8)
    engine = next(row for row in _rows(work / "bench_gate" / "bench.json") if row["block_size"] == 8)
    gap = offline["walk_accept_length"] / engine["accept_length_after_first"] - 1
    return {
        "offline_walk_accept_length": offline["walk_accept_length"],
        "engine_accept_length_after_first": engine["accept_length_after_first"],
        "relative_gap": gap,
        "passed": abs(gap) <= tolerance,
    }


def _comparisons(offline: list[dict], engine: list[dict]) -> dict:
    walk = {(row["drafter"], row["block_size"]): row["walk_accept_length"] for row in offline}
    accept = {(row["drafter"], row["block_size"], row["draft_precision"]): row["accept_length"] for row in engine}
    speed = {(row["drafter"], row["block_size"], row["draft_precision"]): row["output_tok_per_s"] for row in engine}

    def gain(table: dict, new: tuple, old: tuple) -> float | None:
        return table[new] / table[old] - 1 if new in table and old in table else None

    drafters, precisions = ("shipped", "rl"), ("bf16", "fp8")
    return {
        "offline_walk_accept_rl_vs_shipped": {b: gain(walk, ("rl", b), ("shipped", b)) for b in _BLOCK_SIZES},
        "engine_accept_rl_vs_shipped": {
            f"block{b}_{p}": gain(accept, ("rl", b, p), ("shipped", b, p)) for b in _BLOCK_SIZES for p in precisions
        },
        "engine_tok_per_s_block6_vs_block8": {
            f"{d}_{p}": gain(speed, (d, 6, p), (d, 8, p)) for d in drafters for p in precisions
        },
        "engine_tok_per_s_fp8_vs_bf16": {
            f"{d}_block{b}": gain(speed, (d, b, "fp8"), (d, b, "bf16")) for d in drafters for b in _BLOCK_SIZES
        },
        "engine_tok_per_s_rl_fp8_block6_vs_shipped_bf16_block8": gain(speed, ("rl", 6, "fp8"), ("shipped", 8, "bf16")),
    }


def _write_summary(stages: _Stages, gate: dict) -> dict:
    work = stages.work
    offline = _rows(work / "eval_shipped.json") + _rows(work / "eval_rl.json")
    engine = [row for name in ("gate", "rl", "shipped_fp8") for row in _rows(work / f"bench_{name}" / "bench.json")]
    rollouts = json.loads((stages.args.rollouts / "summary.json").read_text())
    train_logs = _train_logs(work)
    summary = {
        "rollouts": {key: value for key, value in rollouts.items() if not key.endswith("task_mix")},
        "gate": gate,
        "train": {"first": train_logs[0], "last": train_logs[-1]} if train_logs else None,
        "offline": offline,
        "engine": engine,
        "comparisons": _comparisons(offline, engine),
        "elapsed_s": stages.elapsed,
    }
    (work / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--hf-checkpoint", type=Path, default=Path(launcher.ScriptArgs.hf_checkpoint))
    parser.add_argument("--drafter", type=Path, default=None, help="default: <hf-checkpoint>/dflash")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--gate-tolerance", type=float, default=0.05)
    args = parser.parse_args(argv)
    (args.work / "logs").mkdir(parents=True, exist_ok=True)
    stages = _Stages(args)

    stages.extract("heldout")
    stages.evaluate("shipped", stages.shipped)
    stages.bench("gate", f"shipped={stages.shipped}", ["bf16"])
    gate = _gate(args.work, args.gate_tolerance)
    print(json.dumps({"gate": gate}), flush=True)
    if not gate["passed"]:
        _write_summary(stages, gate)
        sys.exit(f"parity gate failed: {json.dumps(gate)}")

    stages.extract("train")
    stages.train()
    stages.evaluate("rl", stages.rl)
    stages.bench("rl", f"rl={stages.rl}", ["bf16", "fp8"])
    stages.bench("shipped_fp8", f"shipped={stages.shipped}", ["fp8"])
    summary = _write_summary(stages, gate)
    print(json.dumps(summary["comparisons"], indent=2))
    print(f"summary: {args.work / 'summary.json'}")


if __name__ == "__main__":
    main()
