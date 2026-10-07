"""Final gate before a live MiMo-V2.6-Flash-RL run, on a dev pod booted from the integrated image.

Restarts the pod's service once per phase with mimo_dev.py, runs each check in the pod, prints one line
per check and writes one JSON summary; exits 1 unless every selected check passes.

  base: launcher defaults (Adam) and --serve-args
    kl-decompose   mimo_gates kl-decompose at top-p 1.0: decode->trainer k3 <= 0.0015
    parity         mimo_gates parity at top-p 0.97 without top_k (bitmap supports): k3 <= 0.001 before and after
    dflash-probe   dflash_support_check.py on the first engine passes; masks-on/off decode tok/s ratio >= 0.95
    long-train     mimo_gates long-train on one 250K-token datum trains (fused loss); its forward_backward
                   metrics include expert_load/*
    replay         mimo_gates replay at concurrency 64: >= 700 generated tok/s
    engine-load    GET /api/v1/engine_load reports the load of every engine
  collect: base plus the engine flags records-check needs, --rollout-record-cache-gb 64 and
  SGLANG_ROLLOUT_RECORDS_ALSO_RETURN=1
    records-check  bench_rollout_records.py --check prints "mismatching datums: none"
  muown: base plus --optimizer muown (DoRA adapters, SGLang --enable-lora-dora)
    muown-parity   run_dora_parity_check.py, 5 steps at lr 1e-4: engine/trainer k3 <= 2e-3 and trainer
                   movement k3 >= 2e-2

Phases run in this order so the service restarts three times. --workload is a replay workload in the pod,
uncompressed for bench_rollout_records.py.

python examples/multi_lora/tools/mimo_final_gate.py mimo-dev --workload /root/replay.json --output final_gate.json
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
GATEWAY_URL = "http://localhost:10613"
SERVICE_LOG = "/tmp/mimo-dev/service.log"
PHASE_ARGS = {
    "base": [],
    "collect": ["--rollout-record-cache-gb", "64", "--extra-env-vars", "SGLANG_ROLLOUT_RECORDS_ALSO_RETURN=1"],
    "muown": ["--optimizer", "muown"],
}
KL_MAX_K3 = 0.0015
PARITY_MAX_K3 = 0.001
MIN_MASKS_ON_OVER_OFF = 0.95
LONG_DATUM_TOKENS = 250_000
REPLAY_CONCURRENCY = 64
MIN_REPLAY_TOKENS_PER_SECOND = 700.0
DORA_STEPS = 5
DORA_LEARNING_RATE = 1e-4
DORA_MAX_K3 = 2e-3
DORA_MIN_MOVEMENT_K3 = 2e-2
ENGINE_LOAD_FIELDS = ("rank", "url", "running_requests", "queued_requests", "max_running_requests", "full_kv")

_ROUTER_READY = re.compile(r"Router ready at host='([^']+)' port=(\d+)")


@dataclass(frozen=True)
class Check:
    name: str
    phase: str
    bar: str
    # measure(args) -> result runs in the pod; failures(result) applies the bar.
    measure: Callable[[argparse.Namespace], dict]
    failures: Callable[[dict], list[str]]


def _run(argv: list[str], timeout: float | None = None, capture: bool = True) -> tuple[int, str]:
    """Run a command, passing its stderr (and stdout unless captured) through; (exit code, stdout)."""
    result = subprocess.run(argv, stdout=subprocess.PIPE if capture else None, text=True, timeout=timeout)
    return result.returncode, result.stdout or ""


def _dev(*argv: str) -> list[str]:
    return [sys.executable, str(TOOLS_DIR / "mimo_dev.py"), *argv]


def _pod_shell(pod: str, *argv: str) -> list[str]:
    return ["kubectl", "exec", pod, "--", *argv]


def router_url(log_lines: str) -> str:
    host, port = _ROUTER_READY.findall(log_lines)[-1]
    return f"http://{host}:{port}"


def trailing_json(stdout: str) -> dict:
    """The indented JSON report a script prints after any progress lines."""
    return json.loads(stdout[stdout.rfind("\n{") + 1 :])


def _gate(args, gate: str, *gate_args: str) -> dict:
    """The JSON summary a mimo_gates gate prints as its last stdout line."""
    _, stdout = _run(_dev("gate", args.pod, gate, *gate_args), args.check_timeout)
    return json.loads(stdout.strip().splitlines()[-1])


def _engine_load(args) -> dict:
    request = ["curl", "-sS", "-H", "X-API-Key: tml-dummy", f"{GATEWAY_URL}/api/v1/engine_load"]
    _, stdout = _run(_pod_shell(args.pod, *request), args.check_timeout)
    return json.loads(stdout)


def _dflash_probe(args) -> dict:
    engine = _engine_load(args)["engines"][0]["url"]
    probe = ["--url", engine, "--tokenizer-path", args.tokenizer_path]
    _, stdout = _run(_dev("run", args.pod, str(TOOLS_DIR / "dflash_support_check.py"), *probe), args.check_timeout)
    return trailing_json(stdout)


def _records_check(args) -> dict:
    _, log = _run(_pod_shell(args.pod, "grep", "-a", "Router ready at", SERVICE_LOG), args.check_timeout)
    router = router_url(log)
    bench = str(TOOLS_DIR.parent / "bench_rollout_records.py")
    check = ["--router-url", router, "--replay", args.workload, "--check", "--concurrency", str(REPLAY_CONCURRENCY)]
    _, stdout = _run(_dev("run", args.pod, bench, *check), args.check_timeout)
    lines = [line for line in stdout.splitlines() if line.startswith("check:")]
    return {"router_url": router, "check_line": lines[-1] if lines else None}


def _dora_parity(args) -> dict:
    check = [
        "--base-url",
        GATEWAY_URL,
        "--steps",
        str(DORA_STEPS),
        "--learning-rate",
        str(DORA_LEARNING_RATE),
        "--max-k3",
        str(DORA_MAX_K3),
        "--min-movement-k3",
        str(DORA_MIN_MOVEMENT_K3),
    ]
    script = str(TOOLS_DIR.parent / "run_dora_parity_check.py")
    _, stdout = _run(_dev("run", args.pod, script, *check), args.check_timeout)
    return trailing_json(stdout)


def _gate_bar(summary: dict) -> list[str]:
    return [] if summary["passed"] else summary["failures"]


def _dflash_bar(report: dict) -> list[str]:
    failures = [] if report["passed"] else report["failures"][:20]
    ratio = report["throughput"]["masks_on_over_off"]
    if ratio < MIN_MASKS_ON_OVER_OFF:
        failures.append(f"masks-on/off decode tok/s ratio {ratio:.3f} < {MIN_MASKS_ON_OVER_OFF}")
    return failures


def _records_bar(result: dict) -> list[str]:
    line = result["check_line"]
    if line is None:
        return ["bench_rollout_records.py --check printed no check line"]
    return [] if line.endswith("mismatching datums: none") else [line]


def _long_train_bar(summary: dict) -> list[str]:
    failures = []
    if max(summary["datum_lengths"]) < LONG_DATUM_TOKENS:
        failures.append(f"longest datum {max(summary['datum_lengths'])} tokens < {LONG_DATUM_TOKENS}")
    if not any(key.startswith("expert_load/") for key in summary["forward_backward_metrics"]):
        failures.append("forward_backward metrics have no expert_load/* key")
    return failures


def _replay_bar(summary: dict) -> list[str]:
    rate = summary["generated_tokens_per_second"]
    return [] if rate >= MIN_REPLAY_TOKENS_PER_SECOND else [f"{rate:.0f} tok/s < {MIN_REPLAY_TOKENS_PER_SECOND:.0f}"]


def _dora_bar(report: dict) -> list[str]:
    failures = []
    if report["k3_engine_trainer"] > DORA_MAX_K3:
        failures.append(f"engine/trainer k3 {report['k3_engine_trainer']:.2e} > {DORA_MAX_K3}")
    if report["k3_trainer_movement"] < DORA_MIN_MOVEMENT_K3:
        failures.append(f"trainer movement k3 {report['k3_trainer_movement']:.2e} < {DORA_MIN_MOVEMENT_K3}")
    return failures


def _engine_load_bar(load: dict) -> list[str]:
    engines = load.get("engines") or []
    if not engines:
        return [f"no engines in {load}"]
    return [
        f"engine {index} lacks {missing}"
        for index, engine in enumerate(engines)
        if (missing := [field for field in ENGINE_LOAD_FIELDS if field not in engine])
    ]


CHECKS = (
    Check(
        "kl-decompose",
        "base",
        f"decode->trainer k3 <= {KL_MAX_K3} at top-p 1.0",
        lambda args: _gate(args, "kl-decompose", "--workload", args.workload, "--max-k3", str(KL_MAX_K3)),
        _gate_bar,
    ),
    Check(
        "parity",
        "base",
        f"k3 <= {PARITY_MAX_K3} before and after an update at top-p 0.97 without top_k",
        lambda args: _gate(args, "parity", "--top-p", "0.97", "--top-k", "-1", "--max-k3", str(PARITY_MAX_K3)),
        _gate_bar,
    ),
    Check(
        "dflash-probe",
        "base",
        f"probe passes and masks-on/off decode tok/s >= {MIN_MASKS_ON_OVER_OFF}",
        _dflash_probe,
        _dflash_bar,
    ),
    Check(
        "records-check",
        "collect",
        'bench_rollout_records.py --check prints "mismatching datums: none"',
        _records_check,
        _records_bar,
    ),
    Check(
        "long-train",
        "base",
        f"a {LONG_DATUM_TOKENS}-token datum trains; forward_backward metrics include expert_load/*",
        lambda args: _gate(
            args, "long-train", "--workload", args.workload, "--context-tokens", str(LONG_DATUM_TOKENS)
        ),
        _long_train_bar,
    ),
    Check(
        "replay",
        "base",
        f">= {MIN_REPLAY_TOKENS_PER_SECOND:.0f} generated tok/s at concurrency {REPLAY_CONCURRENCY}",
        lambda args: _gate(
            args,
            "replay",
            "--workload",
            args.workload,
            "--concurrency",
            str(REPLAY_CONCURRENCY),
            "--engine-log",
            SERVICE_LOG,
        ),
        _replay_bar,
    ),
    Check("engine-load", "base", "/api/v1/engine_load reports every engine's load", _engine_load, _engine_load_bar),
    Check(
        "muown-parity",
        "muown",
        f"{DORA_STEPS} Muown-DoRA steps: engine/trainer k3 <= {DORA_MAX_K3}, trainer movement k3 >= "
        f"{DORA_MIN_MOVEMENT_K3}",
        _dora_parity,
        _dora_bar,
    ),
)


def _outcome(check: Check, result: dict | None, failures: list[str], seconds: float) -> dict:
    return {
        "name": check.name,
        "phase": check.phase,
        "bar": check.bar,
        "passed": not failures,
        "failures": failures,
        "seconds": seconds,
        "result": result,
    }


def run_check(check: Check, args) -> dict:
    start = time.monotonic()
    try:
        result = check.measure(args)
        failures = check.failures(result)
    except Exception as error:  # a crashed, timed-out or unparsable check fails the gate, not the run
        result, failures = None, [f"{type(error).__name__}: {error}"]
    return _outcome(check, result, failures, time.monotonic() - start)


def run_gate(args, checks: list[Check]) -> dict:
    """Run the checks phase by phase; the summary lists them in the order given."""
    outcomes = {}
    for phase, phase_args in PHASE_ARGS.items():
        selected = [check for check in checks if check.phase == phase]
        if not selected:
            continue
        print(f"== {phase}: restarting the service", file=sys.stderr, flush=True)
        started, _ = _run(_dev("restart", args.pod, *shlex.split(args.serve_args), *phase_args), capture=False)
        for check in selected:
            if started == 0:
                print(f"== {check.name}: {check.bar}", file=sys.stderr, flush=True)
                outcomes[check.name] = run_check(check, args)
            else:
                outcomes[check.name] = _outcome(check, None, [f"the {phase} service did not start"], 0.0)
            outcome = outcomes[check.name]
            print(
                f"{'PASS' if outcome['passed'] else 'FAIL'} {check.name}: {'; '.join(outcome['failures'])}", flush=True
            )
    ordered = [outcomes[check.name] for check in checks]
    return {"pod": args.pod, "passed": all(outcome["passed"] for outcome in ordered), "checks": ordered}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pod")
    parser.add_argument("--workload", default="/root/replay.json", help="replay workload JSON in the pod")
    parser.add_argument(
        "--tokenizer-path",
        default="/data/model-cache/mimo-v26/MiMo-V2.6-Flash-RL-w4a16-linear",
        help="local tokenizer in the pod, for the DFlash probe",
    )
    parser.add_argument("--serve-args", default="", help="launcher serve args for every phase")
    parser.add_argument("--checks", default=",".join(check.name for check in CHECKS), help="comma-separated")
    parser.add_argument("--check-timeout", type=float, default=7200.0, help="seconds per check")
    parser.add_argument("--output", type=Path, default=Path("mimo_final_gate.json"))
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    names = args.checks.split(",")
    unknown = set(names) - {check.name for check in CHECKS}
    if unknown:
        raise SystemExit(f"unknown checks {sorted(unknown)}")
    summary = run_gate(args, [check for check in CHECKS if check.name in names])
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"{'PASSED' if summary['passed'] else 'FAILED'}; summary in {args.output}")
    if not summary["passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
