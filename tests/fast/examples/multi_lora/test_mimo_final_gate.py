"""The final gate restarts the service once per phase, applies each check's bar and writes one summary."""

import json
from pathlib import Path

import pytest
from examples.multi_lora.tools import mimo_dev as dev
from examples.multi_lora.tools import mimo_final_gate as final
from examples.multi_lora.tools import mimo_gates as gates

ROUTER_LINE = (
    "(MilesDriver pid=1) [2026-10-07 01:00:00 router_manager.py:64] Router ready at host='10.42.9.87' port=20033 "
    "external_host=None\n"
)
ENGINE = {
    "rank": 0,
    "url": "http://10.42.9.87:30000",
    "running_requests": 0,
    "queued_requests": 0,
    "max_running_requests": 64,
    "full_kv": {"used_tokens": 0, "evictable_tokens": 0, "total_tokens": 1000},
    "swa_kv": None,
    "host_kv": None,
}
PASSING = {
    "kl-decompose": {"passed": True, "failures": [], "pairs": {"dec->tr": {"k3": 0.0009}}},
    "parity": {"passed": True, "failures": [], "base": {"k3": 0.00043}, "updated": {"k3": 0.00044}},
    "long-train": {
        "datum_lengths": [250_255],
        "forward_backward_metrics": {"expert_load/cv/layer_mean:mean": 0.4},
        "seconds": {"forward_backward": 300.0},
    },
    "replay": {"generated_tokens_per_second": 812.0, "trajectories_per_minute": 12.5},
}
PROBE = {"passed": True, "failures": [], "throughput": {"masks_on_over_off": 0.98}}
DORA = {"tokens": 4096, "k3_engine_trainer_step_0": 8e-4, "k3_engine_trainer": 1.1e-3, "k3_trainer_movement": 0.05}
RECORDS_LINE = "check: 64 datums, 513 records collected, 0 missing, 0 recomputed, 64 routed, 0 sampled positions unsupported; mismatching datums: none"


class FakePod:
    """Answers the gate's mimo_dev and kubectl commands; `gate_outputs` overrides a gate's stdout."""

    def __init__(self, gate_outputs=None, failing_phase_arg=None, top_k_free_rate=980.0, noise=("", "")):
        self.noise = noise
        self.gate_outputs = {name: json.dumps(summary) for name, summary in PASSING.items()} | (gate_outputs or {})
        self.failing_phase_arg = failing_phase_arg
        self.rates = {"a_support": 1000.0, "f_top_p": top_k_free_rate}
        self.calls = []

    def probe(self, argv) -> str:
        case = argv[argv.index("--throughput-case") + 1] if "--throughput-case" in argv else "a_support"
        on = {"decode_tokens_per_second": self.rates[case], "accept_length": 3.2}
        return json.dumps(PROBE | {"throughput": PROBE["throughput"] | {"case": case, "on": on}}, indent=2)

    def __call__(self, argv, timeout=None, capture=True):
        code, stdout = self.answer(argv)
        before, after = self.noise
        return code, before + stdout + after

    def answer(self, argv):
        self.calls.append(argv)
        if argv[0] == "kubectl":
            command = argv[4]
            return 0, json.dumps({"engines": [ENGINE]}) if command == "curl" else "noise\n" + ROUTER_LINE
        command = argv[2]
        if command == "restart":
            return (1 if self.failing_phase_arg in argv else 0), ""
        if command == "gate":
            return 0, self.gate_outputs[argv[4]] + "\n"
        return (
            0,
            {
                "dflash_support_check.py": self.probe(argv),
                "bench_rollout_records.py": f"progress\n{RECORDS_LINE}\n",
                "run_dora_parity_check.py": "step 1: {'loss': 1.0} {'grad_norm': 0.4}\n" + json.dumps(DORA, indent=2),
            }[Path(argv[4]).name],
        )

    def commands(self, command: str) -> list[list[str]]:
        return [argv for argv in self.calls if argv[0] != "kubectl" and argv[2] == command]


def run_final(monkeypatch, tmp_path, pod: FakePod, *extra: str) -> tuple[dict, int]:
    monkeypatch.setattr(final, "_run", pod)
    output = tmp_path / "gate.json"
    code = 0
    try:
        final.main(["mimo-dev", "--serve-args", "--hf-checkpoint /ckpt", "--output", str(output), *extra])
    except SystemExit as exit:
        code = exit.code
    return json.loads(output.read_text()), code


NOISE = (
    "<stdin>:233: DeprecationWarning: The 'name' parameter is deprecated\n",
    "\nException ignored in: <function _close_poller_threadsafe>\n{'not': 'json'}\nshutting down\n",
)


@pytest.mark.parametrize("noise", [("", ""), NOISE])
def test_a_passing_run_restarts_once_per_phase_and_lists_checks_in_the_requested_order(
    monkeypatch, tmp_path, capsys, noise
):
    pod = FakePod(noise=noise)
    summary, code = run_final(monkeypatch, tmp_path, pod)

    assert code == 0 and summary["passed"]
    assert [check["name"] for check in summary["checks"]] == [check.name for check in final.CHECKS]
    restarts = [argv[4:] for argv in pod.commands("restart")]
    assert restarts == [
        ["--hf-checkpoint", "/ckpt", *final.PHASE_ARGS[phase]] for phase in ("base", "collect", "muown")
    ]
    results = {check["name"]: check["result"] for check in summary["checks"]}
    assert results["records-check"]["router_url"] == "http://10.42.9.87:20033"
    assert results["dflash-top-k-free"]["f_top_p_over_a_support"] == 0.98
    lines = capsys.readouterr().out.splitlines()
    assert "PASS kl-decompose: dec->tr k3 0.00090" in lines
    assert "PASS parity: base k3 0.00043, updated k3 0.00044" in lines
    assert "PASS dflash-top-k-free: a_support 1000 tok/s accept 3.2, f_top_p 980 tok/s accept 3.2; f/a 0.980" in lines
    assert "PASS replay: 812 tok/s, 12.5 trajectories/min" in lines


def test_gate_commands_parse_with_the_bars_settings(monkeypatch, tmp_path):
    pod = FakePod()
    run_final(monkeypatch, tmp_path, pod)

    parsed = {
        argv[4]: gates.build_parser().parse_args([argv[4], "--base-url", "u", *argv[5:]])
        for argv in pod.commands("gate")
    }
    assert parsed["kl-decompose"].max_k3 == 0.0015 and parsed["kl-decompose"].workload == Path("/root/replay.json")
    assert (parsed["parity"].top_p, parsed["parity"].top_k, parsed["parity"].max_k3) == (0.97, -1, 0.001)
    assert parsed["long-train"].context_tokens == 250_000
    assert (parsed["replay"].concurrency, str(parsed["replay"].engine_log)) == (64, final.SERVICE_LOG)
    probe, top_k, top_k_free, bench, dora = (dev.build_parser().parse_args(argv[2:]) for argv in pod.commands("run"))
    assert probe.script == final.TOOLS_DIR / "dflash_support_check.py" and probe.script.exists()
    assert probe.script_args[:2] == ["--url", ENGINE["url"]]
    for throughput, case in ((top_k, "a_support"), (top_k_free, "f_top_p")):
        assert throughput.script_args[-4:] == ["--sweep-requests", "0", "--throughput-case", case]
    assert bench.script == final.TOOLS_DIR.parent / "bench_rollout_records.py" and bench.script.exists()
    assert bench.script_args == [
        "--router-url",
        "http://10.42.9.87:20033",
        "--replay",
        "/root/replay.json",
        "--check",
        "--concurrency",
        "64",
    ]
    assert dora.script == final.TOOLS_DIR.parent / "run_dora_parity_check.py" and dora.script.exists()
    # the flags the gate passes exist in the scripts it runs
    for script, flags in ((probe, ("--sweep-requests", "--throughput-case")), (dora, dora.script_args[::2])):
        assert all(f'"{flag}"' in script.script.read_text() for flag in flags), script.script
    assert dora.script_args == [
        "--base-url",
        dev.GATEWAY_URL,
        "--steps",
        "5",
        "--learning-rate",
        "0.0001",
        "--max-k3",
        "0.002",
        "--min-movement-k3",
        "0.02",
    ]


def test_a_check_below_its_bar_fails_the_gate_but_the_others_still_run(monkeypatch, tmp_path):
    pod = FakePod({"replay": json.dumps({"generated_tokens_per_second": 650.0}), "parity": ""})
    summary, code = run_final(monkeypatch, tmp_path, pod)

    assert code == 1 and not summary["passed"]
    failed = {check["name"]: check["failures"] for check in summary["checks"] if not check["passed"]}
    assert failed.keys() == {"replay", "parity"}
    assert failed["replay"] == ["650 tok/s < 700"]
    assert failed["parity"] == ["ValueError: no JSON object on stdout; its last lines: []"]


def test_a_phase_whose_service_does_not_start_fails_only_its_checks(monkeypatch, tmp_path):
    pod = FakePod(failing_phase_arg="--rollout-record-cache-gb")
    summary, code = run_final(monkeypatch, tmp_path, pod)

    failed = {check["name"]: check["failures"] for check in summary["checks"] if not check["passed"]}
    assert code == 1 and failed == {"records-check": ["the collect service did not start"]}
    assert [Path(argv[4]).name for argv in pod.commands("run")] == [
        *["dflash_support_check.py"] * 3,
        "run_dora_parity_check.py",
    ]


def test_a_slow_top_k_free_path_is_reported_without_failing_the_gate(monkeypatch, tmp_path):
    pod = FakePod(top_k_free_rate=900.0)
    summary, code = run_final(monkeypatch, tmp_path, pod)

    check = next(check for check in summary["checks"] if check["name"] == "dflash-top-k-free")
    assert code == 0 and summary["passed"] and not check["passed"] and not check["blocking"]
    assert check["failures"] == ["top_k-free decode tok/s is 0.900x top_k 1024's, < 0.95: keep top_k=1024"]
    assert check["result"]["f_top_p"]["on"] == {"decode_tokens_per_second": 900.0, "accept_length": 3.2}


def test_selected_checks_restart_only_their_phases(monkeypatch, tmp_path):
    pod = FakePod()
    summary, code = run_final(monkeypatch, tmp_path, pod, "--checks", "parity,muown-parity")

    assert code == 0 and [check["name"] for check in summary["checks"]] == ["parity", "muown-parity"]
    assert len(pod.commands("restart")) == 2
    with pytest.raises(SystemExit, match="unknown checks"):
        final.main(["mimo-dev", "--checks", "parity,typo"])


@pytest.mark.parametrize(
    "bar, result, failure",
    [
        (final._dflash_bar, PROBE, None),
        (final._dflash_bar, PROBE | {"throughput": {"masks_on_over_off": 0.9}}, "ratio 0.900 < 0.95"),
        (final._dflash_bar, PROBE | {"passed": False, "failures": ["cases: x"]}, "cases: x"),
        (final._records_bar, {"check_line": RECORDS_LINE}, None),
        (
            final._records_bar,
            {"check_line": "check: 64 datums; mismatching datums: {'rollout_routed_experts': 2}"},
            "{'rollout_routed_experts': 2}",
        ),
        (final._records_bar, {"check_line": None}, "printed no check line"),
        (final._long_train_bar, PASSING["long-train"], None),
        (final._long_train_bar, PASSING["long-train"] | {"datum_lengths": [131_072]}, "131072 tokens < 250000"),
        (final._long_train_bar, PASSING["long-train"] | {"forward_backward_metrics": {"loss": 1.0}}, "no expert_load"),
        (final._replay_bar, {"generated_tokens_per_second": 700.0}, None),
        (final._engine_load_bar, {"engines": [ENGINE]}, None),
        (final._engine_load_bar, {"engines": []}, "no engines"),
        (final._engine_load_bar, {"engines": [{"rank": 0, "url": "u"}]}, "engine 0 lacks ['running_requests'"),
        (final._gate_bar, {"passed": False, "failures": ["dec->tr k3 0.00200 > 0.0015"]}, "dec->tr k3 0.00200"),
        (final._dora_bar, DORA, None),
        (final._dora_bar, DORA | {"k3_engine_trainer": 3e-3}, "engine/trainer k3 3.00e-03 > 0.002"),
        (final._dora_bar, DORA | {"k3_trainer_movement": 0.01}, "trainer movement k3 1.00e-02 < 0.02"),
    ],
)
def test_bars(bar, result, failure):
    failures = bar(result)
    if failure is None:
        assert failures == []
    else:
        assert len(failures) == 1 and failure in failures[0], failures


def test_last_json_finds_the_summary_among_other_output():
    summary = {"passed": True, "pairs": {"dec->tr": {"k3": 0.0009}}}
    assert final.last_json(f"warning\n{json.dumps(summary)}\nException ignored\n{{'repr': 1}}\n") == summary
    assert final.last_json("step 1: {'loss': 1.0}\n" + json.dumps(DORA, indent=2) + "\ndone\n") == DORA
    with pytest.raises(ValueError, match="its last lines: \\['parity top-p 0.97: base k3 0.00043'\\]"):
        final.last_json("parity top-p 0.97: base k3 0.00043\n")


def test_router_url_takes_the_last_router_ready_line():
    later = ROUTER_LINE.replace("20033", "20044")
    assert final.router_url(ROUTER_LINE + later) == "http://10.42.9.87:20044"
