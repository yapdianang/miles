"""The MiMo gates compute k3 from trainer - sampler log-probs and run every gate against a fake gateway."""

import gzip
import json
import math
import random
from types import SimpleNamespace

import pytest
from examples.multi_lora.tools import mimo_gates as gates

PREFILL_LINE = (
    "\x1b[36m(CommandActor pid=105447)\x1b[0m [2026-10-06 11:00:38 TP0 EP0] Prefill batch, #new-seq: 1, "
    "#new-token: 4061, #cached-token: 1000, full token usage: 0.01, swa token usage: 0.17, #running-req: 0, "
    "#queue-req: 51, #pending-token: 0, cuda graph: False, input throughput (token/s): 16.21"
)
DECODE_LINES = [
    "\x1b[36m(CommandActor pid=105447)\x1b[0m [2026-10-06 11:00:59 TP0 EP0] Decode batch, #running-req: 56, "
    "#full token: 20224, full token usage: 0.00, #swa token: 14976, swa token usage: 0.05, accept len: 2.69, "
    "accept rate: 0.24, cuda graph: True, gen throughput (token/s): 22.47, #queue-req: 0",
    "\x1b[36m(CommandActor pid=105447)\x1b[0m [2026-10-06 11:01:01 TP0 EP0] Decode batch, #running-req: 13, "
    "#full token: 8704, full token usage: 0.00, #swa token: 3712, swa token usage: 0.01, accept len: 3.19, "
    "accept rate: 0.31, cuda graph: True, gen throughput (token/s): 0.50, #queue-req: 2",
]


def _workload(lengths: list[list[int]]) -> list[list[dict]]:
    return [[{"prompt": [1] * length, "output_len": 2} for length in turns] for turns in lengths]


class _Future:
    def __init__(self, value):
        self._value = value

    def result(self):
        return self._value


class _Logprobs(list):
    def tolist(self) -> list[float]:
        return list(self)


class FakeGateway:
    """A Tinker gateway whose sampled tokens have log-prob -1; prefill and trainer read them shifted."""

    def __init__(self, trainer_shift=0.0, prefill_shift=0.0, engine_log=None):
        self.trainer_shift, self.prefill_shift, self.engine_log = trainer_shift, prefill_shift, engine_log
        self.prompts = []

    def __call__(self, base_url, api_key):
        return self

    def create_lora_training_client(self, model, rank):
        return FakeTraining(self)

    async def create_lora_training_client_async(self, model, rank):
        return FakeTraining(self)


class FakeSampler:
    def __init__(self, gateway):
        self.gateway = gateway

    def _response(self, prompt, params, num_samples):
        self.gateway.prompts.append(prompt.to_ints())
        tokens = [11, 12, 13][: params.max_tokens]
        sequence = SimpleNamespace(tokens=tokens, logprobs=[-1.0] * len(tokens))
        return SimpleNamespace(sequences=[sequence] * num_samples)

    def sample(self, prompt, num_samples, sampling_params):
        return _Future(self._response(prompt, sampling_params, num_samples))

    async def sample_async(self, prompt, num_samples, sampling_params):
        if self.gateway.engine_log is not None:
            with self.gateway.engine_log.open("a") as log:
                log.write(PREFILL_LINE + "\n")
        return self._response(prompt, sampling_params, num_samples)

    def compute_logprobs(self, model_input):
        return _Future([None] + [-1.0 + self.gateway.prefill_shift] * (model_input.length - 1))


class FakeTraining:
    def __init__(self, gateway):
        self.gateway = gateway

    def save_weights_and_get_sampling_client(self, name):
        return FakeSampler(self.gateway)

    async def save_weights_and_get_sampling_client_async(self, name):
        return FakeSampler(self.gateway)

    def forward(self, datums, loss_fn):
        outputs = [
            {
                "logprobs": _Logprobs(
                    value + self.gateway.trainer_shift for value in datum.loss_fn_inputs["logprobs"].tolist()
                )
            }
            for datum in datums
        ]
        return _Future(SimpleNamespace(loss_fn_outputs=outputs, metrics={"expert_load/cv/layer_mean:mean": 0.3}))

    forward_backward = forward

    def optim_step(self, params):
        return _Future(None)


class FakeTokenizer:
    def apply_chat_template(self, messages, add_generation_prompt, tokenize):
        return f"<user>{messages[0]['content']}<assistant>"

    def encode(self, text, add_special_tokens):
        return [len(text) % 50 + 3, 5, 6]

    def decode(self, tokens):
        return " ".join(map(str, tokens))


def run_gate(monkeypatch, capsys, argv, gateway) -> tuple[dict, str, int]:
    monkeypatch.setattr(gates.tinker, "ServiceClient", gateway)
    monkeypatch.setattr(gates.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: FakeTokenizer())
    code = 0
    try:
        gates.main([argv[0], "--base-url", "http://gateway", *argv[1:]])
    except SystemExit as exit:
        code = exit.code
    out, err = capsys.readouterr()
    return json.loads(out), err.strip(), code


def test_gap_is_k3_mean_abs_and_mean_of_target_minus_reference():
    stats = gates.gap([-0.9, -1.2], [-1.0, -1.0])
    assert stats["tokens"] == 2
    assert stats["k3"] == pytest.approx(((math.exp(0.1) - 0.1 - 1) + (math.exp(-0.2) + 0.2 - 1)) / 2)
    assert stats["mean_abs"] == pytest.approx(0.15)
    assert stats["mean_d"] == pytest.approx(-0.05)
    assert gates.gap([-1.0], [-1.0])["k3"] == 0.0


def test_datum_zeroes_the_prompt_and_trainer_logprobs_drop_it():
    datum = gates.build_datum([1, 2, 3], [4, 5], [-0.1, -0.2], advantage=1.0)
    assert datum.model_input.to_ints() == [1, 2, 3, 4]
    assert datum.loss_fn_inputs["target_tokens"].tolist() == [2, 3, 4, 5]
    assert datum.loss_fn_inputs["logprobs"].tolist() == pytest.approx([0.0, 0.0, -0.1, -0.2])
    assert datum.loss_fn_inputs["advantages"].tolist() == [0.0, 0.0, 1.0, 1.0]
    assert gates.trainer_logprobs({"logprobs": _Logprobs([-9.0, -9.0, -0.3, -0.4])}, prompt_length=3) == [-0.3, -0.4]


def test_workload_selection():
    workload = _workload([[3_000, 5_000, 40_000], [20_000, 7_000], [100]])
    assert [len(prompt) for prompt in gates.kl_contexts(workload, 2, 4_000, 32_000)] == [5_000, 20_000]
    buckets = gates.length_buckets(workload, per_bucket=1)
    assert [(label, len(contexts[0])) for label, contexts in buckets] == [
        ("0-6K", 3_000),
        ("6-16K", 7_000),
        ("16-32K", 20_000),
        ("32-64K", 40_000),
    ]
    assert [len(prompt) for prompt in gates.longest_contexts(workload, 2)] == [40_000, 7_000]


def test_long_context_is_cut_from_the_longest_prompts_and_repeats_them_when_short():
    workload = _workload([[10, 30], [20]])
    assert gates.long_context(workload, 40) == [1] * 40
    assert len(gates.long_context(workload, 125)) == 125


def test_replay_copies_get_distinct_prefixes_and_a_seeded_order():
    workload = _workload([[10], [20], [30]])
    jobs = gates.replay_jobs(workload, copies=4, rng=random.Random(0))
    assert len(jobs) == 12
    assert len({tuple(prefix) for prefix, _ in jobs}) == 12
    assert all(len(prefix) == gates.REPLAY_PREFIX_TOKENS for prefix, _ in jobs)
    assert jobs == gates.replay_jobs(workload, copies=4, rng=random.Random(0))


def test_workloads_load_from_json_and_gzip(tmp_path):
    workload = _workload([[5, 6]])
    (tmp_path / "w.json").write_text(json.dumps(workload))
    with gzip.open(tmp_path / "w.json.gz", "wt") as file:
        json.dump(workload, file)
    assert gates.load_workload(tmp_path / "w.json") == gates.load_workload(tmp_path / "w.json.gz") == workload


def test_engine_stats_from_sglang_batch_lines():
    stats = gates.engine_stats([PREFILL_LINE, *DECODE_LINES, "unrelated line"])
    assert (stats["prefill_seqs"], stats["new_tokens"], stats["cached_tokens"]) == (1, 4061, 1000)
    assert stats["cache_hit_rate"] == pytest.approx(1000 / 5061)
    assert stats["max_full_token_usage"] == 0.01
    assert stats["decode_logs"] == 2
    # The idle log's 0.5 tok/s is left out of the mean but not the max.
    assert stats["mean_gen_tokens_per_second"] == 22.47
    assert stats["max_gen_tokens_per_second"] == 22.47
    assert stats["mean_running"] == 34.5 and stats["mean_queue"] == 1.0
    assert stats["mean_accept_length"] == pytest.approx(2.94)
    empty = gates.engine_stats([])
    assert empty["cache_hit_rate"] is None and empty["mean_gen_tokens_per_second"] is None


def test_gate_defaults_are_the_settings_the_bars_were_measured_at():
    parser = gates.build_parser()
    parity = parser.parse_args(["parity", "--base-url", "u"])
    assert (parity.top_p, parity.top_k, parity.max_tokens, parity.lr, parity.max_k3) == (0.97, 1024, 1024, 1e-3, 0.001)
    decompose = parser.parse_args(["kl-decompose", "--base-url", "u", "--workload", "w"])
    assert (decompose.contexts, decompose.min_prompt, decompose.max_prompt, decompose.max_k3) == (
        24,
        4000,
        32000,
        0.0015,
    )
    replay = parser.parse_args(["replay", "--base-url", "u", "--workload", "w"])
    assert (replay.copies, replay.concurrency, replay.env_delay, replay.engine_log) == (1, 0, 0.0, None)
    with pytest.raises(SystemExit):
        parser.parse_args(["kl-decompose", "--base-url", "u"])


@pytest.mark.parametrize("shift, passed", [(0.01, True), (0.1, False)])
def test_parity_fails_past_the_k3_bar(monkeypatch, capsys, shift, passed):
    summary, line, code = run_gate(monkeypatch, capsys, ["parity"], FakeGateway(trainer_shift=shift))
    assert summary["base"]["k3"] == pytest.approx(math.exp(shift) - shift - 1)
    assert summary["updated"]["mean_d"] == pytest.approx(shift)
    assert summary["base"]["tokens"] == 4 * 2 * 3
    assert [move["advantage"] for move in summary["logprob_sums"]] == [1.0, -1.0] * 4
    assert summary["passed"] is passed and code == (0 if passed else 1)
    assert line.startswith("parity top-p 0.97: base k3") and line.endswith("(pass)" if passed else "(FAIL)")


def test_kl_decompose_separates_prefill_from_trainer(monkeypatch, capsys, tmp_path):
    (tmp_path / "w.json").write_text(json.dumps(_workload([[5_000, 6_000], [10]])))
    gateway = FakeGateway(trainer_shift=0.05, prefill_shift=0.02)
    summary, line, code = run_gate(
        monkeypatch, capsys, ["kl-decompose", "--workload", str(tmp_path / "w.json")], gateway
    )
    means = {name: pair["mean_d"] for name, pair in summary["pairs"].items()}
    assert means == pytest.approx({"pre1->pre2": 0.0, "dec->pre1": 0.02, "pre1->tr": 0.03, "dec->tr": 0.05})
    assert summary["turns"] == 2 and summary["tokens"] == 6 and summary["entropy_proxy"] == 1.0
    assert summary["passed"] and code == 0 and "dec->tr k3 0.00127" in line


def test_length_and_long_train_gates_run(monkeypatch, capsys, tmp_path):
    (tmp_path / "w.json").write_text(json.dumps(_workload([[1_000, 7_000], [20_000]])))
    workload = ["--workload", str(tmp_path / "w.json")]
    summary, _, _ = run_gate(monkeypatch, capsys, ["parity-by-length", *workload], FakeGateway(trainer_shift=0.01))
    assert [row["bucket"] for row in summary["buckets"]] == ["0-6K", "6-16K", "16-32K"]
    summary, line, _ = run_gate(monkeypatch, capsys, ["long-train", *workload, "--contexts", "1"], FakeGateway())
    assert summary["datum_lengths"] == [20_000 + 3 - 1]
    assert set(summary["seconds"]) == {"sample", "forward", "forward_backward"}
    assert summary["forward_backward_metrics"] == {"expert_load/cv/layer_mean:mean": 0.3}
    summary, _, _ = run_gate(
        monkeypatch, capsys, ["long-train", *workload, "--context-tokens", "50000"], FakeGateway()
    )
    assert summary["datum_lengths"] == [50_000 + 3 - 1]


def test_replay_reads_engine_stats_only_from_its_own_log_lines(monkeypatch, capsys, tmp_path):
    (tmp_path / "w.json").write_text(json.dumps(_workload([[10, 20], [30]])))
    log = tmp_path / "service.log"
    log.write_text(PREFILL_LINE + "\n")
    gateway = FakeGateway(engine_log=log)
    argv = ["replay", "--workload", str(tmp_path / "w.json"), "--copies", "2", "--engine-log", str(log)]
    summary, line, _ = run_gate(monkeypatch, capsys, argv, gateway)
    assert summary["trajectories"] == 4 and summary["concurrency"] == 4 and summary["turns"] == 6
    assert summary["generated_tokens"] == 12
    assert summary["engine"]["prefill_seqs"] == 6
    assert summary["prompt_tokens"] == 6 * gates.REPLAY_PREFIX_TOKENS + 2 * (10 + 20 + 30)
    assert all(len(prompt) - gates.REPLAY_PREFIX_TOKENS in (10, 20, 30) for prompt in gateway.prompts)
    assert "cache hit 0.20" in line
