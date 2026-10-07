"""DFlash RL drafter pipeline: data formats, the drafter forward against a per-query reference, the loss, and a
CPU dry run of capture -> train -> offline eval on tiny shapes."""

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
import torch
from examples.multi_lora.dflash_rl import (
    bench_engine,
    capture_hook,
    draft_probe,
    eval_offline,
    run_pipeline,
    train_drafter,
)
from examples.multi_lora.dflash_rl.data import (
    assemble_shard,
    context_spans,
    load_shard,
    make_chunk,
    save_shard,
    turn_spans,
    valid_anchors,
    walk_accept,
)
from examples.multi_lora.dflash_rl.drafter import (
    DFlashDrafter,
    DrafterConfig,
    _rope,
    block_loss,
    decay_weights,
    load_drafter,
    save_drafter,
)
from examples.multi_lora.dflash_rl.engine import server_argv
from examples.multi_lora.dflash_rl.export_rollouts import rollout_from_step, select_rollouts
from safetensors.torch import save_file

VOCAB = 64
TINY_CONFIG = {
    "architectures": ["DFlashDraftModel"],
    "hidden_size": 32,
    "intermediate_size": 64,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 8,
    "partial_rotary_factor": 0.5,
    "block_size": 4,
    "dflash_config": {
        "target_layer_ids": [0, 2],
        "mask_token_id": VOCAB - 1,
        "num_anchors": 4096,
        "block_size": 4,
        "loss_decay_gamma": 7.0,
        "attention_value_scale": 0.612,
        "attention_sink_bias": True,
    },
    "layer_types": ["sliding_attention", "sliding_attention"],
    "sliding_window": 6,
    "is_causal": False,
    "rope_theta": 10000.0,
    "rms_norm_eps": 1e-6,
    "vocab_size": VOCAB,
}


def _tiny_drafter_dir(root: Path) -> Path:
    source, out = root / "config", root / "drafter"
    source.mkdir()
    (source / "config.json").write_text(json.dumps(TINY_CONFIG))
    torch.manual_seed(0)
    drafter = DFlashDrafter(DrafterConfig.from_dir(source))
    for parameter in drafter.parameters():
        torch.nn.init.normal_(parameter, std=0.3)
    save_drafter(drafter, source=source, out=out)
    return out


def _tiny_target(root: Path) -> Path:
    path = root / "target"
    path.mkdir()
    tensors = {"model.embed_tokens.weight": torch.randn(VOCAB, 32), "lm_head.weight": torch.randn(VOCAB, 32)}
    save_file({name: tensor.to(torch.bfloat16) for name, tensor in tensors.items()}, path / "model.safetensors")
    weight_map = dict.fromkeys(tensors, "model.safetensors")
    (path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    return path


def _shard(tokens: list[int], loss_mask: list[int], keep: list[int], width: int) -> dict[str, torch.Tensor]:
    positions = torch.tensor(keep)
    return {
        "input_ids": torch.tensor(tokens)[positions],
        "positions": positions,
        "loss_mask": torch.tensor(loss_mask, dtype=torch.bool)[positions],
        "hidden": torch.randn(len(keep), width),
    }


def test_turn_and_context_spans_cover_each_turn_and_its_window() -> None:
    loss_mask = [0] * 5 + [1] * 3 + [0] * 2 + [1] * 2 + [0] * 20 + [1] * 2
    assert turn_spans(loss_mask) == [(5, 8), (10, 12), (32, 34)]
    assert context_spans(loss_mask, window=4) == [(2, 12), (29, 34)]
    assert context_spans([1, 1, 0], window=1024) == [(0, 2)]


def test_rollout_from_step_rejects_misaligned_masks() -> None:
    assert rollout_from_step({"tokens": [1, 2, 3], "token_masks": [0, 1, 1]})["loss_mask"] == [0, 1, 1]
    with pytest.raises(ValueError):
        rollout_from_step({"tokens": [1, 2, 3], "token_masks": [0, 1]})
    with pytest.raises(ValueError):
        rollout_from_step({"tokens": [1, 2], "token_masks": [0, 0]})


def _rows(task: str, count: int, *, trained: int = 0, collected: int = 0) -> list[dict]:
    return [
        {
            "key": f"{task}{index}",
            "task_id": task,
            "policy_step": index % 10,
            "collected": index < max(collected, trained),
            "trained": index < trained,
            "env_done": index % 4 != 0,
        }
        for index in range(count)
    ]


def test_select_rollouts_matches_the_trained_task_mix_and_filters_policy_steps() -> None:
    rows = [*_rows("a", 1000, trained=50), *_rows("b", 1000, trained=450), *_rows("never", 1000, collected=10)]
    chosen = select_rollouts(rows, num=200, max_policy_step=None, seed=0)
    counts = {task: sum(row["task_id"] == task for row in chosen) for task in ("a", "b", "never")}
    assert len({row["key"] for row in chosen}) == 200 and all(row["collected"] for row in chosen)
    assert counts["never"] == 0 and 10 <= counts["a"] <= 30
    assert all(row["policy_step"] <= 2 for row in select_rollouts(rows, num=100, max_policy_step=2, seed=0))


def test_select_rollouts_falls_back_to_env_done_rollouts_of_an_untrained_run() -> None:
    rows = [*_rows("a", 100, collected=5), *_rows("b", 300, collected=5)]
    chosen = select_rollouts(rows, num=200, max_policy_step=None, seed=0)
    assert len(chosen) == 200 and all(row["env_done"] for row in chosen)
    assert 30 <= sum(row["task_id"] == "a" for row in chosen) <= 70


def test_capture_hook_saves_spanned_rows_of_extend_batches_only(tmp_path: Path) -> None:
    (tmp_path / "r1.spans.json").write_text(json.dumps([[2, 5], [7, 9]]))
    hook = capture_hook.make_hook({"dir": str(tmp_path)})
    hidden = torch.arange(9 * 3, dtype=torch.float32).view(9, 3)

    def batch(mode: str, prefix: int) -> tuple:
        forward_batch = SimpleNamespace(
            forward_mode=SimpleNamespace(name=mode),
            rids=["r2", "r1"],
            extend_prefix_lens_cpu=[0, prefix],
            extend_seq_lens_cpu=[3, 6],
        )
        return (None, None, None, forward_batch)

    hook(None, batch("TARGET_VERIFY", 0), SimpleNamespace(hidden_states=hidden))
    assert not list(tmp_path.glob("*.pt"))
    hook(None, batch("EXTEND", 0), SimpleNamespace(hidden_states=hidden))
    hook(None, batch("EXTEND", 6), SimpleNamespace(hidden_states=hidden))
    first, second = (torch.load(tmp_path / f"r1.{prefix}.pt") for prefix in (0, 6))
    assert first["positions"].tolist() == [2, 3, 4] and torch.equal(first["hidden"], hidden[5:8])
    assert second["positions"].tolist() == [7, 8] and torch.equal(second["hidden"], hidden[4:6])
    assert not list(tmp_path.glob("r2.*"))
    with pytest.raises(RuntimeError):
        hook(None, batch("EXTEND", 0), SimpleNamespace(hidden_states=hidden[:4]))


def test_assemble_shard_joins_chunks_and_rejects_gaps() -> None:
    tokens, loss_mask = list(range(10)), [0, 0, 0, 1, 1, 0, 0, 0, 1, 1]
    spans = context_spans(loss_mask, window=3)
    chunks = [
        {"positions": torch.tensor([6, 7, 8, 9]), "hidden": torch.ones(4, 2)},
        {"positions": torch.tensor([1, 2, 3, 4]), "hidden": torch.zeros(4, 2)},
    ]
    shard = assemble_shard(tokens, loss_mask, spans, chunks, hidden_size=2)
    assert shard["positions"].tolist() == [1, 2, 3, 4, 6, 7, 8, 9]
    assert shard["input_ids"].tolist() == [1, 2, 3, 4, 6, 7, 8, 9]
    assert shard["loss_mask"].tolist() == [False, False, True, True, False, False, True, True]
    with pytest.raises(ValueError, match="positions"):
        assemble_shard(tokens, loss_mask, spans, chunks[:1], hidden_size=2)
    with pytest.raises(ValueError, match="width"):
        assemble_shard(tokens, loss_mask, spans, chunks, hidden_size=3)


def test_make_chunk_labels_stop_at_gaps_and_context_respects_the_window() -> None:
    tokens = list(range(100, 130))
    loss_mask = [0] * 10 + [1] * 5 + [0] * 10 + [1] * 5
    keep = [index for start, end in context_spans(loss_mask, window=4) for index in range(start, end)]
    shard = _shard(tokens, loss_mask, keep, width=2)
    anchors = valid_anchors(shard)
    assert shard["positions"][anchors].tolist() == [10, 11, 12, 13, 25, 26, 27, 28]
    chunk = make_chunk(shard, anchors, block_size=4, window=4, mask_token_id=7)
    assert chunk.block_ids[0].tolist() == [110, 7, 7, 7]
    assert chunk.labels[2].tolist() == [113, 114, 122]
    assert chunk.label_mask[2].tolist() == [True, True, False]
    assert chunk.label_mask[4].tolist() == [True, True, True]
    for block, anchor in enumerate(shard["positions"][anchors].tolist()):
        for slot in range(4):
            expected = [anchor + slot - 3 <= position < anchor for position in chunk.ctx_positions.tolist()]
            assert chunk.ctx_visible[block, slot].tolist() == expected
    with pytest.raises(ValueError, match="context"):
        make_chunk(_shard(tokens, loss_mask, keep[2:], width=2), anchors - 2, block_size=4, window=4, mask_token_id=7)


def test_rope_rotates_the_leading_channels_by_relative_position() -> None:
    q, k = torch.randn(1, 2, 8), torch.randn(1, 2, 8)

    def score(m: int, n: int) -> torch.Tensor:
        rope = lambda x, p: _rope(x, torch.tensor([p]), rotary_dim=4, theta=10000.0)  # noqa: E731
        return (rope(q, m) * rope(k, n)).sum(-1)

    assert torch.allclose(score(3, 1), score(12, 10), atol=1e-5)
    assert torch.equal(_rope(q, torch.tensor([5]), rotary_dim=4, theta=10000.0)[..., 4:], q[..., 4:])


def _reference_block(drafter: DFlashDrafter, block_ids, anchor: int, ctx_hidden, ctx_positions, embed):
    """One block, query by query: keys within the two-sided window over context and block, plus a sink."""
    config = drafter.config
    x = torch.where((block_ids == config.mask_token_id)[:, None], drafter.mask_embedding, embed[block_ids])
    block_positions = anchor + torch.arange(len(block_ids))
    ctx = drafter.hidden_norm(drafter.fc(ctx_hidden))
    key_positions = torch.cat([ctx_positions, block_positions])
    rope = lambda x: _rope(x, key_positions, rotary_dim=config.rotary_dim, theta=config.rope_theta)  # noqa: E731
    for layer in drafter.layers:
        attention = layer.self_attn
        h = layer.input_layernorm(x)
        q = attention.q_norm(attention.q_proj(h).view(len(block_ids), config.num_heads, config.head_dim))
        q = _rope(q, block_positions, rotary_dim=config.rotary_dim, theta=config.rope_theta)
        sequence = torch.cat([ctx, h])
        keys = rope(attention.k_norm(attention.k_proj(sequence).view(len(sequence), config.num_kv_heads, -1)))
        values = attention.v_proj(sequence).view(len(sequence), config.num_kv_heads, -1) * config.value_scale
        out = torch.zeros(len(block_ids), config.num_heads, config.head_dim)
        for slot, position in enumerate(block_positions.tolist()):
            visible = (key_positions - position).abs() <= config.sliding_window - 1
            for head in range(config.num_heads):
                kv_head = head // (config.num_heads // config.num_kv_heads)
                scores = keys[visible, kv_head] @ q[slot, head] * config.head_dim**-0.5
                probs = torch.cat([scores, attention.attention_sink_bias[head : head + 1]]).softmax(-1)[:-1]
                out[slot, head] = probs @ values[visible, kv_head]
        x = x + attention.o_proj(out.flatten(1))
        x = x + layer.mlp(layer.post_attention_layernorm(x))
    return drafter.norm(x)


def test_drafter_forward_matches_a_per_query_reference(tmp_path: Path) -> None:
    drafter = load_drafter(_tiny_drafter_dir(tmp_path))
    config = drafter.config
    embed = torch.randn(VOCAB, config.hidden_size)
    tokens = torch.randint(0, VOCAB - 1, (30,)).tolist()
    loss_mask = [0] * 8 + [1] * 6 + [0] * 12 + [1] * 4
    keep = [index for start, end in context_spans(loss_mask, config.sliding_window) for index in range(start, end)]
    shard = _shard(tokens, loss_mask, keep, width=config.target_hidden_size)
    anchors = valid_anchors(shard)
    chunk = make_chunk(shard, anchors, block_size=4, window=config.sliding_window, mask_token_id=config.mask_token_id)
    with torch.no_grad():
        hidden = drafter(
            chunk.block_ids, chunk.block_positions, chunk.ctx_hidden, chunk.ctx_positions, chunk.ctx_visible, embed
        )
        for block, index in enumerate(anchors.tolist()):
            anchor = int(shard["positions"][index])
            before = shard["positions"] < anchor
            expected = _reference_block(
                drafter, chunk.block_ids[block], anchor, shard["hidden"][before], shard["positions"][before], embed
            )
            assert torch.allclose(hidden[block], expected, atol=1e-4), block


def test_block_loss_weights_slots_by_decay_and_counts_leading_matches() -> None:
    assert torch.allclose(decay_weights(4, 7.0, torch.device("cpu")), torch.exp(-torch.tensor([0.0, 1.0, 2.0]) / 7))
    lm_head = torch.eye(5)
    draft = torch.zeros(2, 4, 5)
    draft[:, 1:] = torch.nn.functional.one_hot(torch.tensor([[1, 2, 3], [1, 4, 3]]), 5).float() * 10
    labels = torch.tensor([[1, 2, 3], [1, 2, 3]])
    label_mask = torch.tensor([[True, True, False], [True, True, True]])
    loss, weight, accepted = block_loss(draft, labels, label_mask, lm_head, gamma=7.0)
    assert accepted.tolist() == [2, 1]
    assert torch.isclose(weight, (label_mask * decay_weights(4, 7.0, torch.device("cpu"))).sum())
    assert loss > 0


def test_walk_accept_steps_through_each_turn_and_stops_at_its_last_token() -> None:
    shard = {"loss_mask": torch.tensor([0, 1, 1, 1, 1, 1, 0, 1, 1, 1], dtype=torch.bool)}
    anchors, accepted = [1, 2, 3, 4, 7, 8], [2, 0, 1, 0, 0, 1]
    # Turn 1: 1 -> 4 (2 drafts + bonus) -> 5 (bonus); turn 2: 7 -> 8 -> 9 (one draft, no bonus past the end).
    assert walk_accept(shard, anchors, accepted) == (4 + 2, 4)


def test_save_drafter_round_trips_checkpoint_names(tmp_path: Path) -> None:
    out = _tiny_drafter_dir(tmp_path)
    drafter = load_drafter(out)
    index = json.loads((out / "model.safetensors.index.json").read_text())
    assert "layers.1.self_attn.attention_sink_bias" in index["weight_map"]
    assert "mask_embedding" not in index["weight_map"]
    mask = torch.load(out / "mask_embedding.pt", weights_only=True)
    assert mask["mask_token_id"] == VOCAB - 1 and mask["embedding"].dtype == torch.bfloat16
    assert json.loads((out / "config.json").read_text()) == TINY_CONFIG
    save_drafter(drafter, source=out, out=tmp_path / "again")
    again = load_drafter(tmp_path / "again")
    for (name, before), (_, after) in zip(drafter.state_dict().items(), again.state_dict().items(), strict=True):
        assert torch.equal(before, after), name


def test_server_argv_maps_engine_flags_to_sglang_server_flags() -> None:
    argv = server_argv(
        hf_checkpoint="/ckpt",
        drafter="/rl",
        block_size=6,
        port=30001,
        mem_fraction=0.4,
        draft_quantization="fp8",
        lora_path="/lora",
    )
    assert argv[argv.index("--mem-fraction-static") + 1] == "0.4"
    assert argv[argv.index("--tp-size") + 1] == "4"
    assert argv[argv.index("--ep-size") + 1] == "4"
    assert argv[argv.index("--speculative-algorithm") + 1] == "DFLASH"
    assert argv[argv.index("--speculative-draft-model-path") + 1] == "/rl"
    assert argv[argv.index("--speculative-num-draft-tokens") + 1] == "6"
    assert argv[argv.index("--speculative-draft-model-quantization") + 1] == "fp8"
    assert argv[argv.index("--lora-paths") + 1] == "policy=/lora"
    assert not [token for token in argv if token.startswith("--sglang-") or token.startswith("--rollout-")]


def _capture_shards(config: DrafterConfig, capture: Path, data: Path) -> None:
    """Three rollouts through the hook (two prefill chunks each) and assemble_shard, as extract_hidden does."""
    generator = torch.Generator().manual_seed(1)
    loss_mask = [0] * 6 + [1] * 12 + [0] * 10 + [1] * 12
    spans = context_spans(loss_mask, config.sliding_window)
    for key in range(3):
        tokens = torch.randint(0, VOCAB - 1, (40,), generator=generator).tolist()
        hidden = torch.randn(40, config.target_hidden_size, generator=generator)
        (capture / f"r{key}.spans.json").write_text(json.dumps(spans))
        for prefix, length in ((0, 25), (25, 15)):
            capture_hook.save_extend_rows(
                capture, rids=[f"r{key}"], prefix_lens=[prefix], extend_lens=[length], hidden=hidden[prefix:][:length]
            )
        chunks = [torch.load(path) for path in sorted(capture.glob(f"r{key}.*.pt"))]
        shard = assemble_shard(tokens, loss_mask, spans, chunks, config.target_hidden_size)
        save_shard(shard, data / f"{key}.safetensors")
        kept = [index for start, end in spans for index in range(start, end)]
        assert torch.equal(load_shard(data / f"{key}.safetensors")["hidden"], hidden[kept].bfloat16())


def test_dry_run_capture_train_and_eval_on_tiny_shapes(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    drafter_dir, target = _tiny_drafter_dir(tmp_path), _tiny_target(tmp_path)
    capture, data, out = tmp_path / "capture", tmp_path / "hidden", tmp_path / "trained"
    capture.mkdir()
    data.mkdir()
    _capture_shards(DrafterConfig.from_dir(drafter_dir), capture, data)

    common = ["--target-checkpoint", str(target), "--data", str(data)]
    train_args = [
        "--drafter",
        str(drafter_dir),
        "--out",
        str(out),
        "--epochs",
        "8",
        "--lr",
        "3e-3",
        "--log-every",
        "1",
    ]
    train_drafter.main([*common, *train_args])
    logs = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [log["step"] for log in logs] == list(range(1, 25))
    assert sum(log["loss"] for log in logs[-3:]) < 0.7 * sum(log["loss"] for log in logs[:3])

    report = tmp_path / "eval.json"
    drafters = ["--drafter", f"shipped={drafter_dir}", "--drafter", f"rl={out}"]
    eval_offline.main([*common, *drafters, "--block-sizes", "3", "4", "--output", str(report)])
    results = {(row["drafter"], row["block_size"]): row for row in json.loads(report.read_text())}
    assert set(results) == {("shipped", 3), ("shipped", 4), ("rl", 3), ("rl", 4)}
    assert results[("rl", 4)]["loss"] < results[("shipped", 4)]["loss"]
    assert all(1 <= row["walk_accept_length"] <= row["block_size"] for row in results.values())


def _fake_stage(engine_accept: float):
    """Stage outputs keyed by stage name, in place of the subprocesses."""

    def run(self, name: str, command: list[str]) -> None:
        self.elapsed[name] = 0
        if name.startswith("eval_"):
            drafter = name.removeprefix("eval_")
            walk = {"shipped": 3.0, "rl": 3.3}[drafter]
            rows = [{"drafter": drafter, "block_size": b, "walk_accept_length": walk - (b == 6) * 0.2} for b in (8, 6)]
            (self.work / f"{name}.json").write_text(json.dumps(rows))
        elif name.startswith("bench_"):
            drafter, precisions = {"gate": ("shipped", ["bf16"]), "rl": ("rl", ["bf16", "fp8"])}.get(
                name.removeprefix("bench_"), ("shipped", ["fp8"])
            )
            rows = [
                {
                    "drafter": drafter,
                    "block_size": b,
                    "draft_precision": p,
                    "accept_length": engine_accept + 1,
                    "accept_length_after_first": engine_accept,
                    "output_tok_per_s": 100
                    * (1.1 if b == 6 else 1)
                    * (1.05 if p == "fp8" else 1)
                    * (1.2 if drafter == "rl" else 1),
                }
                for b in (8, 6)
                for p in precisions
            ]
            (self.work / name).mkdir()
            (self.work / name / "bench.json").write_text(json.dumps(rows))
        elif name == "train":
            (self.work / "logs" / "train.log").write_text('{"epoch": 1, "step": 2, "of": 2, "loss": 1.0}\n')

    return run


@pytest.mark.parametrize("engine_accept, passed", [(3.05, True), (3.5, False)])
def test_pipeline_runs_every_stage_only_after_the_parity_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, engine_accept: float, passed: bool
) -> None:
    rollouts, work = tmp_path / "rollouts", tmp_path / "work"
    rollouts.mkdir()
    (rollouts / "summary.json").write_text(json.dumps({"xids": ["1063329"], "chosen_task_mix": {"t": 1}}))
    monkeypatch.setattr(run_pipeline._Stages, "run", _fake_stage(engine_accept))
    args = ["--rollouts", str(rollouts), "--work", str(work), "--hf-checkpoint", str(tmp_path / "ckpt")]
    if passed:
        run_pipeline.main(args)
    else:
        with pytest.raises(SystemExit, match="parity gate failed"):
            run_pipeline.main(args)
    summary = json.loads((work / "summary.json").read_text())
    assert summary["gate"]["passed"] is passed and summary["rollouts"] == {"xids": ["1063329"]}
    gate_stages = ["extract_heldout", "eval_shipped", "bench_gate"]
    later_stages = ["extract_train", "train", "eval_rl", "bench_rl", "bench_shipped_fp8"]
    assert list(summary["elapsed_s"]) == gate_stages + later_stages * passed
    if passed:
        comparisons = summary["comparisons"]
        assert comparisons["offline_walk_accept_rl_vs_shipped"]["8"] == pytest.approx(0.1)
        assert comparisons["engine_tok_per_s_block6_vs_block8"]["rl_fp8"] == pytest.approx(0.1)
        assert comparisons["engine_tok_per_s_rl_fp8_block6_vs_shipped_bf16_block8"] == pytest.approx(
            1.1 * 1.05 * 1.2 - 1
        )
        assert len(summary["engine"]) == 8 and summary["train"]["last"]["step"] == 2


def test_bench_replays_each_turn_prompt_and_counts_accept_length_after_the_first_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(200, json={"meta_info": {"completion_tokens": 5, "spec_verify_ct": 2}})

    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        httpx, "AsyncClient", lambda **kwargs: real_client(transport=httpx.MockTransport(handler), **kwargs)
    )
    rollout = {"tokens": list(range(10)), "loss_mask": [0, 0, 1, 1, 0, 0, 1, 1, 1, 0]}
    args = SimpleNamespace(
        concurrency=4, temperature=1.0, top_p=0.97, top_k=1024, max_new_tokens=8192, lora_path=None, timeout=10
    )
    metrics = asyncio.run(bench_engine.replay("http://engine", [rollout], args))
    assert [request["input_ids"] for request in requests] == [[0, 1], [0, 1, 2, 3, 4, 5]]
    assert requests[0]["sampling_params"] == {"temperature": 1.0, "top_p": 0.97, "top_k": 1024, "max_new_tokens": 8192}
    assert metrics["accept_length"] == 10 / 4 and metrics["accept_length_after_first"] == 8 / 4


def test_draft_probe_counts_the_draft_parameters_per_dtype(tmp_path: Path) -> None:
    class DFlashDraftModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc = torch.nn.Linear(4, 2, bias=False)
            self.weight = torch.nn.Parameter(torch.zeros(3, dtype=torch.float8_e4m3fn), requires_grad=False)

    model = DFlashDraftModel()
    path = tmp_path / "dtypes.json"
    assert draft_probe.record_draft_dtypes({"path": str(path)}) is None
    probe = json.loads(path.read_text())
    assert probe["numel_by_dtype"] == {"torch.float32": 8, "torch.float8_e4m3fn": 3}
    assert probe["modules"] == {"": ["torch.float8_e4m3fn", "NoneType"], "fc": ["torch.float32", "NoneType"]}
    del model
