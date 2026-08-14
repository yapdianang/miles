from examples.tinker_backend import run_tinker_backend


def test_static_batch_mode_omits_packed_sequence_args(monkeypatch):
    captured = {}

    def capture_execute_train(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(run_tinker_backend.U, "execute_train", capture_execute_train)

    args = run_tinker_backend.ScriptArgs(use_dynamic_batch_size=False)
    run_tinker_backend._serve(args, service=True)

    train_args = captured["train_args"]
    assert "--use-dynamic-batch-size" not in train_args
    assert "--max-tokens-per-gpu" not in train_args


def test_dynamic_batch_mode_remains_the_default(monkeypatch):
    captured = {}

    def capture_execute_train(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(run_tinker_backend.U, "execute_train", capture_execute_train)

    args = run_tinker_backend.ScriptArgs(max_tokens_per_gpu=8192, sglang_mem_fraction_static=0.5)
    run_tinker_backend._serve(args, service=True)

    train_args = captured["train_args"]
    assert "--use-dynamic-batch-size" in train_args
    assert "--max-tokens-per-gpu 8192" in train_args
    assert "--sglang-mem-fraction-static 0.5" in train_args


def test_service_forwards_model_type(monkeypatch):
    captured = {}

    def capture_execute_train(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(run_tinker_backend.U, "execute_train", capture_execute_train)

    args = run_tinker_backend.ScriptArgs(megatron_model_type="qwen3.8-27B")
    run_tinker_backend._serve(args, service=True)

    assert captured["megatron_model_type"] == "qwen3.8-27B"
