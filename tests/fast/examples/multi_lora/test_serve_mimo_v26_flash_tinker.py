"""prepare builds only the missing MiMo checkpoints from the pinned official snapshot."""

from types import SimpleNamespace

import pytest

from examples.multi_lora import serve_mimo_v26_flash_tinker as launcher
from tests.fast.utils.command_recorder import record_commands


@pytest.mark.parametrize(
    "converted, conversions",
    [
        ((), ["w4a16 --device cuda --keep-quant --bf16-linears", "bf16 --device cuda"]),
        (("w4a16",), ["bf16 --device cuda"]),
        (("w4a16", "bf16"), []),
    ],
)
def test_prepare_converts_only_what_is_missing(monkeypatch, tmp_path, converted, conversions):
    commands = record_commands(monkeypatch)
    for name in converted:
        (tmp_path / name).mkdir()
        (tmp_path / name / "model.safetensors.index.json").write_text("{}")
    args = launcher.ScriptArgs(
        hf_checkpoint=str(tmp_path / "w4a16"), ref_load=str(tmp_path / "bf16"), hf_hub_cache=str(tmp_path / "hub")
    )

    launcher._prepare(args)

    if not conversions:
        assert commands == []
        return
    revision = args.base_model_revision
    snapshot = f"{tmp_path}/hub/models--XiaomiMiMo--MiMo-V2.6-Flash-RL/snapshots/{revision}"
    assert commands[0] == (
        f"HF_HUB_OFFLINE=0 hf download XiaomiMiMo/MiMo-V2.6-Flash-RL --revision {revision} --cache-dir {tmp_path}/hub"
    )
    assert [
        command.split(f"--model-dir {snapshot} --save-dir {tmp_path}/")[1] for command in commands[1:]
    ] == conversions
    assert all(
        command.startswith("python ") and "tools/convert_mimo_v2_to_bf16.py" in command for command in commands[1:]
    )


@pytest.mark.parametrize("support_bitmaps", [True, False])
def test_packed_ids_reach_engines_and_gateway_only_without_bitmaps(monkeypatch, tmp_path, support_bitmaps):
    requests = []
    args = launcher.ScriptArgs(save_dir=str(tmp_path), support_bitmaps=support_bitmaps)
    monkeypatch.setattr(
        args, "create_backend", lambda: SimpleNamespace(execute_train=lambda **kwargs: requests.append(kwargs))
    )

    launcher._serve(args)

    env = requests[0]["extra_env_vars"]
    assert ("SGLANG_SAMPLING_MASK_PACKED_IDS" in env) is not support_bitmaps
    assert "--tinker-sampling-support-replay" in requests[0]["train_args"]
