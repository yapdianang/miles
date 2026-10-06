import shlex
import sys
from types import SimpleNamespace

import pytest

from examples.multi_lora import serve_mimo_v26_flash_tinker as launcher


def _fake_sglang(root, *, patched: bool) -> None:
    speculative = root / "sglang/srt/speculative"
    speculative.mkdir(parents=True)
    (root / "sglang/__init__.py").write_text("")
    (speculative / "dflash_utils.py").write_text("DFLASH_SAMPLING_MASK_PATCH = 1\n" if patched else "")


@pytest.mark.parametrize("patched", [False, True])
def test_dflash_takes_sampling_support_replay_only_from_a_patched_sglang(tmp_path, monkeypatch, patched):
    _fake_sglang(tmp_path, patched=patched)
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, "sglang", raising=False)
    if not patched:
        with pytest.raises(ValueError, match="no sampling supports under DFlash"):
            launcher.ScriptArgs(sampling_support_replay=True, dflash=True)
        return
    args = launcher.ScriptArgs(sampling_support_replay=True, dflash=True)
    requests = []
    monkeypatch.setattr(
        args, "create_backend", lambda: SimpleNamespace(execute_train=lambda **kwargs: requests.append(kwargs))
    )
    launcher._serve(args)
    argv = shlex.split(requests[0]["train_args"])
    assert "--tinker-sampling-support-replay" in argv
    assert argv[argv.index("--sglang-speculative-algorithm") + 1] == "DFLASH"


def test_replay_without_dflash_needs_no_patch(tmp_path, monkeypatch):
    _fake_sglang(tmp_path, patched=False)
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, "sglang", raising=False)
    assert launcher.ScriptArgs(sampling_support_replay=True, dflash=False).sampling_support_replay
