"""The DFlash sampling-mask patch applies to the installed SGLang once, and the patched code passes CPU checks."""

import importlib.util
import os
import shutil
import subprocess
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest

PATCHER_PATH = Path(__file__).parents[4] / "docker/compat/patch_sglang_dflash_sampling_mask.py"
CHECKS_PATH = Path(__file__).with_name("dflash_sampling_mask_checks.py")
SGLANG_SPEC = importlib.util.find_spec("sglang")

pytestmark = pytest.mark.skipif(SGLANG_SPEC is None, reason="needs the SGLang sources the patch targets")


def _load_patcher():
    spec = spec_from_file_location("patch_sglang_dflash_sampling_mask", PATCHER_PATH)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def patched_sglang(tmp_path_factory) -> Path:
    """A patched copy of the installed sglang package; the installed one stays untouched."""
    root = tmp_path_factory.mktemp("patched")
    source = Path(next(iter(SGLANG_SPEC.submodule_search_locations)))
    shutil.copytree(source, root / "sglang", ignore=shutil.ignore_patterns("__pycache__"))
    _load_patcher().main(root / "sglang/srt", sglang_version=None)
    return root


def test_the_patch_is_idempotent_and_marks_sglang(patched_sglang):
    patcher = _load_patcher()
    srt = patched_sglang / "sglang/srt"
    files = sorted(srt.rglob("*.py"))
    before = [path.read_text() for path in files]
    patcher.main(srt, sglang_version=None)
    assert [path.read_text() for path in files] == before
    assert patcher.MARKER in (srt / "speculative/dflash_utils.py").read_text()


def test_another_sglang_build_is_refused(tmp_path):
    with pytest.raises(RuntimeError, match="targets SGLang"):
        _load_patcher().main(tmp_path, sglang_version="0.0.0+other")


def test_patched_dflash_returns_the_support_of_every_committed_token(patched_sglang):
    environment = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join([str(patched_sglang), os.environ.get("PYTHONPATH", "")]),
    }
    result = subprocess.run(
        [sys.executable, str(CHECKS_PATH)], env=environment, capture_output=True, text=True, timeout=600
    )
    report = "\n".join(line for line in result.stdout.splitlines() if line.startswith(("PASS", "FAIL")))
    assert result.returncode == 0, report + "\n" + result.stderr[-4000:]
    assert report.count("PASS") == 8, report
