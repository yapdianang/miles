from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

PATCHER_PATH = Path(__file__).parents[4] / "docker/compat/patch_sglang_dflash_fused_value_scale.py"


def _load_patcher():
    spec = spec_from_file_location("patch_sglang_dflash_fused_value_scale", PATCHER_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fused_context_values_get_the_draft_value_scale_once(tmp_path: Path) -> None:
    patcher = _load_patcher()
    worker = tmp_path / "dflash_worker_v2.py"
    worker.write_text("class Worker:\n    def append(self):\n" + patcher.BEFORE + "            return attn\n")

    patcher.patch_worker(worker)
    once = worker.read_text()
    patcher.patch_worker(worker)

    assert worker.read_text() == once
    assert once.count("cache_v = cache_v * value_scale") == 1
    assert "self.draft_model.layers[layer_idx].self_attn.v_scale" in once
