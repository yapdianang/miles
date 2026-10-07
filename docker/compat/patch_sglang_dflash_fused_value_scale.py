"""Scale DFlash context values in SGLang's fused KV materialization.

DFlashAttention multiplies values by the draft config's ``attention_value_scale`` (MiMo-V2.6: 0.612) in its
forward and in ``kv_proj_only``, the sequential path that writes context K/V into the draft cache. The fused path
(``FusedKVMaterializeHelper``, enabled on CUDA) projects, normalizes and rotates the context in one kernel but
writes values unscaled, and its eligibility check reads the attention backend's KV-cache ``v_scale`` instead.
Context values then reach the draft 1/0.612 times too large: on held-out MiMo Tau3 rollouts the shipped drafter
accepts 2.93 tokens per step at block 8 instead of the 3.09 its trained forward gives. Scale them per layer when
the fused path writes them.
"""

from pathlib import Path

SGLANG_WORKER = Path("/sgl-workspace/sglang/python/sglang/srt/speculative/dflash_worker_v2.py")

BEFORE = """\
        def _write_layer_kv(
            layer_idx: int,
            cache_k: torch.Tensor,
            cache_v: torch.Tensor,
        ) -> None:
            attn = self.draft_model.layers[layer_idx].self_attn.attn
"""
AFTER = """\
        def _write_layer_kv(
            layer_idx: int,
            cache_k: torch.Tensor,
            cache_v: torch.Tensor,
        ) -> None:
            # kv_proj_only scales values by attention_value_scale; the fused projection does not.
            value_scale = self.draft_model.layers[layer_idx].self_attn.v_scale
            if value_scale is not None:
                cache_v = cache_v * value_scale
            attn = self.draft_model.layers[layer_idx].self_attn.attn
"""


def patch_worker(path: Path = SGLANG_WORKER) -> None:
    source = path.read_text()
    if AFTER in source:
        return
    if BEFORE not in source:
        raise RuntimeError(f"expected source not found in {path}:\n{BEFORE}")
    path.write_text(source.replace(BEFORE, AFTER, 1))


if __name__ == "__main__":
    patch_worker()
