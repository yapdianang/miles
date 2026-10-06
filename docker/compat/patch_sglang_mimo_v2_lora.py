"""Size SGLang's attention LoRA buffers for MiMo-V2.

MiMo-V2 attention has V heads narrower than Q/K heads (128 vs 192), and its SWA
and global layers have different KV head counts (8 vs 4).  SGLang's fused qkv
LoRA layer sizes and slices V with the K shard size, and the default
``get_hidden_dim`` uses one global head layout, so both qkv_proj LoRA-B buffers
are mis-sized.  Patch the qkv LoRA layer to use ``v_proj_shard_size`` and give
``MiMoV2ForCausalLM`` a per-layer ``get_hidden_dim``.

MiMo-V2's audio encoder also has plain ``nn.Linear`` attention projections with the
same leaf names.  The LoRA manager consults the model's ``should_apply_lora``, and
MiMo-V2 limits LoRA to its decoder layers.
"""

from pathlib import Path

SGLANG_SRT = Path("/sgl-workspace/sglang/python/sglang/srt")

LAYERS_PATCHES = (
    (
        """\
        q_proj_shard_size = self.base_layer.q_proj_shard_size
        kv_proj_shard_size = self.base_layer.kv_proj_shard_size
        offsets = [
            0,
            q_proj_shard_size,
            q_proj_shard_size + kv_proj_shard_size,
            q_proj_shard_size + 2 * kv_proj_shard_size,
        ]
""",
        """\
        q_proj_shard_size = self.base_layer.q_proj_shard_size
        kv_proj_shard_size = self.base_layer.kv_proj_shard_size
        v_proj_shard_size = self.base_layer.v_proj_shard_size
        offsets = [
            0,
            q_proj_shard_size,
            q_proj_shard_size + kv_proj_shard_size,
            q_proj_shard_size + kv_proj_shard_size + v_proj_shard_size,
        ]
""",
    ),
    (
        """\
        self.max_qkv_out_dim = max(q_proj_shard_size, kv_proj_shard_size)
""",
        """\
        self.max_qkv_out_dim = max(q_proj_shard_size, kv_proj_shard_size, v_proj_shard_size)
""",
    ),
    (
        """\
        B_v_shard = B[q_size + k_size + kv_start_idx : q_size + k_size + kv_end_idx, :]
""",
        """\
        v_proj_shard_size = base_layer.v_proj_shard_size
        v_start_idx = v_proj_shard_size * kv_shard_id
        B_v_shard = B[q_size + k_size + v_start_idx : q_size + k_size + v_start_idx + v_proj_shard_size, :]
""",
    ),
)

MANAGER_PATCHES = (
    (
        """\
            # The module should be converted if it is included in target_names
            parts = module_name.split(".")
""",
        """\
            should_apply_lora = getattr(self.base_model, "should_apply_lora", None)
            if should_apply_lora is not None and not should_apply_lora(module_name):
                continue

            # The module should be converted if it is included in target_names
            parts = module_name.split(".")
""",
    ),
)

MIMO_ANCHOR = """\
    @property
    def routed_experts_weights_of_layer(self):
        return self._routed_experts_weights_of_layer.value
"""

MIMO_GET_HIDDEN_DIM = """\
    def should_apply_lora(self, module_name: str) -> bool:
        # The audio encoder has plain nn.Linear projections with the decoder's leaf names.
        return module_name.startswith("model.layers.")

    def get_hidden_dim(self, module_name: str, layer_idx: int):
        # SWA and global layers have their own KV heads, and V heads are narrower than Q/K.
        if module_name in ("qkv_proj", "o_proj"):
            attn = self.model.layers[layer_idx].self_attn
            if module_name == "o_proj":
                return attn.total_num_heads * attn.v_head_dim, self.config.hidden_size
            q_size = attn.total_num_heads * attn.head_dim
            kv_size = attn.total_num_kv_heads * (attn.head_dim + attn.v_head_dim)
            return self.config.hidden_size, q_size + kv_size
        from sglang.srt.lora.utils import get_default_hidden_dim

        return get_default_hidden_dim(module_name, self.config, layer_idx)

"""


def patch(path: Path, replacements) -> None:
    source = path.read_text()
    for before, after in replacements:
        if after in source:
            continue
        if before not in source:
            raise RuntimeError(f"expected source not found in {path}:\n{before}")
        source = source.replace(before, after, 1)
    path.write_text(source)


def main(srt: Path = SGLANG_SRT) -> None:
    patch(srt / "lora/layers.py", LAYERS_PATCHES)
    patch(srt / "lora/lora_manager.py", MANAGER_PATCHES)
    patch(srt / "models/mimo_v2.py", ((MIMO_ANCHOR, MIMO_GET_HIDDEN_DIM + MIMO_ANCHOR),))


if __name__ == "__main__":
    main()
