"""MiMo-V2.6 DFlash drafter in PyTorch, matching SGLang's DFLASH draft forward for training.

Reference: sglang-miles f19dcfb, ``srt/models/dflash.py`` and ``srt/speculative/dflash_worker_v2.py``. A draft
block is the last verified token at position ``a`` followed by ``block_size - 1`` mask tokens, embedded with the
target embedding (the mask row replaced by ``mask_embedding.pt``). The context is the target residual stream after
each ``target_layer_ids`` layer at every position before ``a``, concatenated, projected by ``fc`` and
``hidden_norm``, and used unnormalized as keys and values in every draft layer. Attention is bidirectional within
the block, sees context positions ``>= a + k - (sliding_window - 1)`` from block slot ``k``, adds a per-head sink
logit, and scales values by ``attention_value_scale``. Slot ``k >= 1`` predicts the token at ``a + k`` through the
target ``lm_head``.
"""

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from torch import nn

DRAFT_WEIGHTS = "dflash_draft_model.safetensors"
MASK_EMBEDDING = "mask_embedding.pt"


@dataclass(frozen=True)
class DrafterConfig:
    hidden_size: int
    intermediate_size: int
    num_layers: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    rotary_dim: int
    rope_theta: float
    rms_norm_eps: float
    sliding_window: int
    value_scale: float
    target_layer_ids: tuple[int, ...]
    mask_token_id: int
    block_size: int
    loss_decay_gamma: float
    num_anchors: int

    @classmethod
    def from_dir(cls, path: Path) -> "DrafterConfig":
        config = json.loads((path / "config.json").read_text())
        dflash = config["dflash_config"]
        # Only the MiMo layout is implemented: bidirectional sliding-window layers with sinks and a value scale.
        if set(config["layer_types"]) != {"sliding_attention"} or config.get("is_causal") is not False:
            raise ValueError(f"{path}: expected bidirectional sliding_attention layers (MiMo DFlash)")
        if dflash.get("attention_sink_bias") is not True or "attention_value_scale" not in dflash:
            raise ValueError(f"{path}: expected attention_sink_bias and attention_value_scale (MiMo DFlash)")
        return cls(
            hidden_size=config["hidden_size"],
            intermediate_size=config["intermediate_size"],
            num_layers=config["num_hidden_layers"],
            num_heads=config["num_attention_heads"],
            num_kv_heads=config["num_key_value_heads"],
            head_dim=config["head_dim"],
            rotary_dim=int(config["head_dim"] * config.get("partial_rotary_factor", 1.0)),
            rope_theta=config["rope_theta"],
            rms_norm_eps=config["rms_norm_eps"],
            sliding_window=config["sliding_window"],
            value_scale=dflash["attention_value_scale"],
            target_layer_ids=tuple(dflash["target_layer_ids"]),
            mask_token_id=dflash["mask_token_id"],
            block_size=dflash["block_size"],
            loss_decay_gamma=dflash["loss_decay_gamma"],
            num_anchors=dflash["num_anchors"],
        )

    @property
    def target_hidden_size(self) -> int:
        return len(self.target_layer_ids) * self.hidden_size


class _RMSNorm(nn.Module):
    def __init__(self, size: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(x.float(), (x.shape[-1],), self.weight.float(), self.eps).to(x.dtype)


def _rope(x: torch.Tensor, positions: torch.Tensor, *, rotary_dim: int, theta: float) -> torch.Tensor:
    """Neox rotary embedding of the first ``rotary_dim`` channels of ``x`` [..., heads, head_dim]."""
    inv_freq = 1.0 / theta ** (torch.arange(0, rotary_dim, 2, device=x.device, dtype=torch.float32) / rotary_dim)
    freqs = positions.float()[..., None, None] * inv_freq
    cos, sin = freqs.cos(), freqs.sin()
    x1, x2 = x[..., :rotary_dim].float().chunk(2, dim=-1)
    rotated = torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1).to(x.dtype)
    return torch.cat([rotated, x[..., rotary_dim:]], dim=-1)


class _Attention(nn.Module):
    def __init__(self, config: DrafterConfig) -> None:
        super().__init__()
        self.config = config
        heads, kv_heads, head_dim = config.num_heads, config.num_kv_heads, config.head_dim
        self.q_proj = nn.Linear(config.hidden_size, heads * head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(heads * head_dim, config.hidden_size, bias=False)
        self.q_norm = _RMSNorm(head_dim, config.rms_norm_eps)
        self.k_norm = _RMSNorm(head_dim, config.rms_norm_eps)
        self.attention_sink_bias = nn.Parameter(torch.zeros(heads))

    def project_kv(self, hidden: torch.Tensor, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        config = self.config
        k = self.k_norm(self.k_proj(hidden).unflatten(-1, (config.num_kv_heads, config.head_dim)))
        k = _rope(k, positions, rotary_dim=config.rotary_dim, theta=config.rope_theta)
        v = self.v_proj(hidden).unflatten(-1, (config.num_kv_heads, config.head_dim)) * config.value_scale
        return k, v

    def forward(
        self,
        x: torch.Tensor,
        block_positions: torch.Tensor,
        ctx_k: torch.Tensor,
        ctx_v: torch.Tensor,
        ctx_visible: torch.Tensor,
    ) -> torch.Tensor:
        """x [N, B, hidden], block_positions [N, B], ctx_k / ctx_v [C, kv_heads, head_dim], ctx_visible [N, B, C]."""
        config = self.config
        num_blocks, block_size = x.shape[:2]
        groups = config.num_heads // config.num_kv_heads
        q = self.q_norm(self.q_proj(x).unflatten(-1, (config.num_heads, config.head_dim)))
        q = _rope(q, block_positions, rotary_dim=config.rotary_dim, theta=config.rope_theta)
        q = q.unflatten(2, (config.num_kv_heads, groups)) * config.head_dim**-0.5
        k, v = self.project_kv(x, block_positions)
        ctx_scores = torch.einsum("nbkgd,ckd->nkgbc", q, ctx_k)
        ctx_scores = ctx_scores.masked_fill(~ctx_visible[:, None, None], float("-inf"))
        block_scores = torch.einsum("nbkgd,njkd->nkgbj", q, k)
        sinks = self.attention_sink_bias.view(config.num_kv_heads, groups, 1, 1)
        sinks = sinks.to(block_scores.dtype).expand(num_blocks, -1, -1, block_size, 1)
        probs = torch.cat([ctx_scores, block_scores, sinks], dim=-1).float().softmax(dim=-1).to(v.dtype)
        num_ctx = ctx_k.shape[0]
        out = torch.einsum("nkgbc,ckd->nbkgd", probs[..., :num_ctx], ctx_v)
        out = out + torch.einsum("nkgbj,njkd->nbkgd", probs[..., num_ctx:-1], v)
        return self.o_proj(out.flatten(2))


class _MLP(nn.Module):
    def __init__(self, config: DrafterConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class _DecoderLayer(nn.Module):
    def __init__(self, config: DrafterConfig) -> None:
        super().__init__()
        self.input_layernorm = _RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = _Attention(config)
        self.post_attention_layernorm = _RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = _MLP(config)

    def forward(self, x, block_positions, ctx, ctx_positions, ctx_visible) -> torch.Tensor:
        ctx_k, ctx_v = self.self_attn.project_kv(ctx, ctx_positions)
        x = x + self.self_attn(self.input_layernorm(x), block_positions, ctx_k, ctx_v, ctx_visible)
        return x + self.mlp(self.post_attention_layernorm(x))


class DFlashDrafter(nn.Module):
    """Parameter names match the checkpoint, except ``mask_embedding`` (stored in mask_embedding.pt)."""

    def __init__(self, config: DrafterConfig) -> None:
        super().__init__()
        self.config = config
        self.fc = nn.Linear(config.target_hidden_size, config.hidden_size, bias=False)
        self.hidden_norm = _RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.layers = nn.ModuleList(_DecoderLayer(config) for _ in range(config.num_layers))
        self.norm = _RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mask_embedding = nn.Parameter(torch.zeros(config.hidden_size))

    def forward(
        self,
        block_ids: torch.Tensor,
        block_positions: torch.Tensor,
        target_hidden: torch.Tensor,
        ctx_positions: torch.Tensor,
        ctx_visible: torch.Tensor,
        embed_weight: torch.Tensor,
    ) -> torch.Tensor:
        """Draft hidden states [N, B, hidden] of blocks ``block_ids`` [N, B] over context ``target_hidden`` [C, F]."""
        x = F.embedding(block_ids, embed_weight)
        is_mask = (block_ids == self.config.mask_token_id)[..., None]
        x = torch.where(is_mask, self.mask_embedding.to(x.dtype), x)
        ctx = self.hidden_norm(self.fc(target_hidden.to(x.dtype)))
        for layer in self.layers:
            x = layer(x, block_positions, ctx, ctx_positions, ctx_visible)
        return self.norm(x)


def decay_weights(block_size: int, gamma: float, device: torch.device) -> torch.Tensor:
    """Loss weight of slots 1..block_size-1: exp(-(k - 1) / gamma)."""
    return torch.exp(-torch.arange(block_size - 1, device=device, dtype=torch.float32) / gamma)


def block_loss(
    draft_hidden: torch.Tensor, labels: torch.Tensor, label_mask: torch.Tensor, lm_head: torch.Tensor, gamma: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Decay-weighted cross entropy sum, its weight sum, and each block's accepted draft count (greedy drafts)."""
    logits = F.linear(draft_hidden[:, 1:], lm_head).float()
    ce = F.cross_entropy(logits.flatten(0, 1), labels.flatten(), reduction="none").view_as(labels)
    weights = label_mask * decay_weights(labels.shape[1] + 1, gamma, labels.device)
    accepted = ((logits.argmax(dim=-1) == labels) & label_mask).int().cumprod(dim=-1).sum(dim=-1)
    return (ce * weights).sum(), weights.sum(), accepted


def load_drafter(path: Path) -> DFlashDrafter:
    config = DrafterConfig.from_dir(path)
    drafter = DFlashDrafter(config)
    index = json.loads((path / "model.safetensors.index.json").read_text())
    state = {}
    for name in sorted(set(index["weight_map"].values())):
        state.update(load_file(path / name))
    mask = torch.load(path / MASK_EMBEDDING, map_location="cpu", weights_only=True)
    if mask["mask_token_id"] != config.mask_token_id or mask.get("per_position"):
        raise ValueError(f"{path / MASK_EMBEDDING}: expected one embedding for mask_token_id {config.mask_token_id}")
    state["mask_embedding"] = mask["embedding"]
    drafter.load_state_dict({name: tensor.float() for name, tensor in state.items()}, strict=True)
    return drafter


def save_drafter(drafter: DFlashDrafter, *, source: Path, out: Path) -> None:
    """Write a drafter directory SGLang loads with --speculative-draft-model-path: source's config and code."""
    out.mkdir(parents=True, exist_ok=True)
    for path in source.iterdir():
        if path.is_file() and path.name not in (DRAFT_WEIGHTS, MASK_EMBEDDING, "model.safetensors.index.json"):
            shutil.copy2(path, out / path.name)
    state = {
        name: tensor.detach().to(torch.bfloat16).cpu().contiguous()
        for name, tensor in drafter.state_dict().items()
        if name != "mask_embedding"
    }
    save_file(state, out / DRAFT_WEIGHTS, metadata={"format": "pt"})
    index = {
        "metadata": {"total_size": sum(tensor.numel() * tensor.element_size() for tensor in state.values())},
        "weight_map": dict.fromkeys(sorted(state), DRAFT_WEIGHTS),
    }
    (out / "model.safetensors.index.json").write_text(json.dumps(index, indent=2) + "\n")
    embedding = drafter.mask_embedding.detach().to(torch.bfloat16).cpu()
    torch.save({"mask_token_id": drafter.config.mask_token_id, "embedding": embedding}, out / MASK_EMBEDDING)


def load_target_head(path: Path, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """The target's frozen ``model.embed_tokens.weight`` and ``lm_head.weight``."""
    weight_map = json.loads((path / "model.safetensors.index.json").read_text())["weight_map"]
    tensors = []
    for name in ("model.embed_tokens.weight", "lm_head.weight"):
        with safe_open(path / weight_map[name], framework="pt") as handle:
            tensor = handle.get_tensor(name)
        if not tensor.is_floating_point() or tensor.element_size() < 2:
            raise ValueError(f"{path}: {name} is {tensor.dtype}; pass a checkpoint with a BF16 {name}")
        tensors.append(tensor.to(device=device, dtype=dtype))
    return tensors[0], tensors[1]
