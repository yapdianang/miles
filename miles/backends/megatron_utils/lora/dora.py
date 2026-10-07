"""Multi-LoRA DoRA: per-slot row magnitudes ``g`` on the adapted weight, ``W = g V / ||V||_row``, ``V = W0 + s B A``.

The row norm stays attached, so ``dL/dg`` and ``dL/dV`` are Muown's gradients (gist 869da14a, Prop. 6). It is computed
in closed form, ``||V_i||^2 = ||W0_i||^2 + 2s <B_i, (W0 A^T)_i> + s^2 B_i (A A^T) B_i^T``, without an ``m x n`` tensor.
The output is scaled as ``y + y * (d - 1)`` with ``d = g / ||V||`` and ``d - 1`` in the activation dtype, which keeps
``d`` exact near 1; the rollout engine applies the same ``d - 1`` (``lora_dora_delta``) after its LoRA output.
"""

from contextlib import contextmanager
from dataclasses import dataclass

import torch
import torch.distributed.nn.functional as dist_fn
from megatron.bridge.peft.multi_lora import MultiLoRA
from megatron.bridge.peft.multi_lora_layers import MultiLoRALinear, _narrow_token_counts_to_window
from megatron.core import parallel_state
from megatron.core.tensor_parallel import (
    copy_to_tensor_model_parallel_region,
    gather_from_sequence_parallel_region,
    reduce_from_tensor_model_parallel_region,
)
from megatron.core.tensor_parallel.layers import set_tensor_model_parallel_attributes
from megatron.core.utils import make_sharded_tensor_for_checkpoint, make_tp_sharded_tensor_for_checkpoint

from miles.backends.megatron_utils.fp32_param_utils import mark_param_dtype

DORA_DELTA_SUFFIX = ".lora_dora_delta"


class MultiDoRALinear(MultiLoRALinear):
    """``MultiLoRALinear`` whose slots also scale output rows by ``g / ||W0 + s B A||``.

    ``adapters[i].weight_magnitude`` holds slot ``i``'s ``g`` in fp32: this rank's output rows for a column-parallel
    base, every output row (replicated over TP) for a row-parallel one. ``init_adapter_slot`` sets it to ``||V||``.
    """

    def init_dora(self) -> None:
        if self.replicate_adapter or self._external_output_reduce or not self.base_linear_is_parallel:
            raise NotImplementedError(
                f"multi-LoRA DoRA needs a column- or row-parallel base, not {self.base_linear_name}"
            )
        weight = self.to_wrap.weight
        for adapter in self.adapters:
            adapter.weight_magnitude = torch.nn.Parameter(
                torch.ones(weight.shape[0], dtype=torch.float32, device=weight.device)
            )
            # Float16Module casts every parameter to bf16; d = g / ||V|| needs g in fp32
            mark_param_dtype(adapter.weight_magnitude, torch.float32)
            if not self.input_is_parallel:
                set_tensor_model_parallel_attributes(adapter.weight_magnitude, is_parallel=True, dim=0, stride=1)
        self._w0_sq = None

    def init_adapter_slot(self, idx: int, rank: int, alpha: float) -> None:
        super().init_adapter_slot(idx, rank, alpha)
        with torch.no_grad():
            self._w0_sq = self.to_wrap.weight.float().square().sum(dim=1)
            if self.input_is_parallel:
                self._w0_sq = self._tp_sum(self._w0_sq, split=False)
            self.adapters[idx].weight_magnitude.copy_(self.row_norm(idx))

    def forward(self, x: torch.Tensor, *args, **kwargs):
        output, bias = super().forward(x, *args, **kwargs)
        if not self._adapter_enabled:
            return output, bias
        return self.scale_rows(output), bias

    def scale_rows(self, output: torch.Tensor) -> torch.Tensor:
        """``W0 x + s B A x`` -> ``d * (W0 x + s B A x)`` per token, with each token's slot ``d``."""
        rows = output.reshape(-1, output.shape[-1])
        batch_splits = token_splits = self.tokens_per_adapter_splits
        if self._tokens_split():
            start = parallel_state.get_tensor_model_parallel_rank() * rows.shape[0]
            token_splits = _narrow_token_counts_to_window(token_splits, start, rows.shape[0])
        pieces = []
        # Decide on the whole batch's counts: row_norm is collective even where this rank's window has no tokens
        for idx, (piece, batch_tokens) in enumerate(zip(rows.split(token_splits), batch_splits, strict=True)):
            if batch_tokens:
                magnitude = self.adapters[idx].weight_magnitude
                if self._tokens_split():
                    # Each TP rank scales its own tokens with the replicated g, so g's gradient sums over TP
                    magnitude = copy_to_tensor_model_parallel_region(magnitude)
                piece = piece + piece * (magnitude / self.row_norm(idx) - 1).to(piece.dtype)
            pieces.append(piece)
        return torch.cat(pieces).reshape(output.shape)

    def dora_delta(self, idx: int) -> torch.Tensor:
        """Slot ``idx``'s ``d - 1`` in the activation dtype, laid out like this rank's ``linear_out`` rows."""
        with torch.no_grad():
            delta = (self.adapters[idx].weight_magnitude / self.row_norm(idx) - 1).to(self.to_wrap.weight.dtype)
        if self.input_is_parallel:
            delta = delta.chunk(parallel_state.get_tensor_model_parallel_world_size())[
                parallel_state.get_tensor_model_parallel_rank()
            ]
        return delta

    def row_norm(self, idx: int) -> torch.Tensor:
        """``||W0 + s B A||`` per output row of this rank's ``g``, in fp32."""
        adapter = self.adapters[idx]
        scale = (self.alpha_values[idx] / self.rank_values[idx]).float()
        w0, a, b = self.to_wrap.weight, adapter.linear_in.weight, adapter.linear_out.weight
        # Column-parallel ranks use the reduced terms for different rows, sequence-parallel ranks for different tokens
        split = not self.input_is_parallel or self._tokens_split()
        if self.input_is_parallel:
            # W0 and A hold this rank's input columns; B holds this rank's slice of the output rows
            a = a.float()
            cross = self._tp_sum(_FrozenTimesTransposed.apply(w0, a), split=split)
            gram = self._tp_sum(a @ a.T, split=split)
            b = self._tp_gather_rows(b, split=split)
        else:
            # W0 and B hold this rank's output rows; A holds this rank's rank rows
            a = self._tp_gather_rows(a, split=split).float()
            cross = _FrozenTimesTransposed.apply(w0, a)
            gram = a @ a.T
        b = b.float()
        sq = self._w0_sq + 2 * scale * (b * cross).sum(dim=1) + scale**2 * ((b @ gram) * b).sum(dim=1)
        return sq.clamp_min(torch.finfo(torch.float32).tiny).sqrt()

    def _tokens_split(self) -> bool:
        """A row-parallel output under sequence parallelism holds this TP rank's tokens only."""
        tp = parallel_state.get_tensor_model_parallel_world_size()
        return self.input_is_parallel and not self.disable_sequence_parallel_comm and tp > 1

    @staticmethod
    def _tp_sum(x: torch.Tensor, *, split: bool) -> torch.Tensor:
        """Sum over TP; when TP ranks use the sum for different outputs, their gradients sum too."""
        if parallel_state.get_tensor_model_parallel_world_size() == 1:
            return x
        if split:
            return dist_fn.all_reduce(x, group=parallel_state.get_tensor_model_parallel_group())
        return reduce_from_tensor_model_parallel_region(x)

    @staticmethod
    def _tp_gather_rows(x: torch.Tensor, *, split: bool) -> torch.Tensor:
        if parallel_state.get_tensor_model_parallel_world_size() == 1:
            return x
        return gather_from_sequence_parallel_region(x, tensor_parallel_output_grad=split)

    def sharded_state_dict(self, prefix: str = "", sharded_offsets: tuple = (), metadata: dict | None = None) -> dict:
        sharded = super().sharded_state_dict(prefix, sharded_offsets, metadata)
        for i, adapter in enumerate(self.adapters):
            key = f"{prefix}adapters.{i}.weight_magnitude"
            if self.input_is_parallel:
                sharded[key] = make_sharded_tensor_for_checkpoint(
                    adapter.weight_magnitude, key, prepend_offsets=sharded_offsets
                )
            else:
                sharded[key] = make_tp_sharded_tensor_for_checkpoint(
                    adapter.weight_magnitude, key, tp_axis=0, prepend_offsets=sharded_offsets
                )
        return sharded


class _FrozenTimesTransposed(torch.autograd.Function):
    """``W0 A^T`` in fp32 for a frozen ``W0``, saving ``W0`` as stored rather than its fp32 copy."""

    @staticmethod
    def forward(ctx, w0: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(w0)
        return w0.float() @ a.T

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        (w0,) = ctx.saved_tensors
        return None, grad.T @ w0.float()


@dataclass
class MultiDoRA(MultiLoRA):
    """``MultiLoRA`` that builds ``MultiDoRALinear`` for dense linears; routed experts keep LoRA."""

    def transform(self, module, name=None, prefix=None):
        transformed = super().transform(module, name, prefix)
        if type(transformed) is MultiLoRALinear:
            # Upgrade in place rather than rebuild, so the adapters keep their construction-time init
            transformed.__class__ = MultiDoRALinear
            transformed.init_dora()
        return transformed


def iter_dora_layers(model):
    for chunk in model if isinstance(model, (list, tuple)) else [model]:
        for module in chunk.modules():
            if isinstance(module, MultiDoRALinear):
                yield module


@contextmanager
def dora_deltas_as_lora_b(model, slot: int):
    """Put each layer's ``d - 1`` in column 0 of slot ``slot``'s LoRA-B (zeros elsewhere) for the duration.

    The adapter export then splits, reorders and TP-gathers ``d - 1`` exactly like LoRA-B rows.
    """
    saved = []
    for layer in iter_dora_layers(model):
        weight = layer.adapters[slot].linear_out.weight
        column = torch.zeros_like(weight)
        column[:, 0] = layer.dora_delta(slot)
        saved.append((weight, weight.data))
        weight.data = column
    try:
        yield
    finally:
        for weight, data in saved:
            weight.data = data
