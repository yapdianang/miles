"""Muown for LoRA: Muon on the LoRA tangent space of ``V = W0 + s B A``, ``s = alpha / rank``.

Each step rebalances an adapter's factors to ``(B T, T^-1 A)`` with ``B^T B = A A^T``, which keeps ``V``, moves the
gradients and momenta with the same gauge (``M_B -> M_B T^-T``, ``M_A -> T^T M_A``), then takes whitened Nesterov steps

    dA = -(lr / 2s) P^-1 NS(P^-1 M_A),    dB = -(lr / 2s) NS(M_B Q^-1) Q^-1,

with ``P = (B^T B + lam I)^1/2``, ``Q = (A A^T + lam I)^1/2``, ``lam = damping * largest eigenvalue``, and ``NS``
Megatron's Newton-Schulz scaled by ``extra_scale_factor * sqrt(max(m, n))`` for the adapted ``m x n`` weight.
While ``B A`` has rank r the ``V`` trajectory does not depend on how ``B A`` is factored, and at full rank the
first-order step is Muown's direction step (gist 869da14a2961b226343a398e50c3261f, Thm. 1, 3 and 4).
Importing this module registers ``--optimizer muown``.
"""

import torch
from emerging_optimizers import utils
from emerging_optimizers.orthogonalized_optimizers.muon import get_muon_scale_factor
from emerging_optimizers.orthogonalized_optimizers.muon_utils import newton_schulz
from megatron.core.optimizer.emerging_optimizers import (
    _EMERGING_OPTIMIZERS,
    EmergingOptimizerEntry,
    TensorParallelMuon,
    _muon_config_to_kwargs,
)
from megatron.core.utils import get_pg_rank, get_pg_size

MUOWN = "muown"


class TensorParallelMuownLoRA(TensorParallelMuon):
    """Whitened Muon steps on LoRA factor pairs.

    ``lora_pairs`` lists ``(linear_in.weight, linear_out.weight, alpha / rank, rank)``. Pairs this optimizer does
    not hold are skipped, and every parameter it holds must belong to one. Tensor-parallel shards are gathered,
    stepped whole on every TP rank, and written back, so each step needs both factors on this rank.
    """

    def __init__(
        self,
        params,
        lora_pairs=(),
        damping: float = 1e-4,
        *,
        num_ns_steps: int = 5,
        coefficient_type: str = "quintic",
        scale_mode: str = "spectral",
        extra_scale_factor: float = 1.0,
        **kwargs,
    ):
        super().__init__(
            params,
            num_ns_steps=num_ns_steps,
            coefficient_type=coefficient_type,
            scale_mode=scale_mode,
            extra_scale_factor=extra_scale_factor,
            **kwargs,
        )
        self.damping = damping
        self._ns_kwargs = {"steps": num_ns_steps, "coefficient_type": coefficient_type}
        self._step_scale = lambda m, n: get_muon_scale_factor(m, n, mode=scale_mode) * extra_scale_factor
        index = {id(p): (g, i) for g, group in enumerate(self.param_groups) for i, p in enumerate(group["params"])}
        # Group positions survive Float16OptimizerWithFloat16Params swapping in fp32 main params
        self._lora_pairs = []
        for a, b, scale, rank in lora_pairs:
            loc_a, loc_b = index.get(id(a)), index.get(id(b))
            if loc_a is None and loc_b is None:
                continue
            if loc_a is None or loc_b is None or loc_a[0] != loc_b[0]:
                raise ValueError(f"LoRA factors {tuple(a.shape)} and {tuple(b.shape)} must share a param group.")
            self._lora_pairs.append((loc_a[0], loc_a[1], loc_b[1], scale, rank))
        if 2 * len(self._lora_pairs) != len(index):
            raise ValueError("Muown-LoRA only updates LoRA factor pairs; route other parameters to Adam.")

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            raise ValueError("closure is not supported")
        for group in self.param_groups:
            if group["weight_decay"] != 0:
                raise NotImplementedError("Muown-LoRA does not decay LoRA factors; send weight_decay 0.")
            self._init_group(group)
        for g, i_a, i_b, scale, rank in self._lora_pairs:
            group = self.param_groups[g]
            a, b = group["params"][i_a], group["params"][i_b]
            assert (a.grad is None) == (b.grad is None), "LoRA factors must both have gradients or neither."
            if a.grad is not None:
                self._step_pair(a, b, scale, rank, group)
        return None

    def _step_pair(self, a: torch.Tensor, b: torch.Tensor, scale: float, rank: int, group: dict) -> None:
        a_full, m_a_full = (self._gather(a, x) for x in (a, self.state[a]["momentum_buffer"]))
        b_full, m_b_full = (self._gather(b, x) for x in (b, self.state[b]["momentum_buffer"]))
        # Rows of A and columns of B past the adapter's rank are zero padding and stay zero
        A, M_A, G_A = a_full[:rank], m_a_full[:rank], self._gather(a, a.grad)[:rank]
        B, M_B, G_B = b_full[:, :rank], m_b_full[:, :rank], self._gather(b, b.grad)[:, :rank]

        G_A, G_B = self._rebalance(A, B, M_A, M_B, G_A, G_B)
        p_inv, q_inv = _inv_sqrt(B, self.damping), _inv_sqrt(A.T, self.damping)
        step_size = -group["lr"] * self._step_scale(B.size(0), A.size(1)) / (2 * scale)
        d_a = step_size * (p_inv @ self._orthogonalize(p_inv @ self._nesterov(M_A, G_A, group)))
        d_b = step_size * (self._orthogonalize(self._nesterov(M_B, G_B, group) @ q_inv) @ q_inv)
        A.add_(d_a)
        B.add_(d_b)
        for p, param_full, momentum_full in ((a, a_full, m_a_full), (b, b_full, m_b_full)):
            p.copy_(self._local(p, param_full))
            self.state[p]["momentum_buffer"].copy_(self._local(p, momentum_full))

    def _rebalance(self, A, B, M_A, M_B, G_A, G_B) -> tuple[torch.Tensor, torch.Tensor]:
        """Balance ``A`` and ``B`` in place, move the momenta with them, and return the moved gradients."""
        gauge = _balancing_gauge(B, A)
        if gauge is None:
            return G_A, G_B
        t, t_inv = gauge
        B.copy_(B.double() @ t)
        A.copy_(t_inv @ A.double())
        M_B.copy_(M_B.double() @ t_inv.T)
        M_A.copy_(t.T @ M_A.double())
        return (t.T @ G_A.double()).to(G_A.dtype), (G_B.double() @ t_inv.T).to(G_B.dtype)

    def _nesterov(self, momentum_buffer: torch.Tensor, grad: torch.Tensor, group: dict) -> torch.Tensor:
        momentum_buffer.lerp_(grad, 1 - group["momentum"])
        return grad.lerp(momentum_buffer, group["momentum"]) if self.nesterov else momentum_buffer

    def _orthogonalize(self, x: torch.Tensor) -> torch.Tensor:
        with utils.fp32_matmul_precision(self.fp32_matmul_prec):
            return newton_schulz(x, **self._ns_kwargs)

    def _tp_group(self, p: torch.Tensor):
        if getattr(p, "partition_dim", -1) < 0 or self.pg_collection is None:
            return None
        group = self.pg_collection.expt_tp if getattr(p, "expert_tp", False) else self.pg_collection.tp
        return group if get_pg_size(group) > 1 else None

    def _gather(self, p: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        group = self._tp_group(p)
        if group is None:
            return x
        shards = [torch.empty_like(x) for _ in range(get_pg_size(group))]
        torch.distributed.all_gather(shards, x.contiguous(), group=group)
        return torch.cat(shards, dim=p.partition_dim)

    def _local(self, p: torch.Tensor, full: torch.Tensor) -> torch.Tensor:
        group = self._tp_group(p)
        return full if group is None else full.chunk(get_pg_size(group), dim=p.partition_dim)[get_pg_rank(group)]


def _inv_sqrt(x: torch.Tensor, damping: float) -> torch.Tensor:
    """``(x^T x + lam I)^-1/2`` with ``lam = damping * largest eigenvalue``, in float64."""
    x64 = x.double()
    eigvals, eigvecs = torch.linalg.eigh(x64.T @ x64)
    eigvals = eigvals.clamp_min(0)
    eigvals = eigvals + damping * eigvals.max()
    inv_sqrt = torch.where(eigvals > 0, eigvals.rsqrt(), 0.0)
    return ((eigvecs * inv_sqrt) @ eigvecs.T).to(x.dtype)


def _balancing_gauge(b: torch.Tensor, a: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Float64 ``(T, T^-1)`` with ``B T = U S^1/2`` and ``T^-1 A = S^1/2 W^T`` for the SVD ``B A = U S W^T``.

    None when ``B A`` has rank below r at ``b``'s precision, e.g. ``B = 0`` at LoRA initialization.
    """
    r_b = torch.linalg.qr(b.double(), mode="r").R
    r_a = torch.linalg.qr(a.double().T, mode="r").R
    u, s, wt = torch.linalg.svd(r_b @ r_a.T)
    if s[-1] <= s[0] * torch.finfo(b.dtype).eps:
        return None
    s_inv_sqrt = s.rsqrt()
    return (r_a.T @ wt.T) * s_inv_sqrt, s_inv_sqrt[:, None] * (u.T @ r_b)


def _lora_pairs(model_chunks) -> list[tuple[torch.Tensor, torch.Tensor, float, int]]:
    """Every multi-LoRA slot's ``(A, B, alpha / rank, rank)``, with the scale in the dtype the forward uses."""
    return [
        (adapter.linear_in.weight, adapter.linear_out.weight, (alpha / rank).item(), int(rank.item()))
        for chunk in model_chunks
        for module in chunk.modules()
        if hasattr(module, "rank_values")
        for adapter, alpha, rank in zip(module.adapters, module.alpha_values, module.rank_values, strict=True)
    ]


def _muown_config_to_kwargs(config, model_chunks, pg_collection) -> dict:
    return {**_muon_config_to_kwargs(config, model_chunks, pg_collection), "lora_pairs": _lora_pairs(model_chunks)}


# Muon's routing: 2D non-embedding parameters (the LoRA factors) here, the rest to Adam.
_EMERGING_OPTIMIZERS[MUOWN] = EmergingOptimizerEntry(
    optimizer_cls=TensorParallelMuownLoRA, config_to_kwargs=_muown_config_to_kwargs
)
