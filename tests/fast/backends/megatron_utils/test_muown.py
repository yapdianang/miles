"""Muown-LoRA checks from gist 869da14a2961b226343a398e50c3261f (verify.py), against kcc-lion/muown's Muown."""

import importlib.util
import shlex
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from megatron.bridge.peft.multi_lora_layers import expose_adapter_slot
from megatron.core.optimizer import OptimizerConfig
from megatron.core.optimizer.emerging_optimizers import _create_emerging_optimizer
from megatron.training.arguments import parse_args
from tests.fast.dist_utils import init_gloo, run_multiprocess

from miles.backends.megatron_utils import muown
from miles.backends.megatron_utils.lora import dora
from miles.backends.megatron_utils.lora.dora import MultiDoRALinear
from miles.backends.megatron_utils.lora.optimizer import SlotOptimizer
from miles.backends.megatron_utils.muown import TensorParallelMuownLoRA
from miles.utils.arguments import get_miles_extra_args_provider

M, N, R = 48, 40, 6
LR = 0.1
NS_STEPS = 10
# MiMo's extra 0.5 on Muown's 0.2 * sqrt(max(m, n))
EXTRA_SCALE = 0.5 * 0.2
MUON_KWARGS = dict(
    momentum=0.95,
    nesterov=True,
    weight_decay=0.0,
    num_ns_steps=NS_STEPS,
    coefficient_type="simple",
    scale_mode="spectral",
    extra_scale_factor=EXTRA_SCALE,
    fp32_matmul_prec="highest",
)
LAUNCHER = Path(__file__).resolve().parents[4] / "examples/multi_lora/serve_mimo_v26_flash_tinker.py"


def _ns(x: torch.Tensor) -> torch.Tensor:
    """kcc-lion/muown ``zeropower_via_newtonschulz5`` in float64."""
    a, b, c = (3.4445, -4.7750, 2.0315)
    x = x.double() / (x.double().norm() + 1e-7)
    transpose = x.size(0) > x.size(1)
    if transpose:
        x = x.T
    for _ in range(NS_STEPS):
        xxt = x @ x.T
        x = a * x + (b * xxt + c * xxt @ xxt) @ x
    return x.T if transpose else x


def _projector(x: torch.Tensor) -> torch.Tensor:
    return x.double() @ torch.linalg.pinv(x.double())


def _optimizer(a, b, *, scale=1.0, rank=R, cls=TensorParallelMuownLoRA, float64=False, **kwargs):
    optimizer = cls([a, b], lora_pairs=[(a, b, scale, rank)], lr=LR, **{**MUON_KWARGS, **kwargs})
    if float64:
        # Megatron's Newton-Schulz takes float32 only; float64 runs use the reference iteration
        optimizer._orthogonalize = _ns
    return optimizer


def _factor_grads(a, b, grad_v, scale):
    a.grad, b.grad = scale * b.detach().T @ grad_v, scale * grad_v @ a.detach().T


def _balanced(b: torch.Tensor, a: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    gauge = muown._balancing_gauge(b, a)
    if gauge is None:
        return b, a
    t, t_inv = gauge
    return (b.double() @ t).to(b.dtype), (t_inv @ a.double()).to(a.dtype)


def _first_order_dv(b, a, b_new, a_new, scale):
    """``s (B dA + dB A)`` at the rebalanced factors the step started from."""
    b, a = _balanced(b, a)
    return scale * (b @ (a_new - a) + (b_new - b) @ a)


@pytest.fixture
def factors():
    torch.manual_seed(0)
    return torch.randn(M, R), torch.randn(R, N), torch.randn(M, N)


@pytest.mark.parametrize("scale", [1.0, 2.0])
def test_thm1_whitened_steps_are_muon_on_the_tangent_space(factors, scale):
    b, a, grad_v = factors
    pa, pb = torch.nn.Parameter(a.clone()), torch.nn.Parameter(b.clone())
    optimizer = _optimizer(pa, pb, scale=scale, damping=0.0)
    # Tinker's AdamParams set the shared lr
    slot = SlotOptimizer.__new__(SlotOptimizer)
    slot._inner = SimpleNamespace(chained_optimizers=[optimizer])
    slot.apply_adam_params({"learning_rate": LR, "beta1": 0.95, "beta2": 0.95, "eps": 1e-8, "weight_decay": 0.0})
    _factor_grads(pa, pb, grad_v, scale)
    optimizer.step()

    dv = _first_order_dv(b, a, pb.detach(), pa.detach(), scale)
    expected = (
        -(LR / 2)
        * EXTRA_SCALE
        * max(M, N) ** 0.5
        * (_ns(_projector(b) @ grad_v.double()) + _ns(grad_v.double() @ _projector(a.T)))
    )
    # fp32 factors: dA and dB are differences of O(1) entries
    torch.testing.assert_close(dv.double(), expected, rtol=1e-4, atol=5e-6)


class _NoRebalance(TensorParallelMuownLoRA):
    def _rebalance(self, A, B, M_A, M_B, G_A, G_B):
        return G_A, G_B


def _gauged_step(b, a, grad_v, m_b, m_a, t, damping):
    """One unbalanced step from ``(B T, T^-1 A)`` with momenta and gradients moved by ``T``; returns the deltas."""
    t_inv = torch.linalg.inv(t)
    pa, pb = torch.nn.Parameter(t_inv @ a), torch.nn.Parameter(b @ t)
    optimizer = _optimizer(pa, pb, cls=_NoRebalance, float64=True, damping=damping)
    optimizer.state[pa]["momentum_buffer"], optimizer.state[pb]["momentum_buffer"] = t.T @ m_a, m_b @ t_inv.T
    _factor_grads(pa, pb, grad_v, 1.0)
    start_a, start_b = pa.detach().clone(), pb.detach().clone()
    optimizer.step()
    return pa.detach() - start_a, pb.detach() - start_b


def _gauges():
    torch.manual_seed(1)
    orthogonal = torch.linalg.qr(torch.randn(R, R, dtype=torch.float64))[0]
    general = torch.diag(torch.tensor([0.1, 1, 10, 1, 1, 1], dtype=torch.float64)) @ orthogonal
    b, a, grad_v = torch.randn(M, R), torch.randn(R, N), torch.randn(M, N)
    m_b, m_a = torch.randn(M, R), torch.randn(R, N)
    return orthogonal, general, *(x.double() for x in (b, a, grad_v, m_b, m_a))


def test_g1a_without_damping_steps_move_with_every_gauge():
    _, general, b, a, grad_v, m_b, m_a = _gauges()
    d_a, d_b = _gauged_step(b, a, grad_v, m_b, m_a, torch.eye(R, dtype=torch.float64), 0.0)
    d_a2, d_b2 = _gauged_step(b, a, grad_v, m_b, m_a, general, 0.0)
    torch.testing.assert_close(d_a2, torch.linalg.solve(general, d_a), rtol=1e-8, atol=1e-10)
    torch.testing.assert_close(d_b2, d_b @ general, rtol=1e-8, atol=1e-10)


def test_g1b_with_damping_steps_move_with_orthogonal_gauges_only():
    orthogonal, general, b, a, grad_v, m_b, m_a = _gauges()
    d_a, d_b = _gauged_step(b, a, grad_v, m_b, m_a, torch.eye(R, dtype=torch.float64), 0.1)
    d_a2, d_b2 = _gauged_step(b, a, grad_v, m_b, m_a, orthogonal, 0.1)
    torch.testing.assert_close(d_a2, orthogonal.T @ d_a, rtol=1e-8, atol=1e-10)
    torch.testing.assert_close(d_b2, d_b @ orthogonal, rtol=1e-8, atol=1e-10)
    d_a3, _ = _gauged_step(b, a, grad_v, m_b, m_a, general, 0.1)
    assert not torch.allclose(b @ general @ d_a3, b @ d_a)


class _NoMomentumTransport(TensorParallelMuownLoRA):
    def _rebalance(self, A, B, M_A, M_B, G_A, G_B):
        gauge = muown._balancing_gauge(B, A)
        if gauge is None:
            return G_A, G_B
        t, t_inv = gauge
        B.copy_(B @ t)
        A.copy_(t_inv @ A)
        return t.T @ G_A, G_B @ t_inv.T


def _v_trajectory(b, a, target, cls, steps=6):
    pa, pb = torch.nn.Parameter(a.clone()), torch.nn.Parameter(b.clone())
    optimizer = _optimizer(pa, pb, cls=cls, float64=True)
    trajectory = []
    for _ in range(steps):
        _factor_grads(pa, pb, pb.detach() @ pa.detach() - target, 1.0)
        optimizer.step()
        trajectory.append(pb.detach() @ pa.detach())
    return trajectory


@pytest.mark.parametrize("cls, invariant", [(TensorParallelMuownLoRA, True), (_NoMomentumTransport, False)])
def test_g2_v_trajectory_is_gauge_invariant_only_with_momentum_transport(cls, invariant):
    _, general, b, a, target, _, _ = _gauges()
    first = _v_trajectory(b, a, target, cls)
    second = _v_trajectory(b @ general, torch.linalg.solve(general, a), target, cls)
    worst = max(((x - y).norm() / x.norm()).item() for x, y in zip(first, second, strict=True))
    assert worst < 1e-10 if invariant else worst > 1e-4, worst


class _ReferenceMuown:
    """kcc-lion/muown ``Muown.step`` for one 2D weight in float64, with MiMo's direction-step scale."""

    def __init__(self, w: torch.Tensor, betas=(0.95, 0.95), eps=1e-8):
        self.w = w.double().clone()
        self.g = self.w.norm(dim=1, keepdim=True)
        self.v_norm = self.g.clone()
        self.m_v = torch.zeros_like(self.w)
        self.m_g = torch.zeros_like(self.g)
        self.v_g = torch.zeros_like(self.g)
        self.betas, self.eps, self.steps = betas, eps, 0

    def gradients(self, grad: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        u = self.w / self.g
        grad_g = (grad * u).sum(dim=1, keepdim=True)
        return grad_g, (self.g / self.v_norm) * (grad - u * grad_g)

    def step(self, grad: torch.Tensor, momentum: float = 0.95) -> tuple[torch.Tensor, torch.Tensor]:
        """-> the direction ``v`` before and after the step."""
        self.steps += 1
        u = self.w / self.g
        v = u * self.v_norm
        grad_g, grad_v = self.gradients(grad)
        self.m_v.mul_(momentum).add_(grad_v)
        v_new = v.add(_ns(grad_v.add(self.m_v, alpha=momentum)), alpha=-LR * EXTRA_SCALE * max(self.w.shape) ** 0.5)
        beta1, beta2 = self.betas
        self.m_g.mul_(beta1).add_(grad_g, alpha=1 - beta1)
        self.v_g.mul_(beta2).addcmul_(grad_g, grad_g, value=1 - beta2)
        m_hat, v_hat = self.m_g / (1 - beta1**self.steps), self.v_g / (1 - beta2**self.steps)
        self.g.addcdiv_(m_hat, v_hat.sqrt().add_(self.eps), value=-LR)
        self.v_norm = v_new.norm(dim=1, keepdim=True)
        self.w = self.g * v_new / self.v_norm
        return v, v_new


class _Adapter(torch.nn.Module):
    def __init__(self, a: torch.Tensor, b: torch.Tensor):
        super().__init__()
        self.linear_in, self.linear_out = torch.nn.Linear(1, 1, bias=False), torch.nn.Linear(1, 1, bias=False)
        self.linear_in.weight, self.linear_out.weight = torch.nn.Parameter(a.clone()), torch.nn.Parameter(b.clone())
        self.alpha = 1.0


def _dora_layer(w0, factors, magnitudes=None, *, rank, input_is_parallel=False, sequence_parallel=False):
    """A MultiDoRALinear around ``w0`` with one slot per ``(A, B)``, built without a Megatron linear; ``s = 2``."""
    layer = MultiDoRALinear.__new__(MultiDoRALinear)
    torch.nn.Module.__init__(layer)
    layer.to_wrap = torch.nn.Linear(1, 1, bias=False)
    layer.to_wrap.weight = torch.nn.Parameter(w0.clone(), requires_grad=False)
    layer.adapters = torch.nn.ModuleList(_Adapter(torch.zeros_like(a), torch.zeros_like(b)) for a, b in factors)
    layer.max_rank, layer.n_adapters, layer.base_linear_name = rank, len(factors), "linear"
    layer.alpha_values, layer.rank_values = torch.zeros(len(factors)), torch.full((len(factors),), float(rank))
    layer.replicate_adapter = layer._external_output_reduce = False
    layer.base_linear_is_parallel = layer._adapter_enabled = True
    layer.input_is_parallel, layer.disable_sequence_parallel_comm = input_is_parallel, not sequence_parallel
    layer.init_dora()
    for i, (a, b) in enumerate(factors):
        layer.init_adapter_slot(i, rank, 2.0 * rank)
        with torch.no_grad():
            layer.adapters[i].linear_in.weight.copy_(a)
            layer.adapters[i].linear_out.weight.copy_(b)
            if magnitudes is not None:
                layer.adapters[i].weight_magnitude.copy_(magnitudes[i])
    return layer


@pytest.fixture
def tp1(monkeypatch):
    monkeypatch.setattr(dora, "parallel_state", SimpleNamespace(get_tensor_model_parallel_world_size=lambda: 1))


def _dora_forward(layer, slot, grad_w):
    """``<W, grad_w>`` through the layer, with the identity as tokens so the output rows are ``V^T``."""
    adapter = layer.adapters[slot]
    v_t = layer.to_wrap.weight.T + 2.0 * adapter.linear_in.weight.T @ adapter.linear_out.weight.T
    layer.tokens_per_adapter_splits = tuple(v_t.shape[0] if i == slot else 0 for i in range(len(layer.adapters)))
    return (layer.scale_rows(v_t) * grad_w.T).sum()


def test_dora_starts_at_the_lora_weight_and_its_norm_is_the_dense_row_norm(tp1, factors):
    b, a, _ = factors
    w0 = torch.randn(M, N)
    layer = _dora_layer(w0, [(a, torch.zeros_like(b))], rank=R)
    torch.testing.assert_close(layer.adapters[0].weight_magnitude, w0.norm(dim=1))
    output = torch.randn(5, M)
    layer.tokens_per_adapter_splits = (5,)
    assert torch.equal(layer.scale_rows(output), output)

    with torch.no_grad():
        layer.adapters[0].linear_out.weight.copy_(b)
    torch.testing.assert_close(layer.row_norm(0), (w0 + 2.0 * b @ a).norm(dim=1))


@pytest.mark.parametrize("in_features", [4096, 8192])
def test_closed_form_norm_matches_the_dense_fp64_norm_on_mimo_sized_bf16_rows(tp1, in_features):
    # MiMo-V2.6 qkv_proj and o_proj inputs, rank 32, with s B A about a tenth of W0 per row
    torch.manual_seed(6)
    rows, rank = 256, 32
    w0 = (0.02 * torch.randn(rows, in_features)).to(torch.bfloat16)
    a = (torch.randn(rank, in_features) / in_features**0.5).to(torch.bfloat16)
    b = (0.02 * torch.randn(rows, rank)).to(torch.bfloat16)
    layer = _dora_layer(w0, [(a, b)], rank=rank)
    dense = (w0.double() + 2.0 * b.double() @ a.double()).norm(dim=1)
    assert ((w0.double().norm(dim=1) - dense).abs() / dense).max() > 1e-2
    torch.testing.assert_close(layer.row_norm(0).double(), dense, rtol=1e-6, atol=0)


def test_prop6_dora_gradients_are_muown_gradients(tp1, factors):
    b, a, grad_w = factors
    w0 = torch.randn(M, N)
    g = torch.rand(M) + 0.5
    layer = _dora_layer(w0, [(a, b)], [g], rank=R)
    _dora_forward(layer, 0, grad_w).backward()

    # Muown with direction v = V and magnitudes g
    v = (w0 + 2.0 * b @ a).double()
    reference = _ReferenceMuown(g.double()[:, None] * v / v.norm(dim=1, keepdim=True))
    reference.v_norm = v.norm(dim=1, keepdim=True)
    grad_g, grad_v = reference.gradients(grad_w.double())
    adapter = layer.adapters[0]
    torch.testing.assert_close(adapter.weight_magnitude.grad.double(), grad_g[:, 0], rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(
        adapter.linear_in.weight.grad.double(), 2.0 * b.double().T @ grad_v, rtol=1e-4, atol=1e-5
    )
    torch.testing.assert_close(
        adapter.linear_out.weight.grad.double(), 2.0 * grad_v @ a.double().T, rtol=1e-4, atol=1e-5
    )


def test_thm3_full_rank_muown_dora_is_kcc_lion_muown(tp1):
    torch.manual_seed(2)
    n = 16
    b, a, w0, grad_w = (torch.randn(n, n) for _ in range(4))
    w0 = w0 - 2.0 * b @ a + 3 * torch.eye(n)
    layer = _dora_layer(w0, [(a, b)], rank=n)
    with torch.no_grad():
        layer.adapters[0].weight_magnitude.copy_(layer.row_norm(0))
    reference = _ReferenceMuown(w0 + 2.0 * b @ a)
    v, v_new = reference.step(grad_w.double())

    _dora_forward(layer, 0, grad_w).backward()
    adapter = layer.adapters[0]
    torch.optim.Adam([adapter.weight_magnitude], lr=LR, betas=(0.95, 0.95), eps=1e-8).step()
    _optimizer(adapter.linear_in.weight, adapter.linear_out.weight, scale=2.0, rank=n, damping=0.0).step()

    torch.testing.assert_close(adapter.weight_magnitude.detach().double(), reference.g[:, 0], rtol=1e-5, atol=1e-6)
    dv = _first_order_dv(b, a, adapter.linear_out.weight.detach(), adapter.linear_in.weight.detach(), 2.0)
    torch.testing.assert_close(dv.double(), v_new - v, rtol=1e-4, atol=1e-6)


def test_exported_dora_delta_is_d_minus_one_in_column_zero_of_lora_b(tp1, factors):
    b, a, _ = factors
    layer = _dora_layer(torch.randn(M, N), [(a, b), (a, b)], [torch.rand(M) + 0.5] * 2, rank=R)
    expected = (layer.adapters[1].weight_magnitude / layer.row_norm(1) - 1).detach()
    # In the order the adapter export enters them: the delta needs the slot list that exposure hides
    with dora.dora_deltas_as_lora_b([layer], 1), expose_adapter_slot([layer], 1):
        exported = layer.adapter.linear_out.weight.detach().clone()
    assert torch.equal(exported[:, 0], expected) and not exported[:, 1:].any()
    assert torch.equal(layer.adapters[1].linear_out.weight, b)


def test_prop5_zero_b_takes_no_a_step_and_a_well_defined_b_step(factors):
    _, a, grad_v = factors
    pa, pb = torch.nn.Parameter(a.clone()), torch.nn.Parameter(torch.zeros(M, R))
    _factor_grads(pa, pb, grad_v, 1.0)
    _optimizer(pa, pb, damping=0.0).step()
    assert torch.equal(pa.detach(), a)
    assert torch.isfinite(pb).all()
    expected = -(LR / 2) * EXTRA_SCALE * max(M, N) ** 0.5 * _ns(grad_v.double() @ _projector(a.T))
    torch.testing.assert_close((pb.detach() @ a).double(), expected, rtol=1e-4, atol=1e-6)


def test_rank_padding_stays_zero_and_live_factors_step_alone(factors):
    b, a, grad_v = factors
    live = 4
    b[:, live:], a[live:] = 0, 0
    pa, pb = torch.nn.Parameter(a.clone()), torch.nn.Parameter(b.clone())
    qa, qb = torch.nn.Parameter(a[:live].clone()), torch.nn.Parameter(b[:, :live].clone())
    padded, alone = _optimizer(pa, pb, rank=live), _optimizer(qa, qb, rank=live)
    for _ in range(3):
        _factor_grads(pa, pb, grad_v, 1.0)
        _factor_grads(qa, qb, grad_v, 1.0)
        padded.step()
        alone.step()
    assert not pa[live:].any() and not pb[:, live:].any()
    torch.testing.assert_close(pb[:, :live] @ pa[:live], qb @ qa)


def _split_worker(rank: int, world_size: int, port: int) -> None:
    init_gloo(rank, world_size, port=port)
    torch.manual_seed(3)
    group = dist.group.WORLD
    # (A, B) partition dims: column-parallel base shards A's rank rows; row-parallel shards A's input columns
    for a_dim in (0, 1):
        b, a = torch.randn(M, R + 2), torch.randn(R + 2, N)
        grads = [torch.randn(M, N) for _ in range(3)]
        whole_a, whole_b = torch.nn.Parameter(a.clone()), torch.nn.Parameter(b.clone())
        shard_a = torch.nn.Parameter(a.chunk(world_size, dim=a_dim)[rank].clone())
        shard_b = torch.nn.Parameter(b.chunk(world_size, dim=0)[rank].clone())
        shard_a.partition_dim, shard_b.partition_dim = a_dim, 0
        sharded = _optimizer(shard_a, shard_b, rank=R + 2)
        sharded.pg_collection = SimpleNamespace(tp=group, expt_tp=group)
        whole = _optimizer(whole_a, whole_b, rank=R + 2)
        for grad_v in grads:
            _factor_grads(whole_a, whole_b, grad_v, 1.0)
            shard_a.grad = whole_a.grad.chunk(world_size, dim=a_dim)[rank].clone()
            shard_b.grad = whole_b.grad.chunk(world_size, dim=0)[rank].clone()
            whole.step()
            sharded.step()
        torch.testing.assert_close(shard_a.detach(), whole_a.detach().chunk(world_size, dim=a_dim)[rank])
        torch.testing.assert_close(shard_b.detach(), whole_b.detach().chunk(world_size, dim=0)[rank])
    dist.destroy_process_group()


def test_tensor_parallel_shards_match_the_whole_factors():
    run_multiprocess(_split_worker, world_size=2)


class _GatherRows(torch.autograd.Function):
    """Megatron's ``gather_from_sequence_parallel_region`` on gloo: reduce-scatter or split backward."""

    @staticmethod
    def forward(ctx, x, tensor_parallel_output_grad):
        ctx.reduce = tensor_parallel_output_grad
        shards = [torch.empty_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(shards, x.contiguous())
        return torch.cat(shards)

    @staticmethod
    def backward(ctx, grad):
        if ctx.reduce:
            grad = grad.clone()
            dist.all_reduce(grad)
        return grad.chunk(dist.get_world_size())[dist.get_rank()], None


class _ReduceFrom(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        x = x.clone()
        dist.all_reduce(x)
        return x

    @staticmethod
    def backward(ctx, grad):
        return grad


class _CopyTo(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        grad = grad.clone()
        dist.all_reduce(grad)
        return grad


def _set_tp(size: int, rank: int) -> None:
    dora.parallel_state = SimpleNamespace(
        get_tensor_model_parallel_world_size=lambda: size,
        get_tensor_model_parallel_rank=lambda: rank,
        get_tensor_model_parallel_group=lambda: dist.group.WORLD,
    )


def _dora_grads(layer, output, upstream, token_splits):
    output = output.clone().requires_grad_()
    layer.tokens_per_adapter_splits = token_splits
    y = layer.scale_rows(output)
    (y * upstream).sum().backward()
    grads = [(ad.linear_in.weight.grad, ad.linear_out.weight.grad, ad.weight_magnitude.grad) for ad in layer.adapters]
    return y.detach(), output.grad, grads


def _dora_split_worker(rank: int, world_size: int, port: int) -> None:
    init_gloo(rank, world_size, port=port)
    dora.gather_from_sequence_parallel_region = _GatherRows.apply
    dora.reduce_from_tensor_model_parallel_region = _ReduceFrom.apply
    dora.copy_to_tensor_model_parallel_region = _CopyTo.apply
    torch.manual_seed(5)
    m, n, r, tokens = 8, 12, 4, 6
    # Slot 1's two tokens sit in rank 1's sequence-parallel window only
    token_splits = (4, 2)
    w0 = torch.randn(m, n)
    factors = [(torch.randn(r, n), torch.randn(m, r)) for _ in range(2)]
    magnitudes = [torch.rand(m) + 0.5 for _ in range(2)]
    output, upstream = torch.randn(tokens, m), torch.randn(tokens, m)

    _set_tp(1, 0)
    y, grad_out, grads = _dora_grads(_dora_layer(w0, factors, magnitudes, rank=r), output, upstream, token_splits)

    _set_tp(world_size, rank)
    rows, cols, toks = (torch.arange(size).chunk(world_size)[rank] for size in (m, n, tokens))
    layouts = {
        # column-parallel: W0, B and g split output rows, A splits rank rows; every rank has every token
        "column": (
            False,
            False,
            w0[rows],
            [(a.chunk(world_size)[rank], b[rows]) for a, b in factors],
            [g[rows] for g in magnitudes],
            (slice(None), rows),
        ),
        # row-parallel: W0 and A split input columns, B output rows, g is replicated
        "row_sequence_parallel": (
            True,
            True,
            w0[:, cols],
            [(a[:, cols], b[rows]) for a, b in factors],
            magnitudes,
            (toks, slice(None)),
        ),
        "row": (
            True,
            False,
            w0[:, cols],
            [(a[:, cols], b[rows]) for a, b in factors],
            magnitudes,
            (slice(None), slice(None)),
        ),
    }
    for name, (input_is_parallel, sequence_parallel, w0_k, factors_k, magnitudes_k, view) in layouts.items():
        layer = _dora_layer(
            w0_k,
            factors_k,
            magnitudes_k,
            rank=r,
            input_is_parallel=input_is_parallel,
            sequence_parallel=sequence_parallel,
        )
        y_k, grad_out_k, grads_k = _dora_grads(layer, output[view], upstream[view], token_splits)
        torch.testing.assert_close(y_k, y[view], msg=name)
        torch.testing.assert_close(grad_out_k, grad_out[view], msg=name)
        for (ga, gb, gg), (ga_k, gb_k, gg_k) in zip(grads, grads_k, strict=True):
            a_view = ga.chunk(world_size)[rank] if name == "column" else ga[:, cols]
            torch.testing.assert_close(ga_k, a_view, msg=name)
            torch.testing.assert_close(gb_k, gb[rows], msg=name)
            torch.testing.assert_close(gg_k, gg[rows] if name == "column" else gg, msg=name)
    dist.destroy_process_group()


def test_dora_tensor_parallel_shards_match_the_whole_layer():
    run_multiprocess(_dora_split_worker, world_size=2)


def test_rejects_unpaired_params_and_weight_decay(factors):
    b, a, _ = factors
    pa, pb, extra = torch.nn.Parameter(a), torch.nn.Parameter(b), torch.nn.Parameter(torch.randn(4, 4))
    with pytest.raises(ValueError, match="only updates LoRA factor pairs"):
        TensorParallelMuownLoRA([pa, pb, extra], lora_pairs=[(pa, pb, 1.0, R)], lr=LR, **MUON_KWARGS)
    with pytest.raises(ValueError, match="share a param group"):
        TensorParallelMuownLoRA(
            [{"params": [pa]}, {"params": [pb]}], lora_pairs=[(pa, pb, 1.0, R)], lr=LR, **MUON_KWARGS
        )
    optimizer = _optimizer(pa, pb, weight_decay=0.1)
    _factor_grads(pa, pb, torch.randn(M, N), 1.0)
    with pytest.raises(NotImplementedError, match="weight_decay 0"):
        optimizer.step()


def test_launcher_flags_build_muown_lora(monkeypatch):
    spec = importlib.util.spec_from_file_location("serve_mimo_v26_flash_tinker", LAUNCHER)
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    monkeypatch.setattr(sys, "argv", ["serve", "--optimizer", "muown", *shlex.split(launcher._MUOWN_ARGS)])
    args = parse_args(extra_args_provider=get_miles_extra_args_provider())
    muon_fields = [name for name in OptimizerConfig.__dataclass_fields__ if name.startswith("muon_")]
    config = OptimizerConfig(optimizer=args.optimizer, **{name: getattr(args, name) for name in muon_fields})

    # The multi-LoRA layer's per-slot buffers; only slot 1's factors are in this optimizer
    adapters = [SimpleNamespace(linear_in=torch.nn.Linear(N, R), linear_out=torch.nn.Linear(R, M)) for _ in range(2)]
    layer = SimpleNamespace(
        adapters=adapters, alpha_values=torch.tensor([8.0, 12.0]), rank_values=torch.tensor([6.0, 4.0])
    )
    chunk = SimpleNamespace(
        modules=lambda: [layer], config=SimpleNamespace(num_attention_heads=4, num_query_groups=4, kv_channels=8)
    )
    params = [adapters[1].linear_in.weight, adapters[1].linear_out.weight]
    optimizer, _ = _create_emerging_optimizer(
        config, [{"params": params, "lr": LR, "weight_decay": 0.0}], "muown", [chunk], None
    )

    assert isinstance(optimizer, TensorParallelMuownLoRA) and optimizer.nesterov
    assert optimizer._lora_pairs == [(0, 0, 1, 3.0, 4)]
    assert optimizer.param_groups[0]["momentum"] == 0.95
    assert optimizer._ns_kwargs == {"steps": 10, "coefficient_type": "simple"}
    assert optimizer._step_scale(M, N) == pytest.approx(EXTRA_SCALE * max(M, N) ** 0.5)
