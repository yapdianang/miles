"""Muown-LoRA checks from gist 869da14a2961b226343a398e50c3261f (verify.py), against kcc-lion/muown's Muown."""

import importlib.util
import shlex
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from megatron.core.optimizer import OptimizerConfig
from megatron.core.optimizer.emerging_optimizers import _create_emerging_optimizer
from megatron.training.arguments import parse_args
from tests.fast.dist_utils import init_gloo, run_multiprocess

from miles.backends.megatron_utils import muown
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


def _dora(w0, g, b, a, scale):
    v = w0 + scale * b @ a
    return g * v / v.norm(dim=1, keepdim=True)


def test_prop6_dora_gradients_are_muown_gradients_only_with_the_norm_attached(factors):
    b, a, grad_w = (x.double() for x in factors)
    w0 = torch.randn(M, N, dtype=torch.float64)
    pa, pb = a.clone().requires_grad_(), b.clone().requires_grad_()
    g = (w0 + 2.0 * b @ a).norm(dim=1, keepdim=True).requires_grad_()
    (_dora(w0, g, pb, pa, 2.0) * grad_w).sum().backward()

    reference = _ReferenceMuown(w0 + 2.0 * b @ a)
    grad_g, grad_v = reference.gradients(grad_w)
    torch.testing.assert_close(g.grad, grad_g)
    torch.testing.assert_close(pa.grad, 2.0 * b.T @ grad_v)
    torch.testing.assert_close(pb.grad, 2.0 * grad_v @ a.T)

    # Megatron-Bridge's DoRA detaches the norm, which leaves the radial component in dL/dV
    pa.grad = None
    v = w0 + 2.0 * pb @ pa
    (g.detach() * v / v.norm(dim=1, keepdim=True).detach() * grad_w).sum().backward()
    assert not torch.allclose(pa.grad, 2.0 * b.T @ grad_v)


def test_thm3_full_rank_muown_lora_is_kcc_lion_muown():
    torch.manual_seed(2)
    n, scale = 16, 2.0
    b, a, w0, grad_w = (torch.randn(n, n, dtype=torch.float64) for _ in range(4))
    w0 = w0 - scale * b @ a + 3 * torch.eye(n, dtype=torch.float64)
    reference = _ReferenceMuown(w0 + scale * b @ a)
    v, v_new = reference.step(grad_w)

    # DoRA forward with g = ||V|| at init, so W = V and the reference sees the same weight
    pa, pb = torch.nn.Parameter(a.clone()), torch.nn.Parameter(b.clone())
    g = torch.nn.Parameter((w0 + scale * b @ a).norm(dim=1, keepdim=True))
    (_dora(w0, g, pb, pa, scale) * grad_w).sum().backward()
    torch.optim.Adam([g], lr=LR, betas=(0.95, 0.95), eps=1e-8).step()
    _optimizer(pa, pb, scale=scale, rank=n, float64=True, damping=0.0).step()

    torch.testing.assert_close(g.detach(), reference.g)
    dv = _first_order_dv(b, a, pb.detach(), pa.detach(), scale)
    # Newton-Schulz's 1e-7 eps sees differently scaled inputs (EMA vs heavy-ball momentum, P^-1 whitening)
    torch.testing.assert_close(dv, v_new - v, rtol=1e-4, atol=1e-8)


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
