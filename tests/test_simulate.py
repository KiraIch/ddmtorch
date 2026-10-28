import torch

from ddmtorch import choice_probability, sample_wfpt
from ddmtorch.simulate import simulate_wfpt_euler


def test_sampler_moments_zero_drift():
    torch.manual_seed(0)
    rt, choice = sample_wfpt(
        0.0,
        1.0,
        0.5,
        0.3,
        n=20000,
        device="cpu",
        dtype=torch.float64,
        t_max=8.0,
        n_grid=8192,
    )
    assert rt.shape == (20000,)
    # E[T] = a^2 w (1-w) = 0.25, plus t0 = 0.3.
    assert abs(float(rt.mean()) - 0.55) < 0.02
    assert abs(float(choice.mean()) - 0.5) < 0.02


def test_sampler_moments_drift():
    torch.manual_seed(1)
    v, a, w, t0 = 1.2, 1.0, 0.5, 0.25
    rt, choice = sample_wfpt(
        v, a, w, t0, n=20000, device="cpu", dtype=torch.float64, t_max=8.0, n_grid=8192
    )
    p = float(choice_probability(v, a, w))
    assert abs(float(choice.mean()) - p) < 0.02
    assert float(rt.min()) > t0


def test_sampler_batch_shapes():
    v = torch.tensor([0.0, 1.0], dtype=torch.float64)
    a = torch.tensor([1.0, 1.2], dtype=torch.float64)
    w = torch.tensor([0.5, 0.4], dtype=torch.float64)
    t0 = torch.tensor([0.2, 0.3], dtype=torch.float64)
    rt, choice = sample_wfpt(
        v, a, w, t0, n=100, device="cpu", dtype=torch.float64, n_grid=4096
    )
    assert rt.shape == (2, 100)
    assert choice.shape == (2, 100)


def test_euler_simulator_sanity():
    torch.manual_seed(2)
    v, a, w, t0 = 1.0, 1.0, 0.5, 0.2
    rt, choice = simulate_wfpt_euler(
        v,
        a,
        w,
        t0,
        n=3000,
        dt=2e-3,
        max_time=5.0,
        device="cpu",
        dtype=torch.float64,
        generator=torch.Generator().manual_seed(3),
    )
    p = float(choice_probability(v, a, w))
    assert torch.isfinite(rt).all()
    assert abs(float(choice.mean()) - p) < 0.05
    assert abs(float(rt.mean()) - 0.46) < 0.08
