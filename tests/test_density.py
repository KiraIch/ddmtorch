import math

import pytest
import torch

from ddmtorch import choice_probability, wfpt_logpdf, wfpt_pdf
from ddmtorch.density import _large_time_sum, _small_time_sum


def _direct_density(tau, v, a, w, kind, k_max=512):
    tau_t = torch.tensor(tau, dtype=torch.float64)
    v_t = torch.tensor(v, dtype=torch.float64)
    a_t = torch.tensor(a, dtype=torch.float64)
    w_t = torch.tensor(w, dtype=torch.float64)
    if kind == "large":
        s = _large_time_sum(tau_t, a_t, w_t, k_max)
        log_mult = math.log(math.pi) - 2 * torch.log(a_t) - v_t * a_t * w_t - 0.5 * v_t**2 * tau_t
    else:
        s = _small_time_sum(tau_t, a_t, w_t, k_max)
        log_mult = (
            torch.log(a_t)
            - 0.5 * math.log(2 * math.pi)
            - 1.5 * torch.log(tau_t)
            - v_t * a_t * w_t
            - 0.5 * v_t**2 * tau_t
        )
    return float(torch.exp(log_mult + torch.log(torch.clamp(s, min=1e-300))))


@pytest.mark.parametrize("tau", [0.02, 0.05, 0.1, 0.3, 0.8, 1.5])
@pytest.mark.parametrize("w", [0.15, 0.35, 0.5, 0.7, 0.9])
def test_large_and_small_series_agree(tau, w):
    d_large = _direct_density(tau, 0.5, 1.0, w, "large")
    d_small = _direct_density(tau, 0.5, 1.0, w, "small")
    assert d_large > 0 and d_small > 0
    assert abs(d_large - d_small) / max(d_large, d_small) < 1e-7


@pytest.mark.parametrize("v", [-2.0, -0.5, 0.0, 0.5, 2.0])
@pytest.mark.parametrize("w", [0.3, 0.5, 0.7])
def test_density_integrates_to_choice_probability(v, w):
    a = 1.2
    tau = torch.linspace(1e-4, 20.0, 60000, dtype=torch.float64)
    dens = wfpt_pdf(tau, v, a, w, boundary="upper")
    integral = float(torch.trapezoid(dens, tau))
    p = float(choice_probability(torch.tensor(v), torch.tensor(a), torch.tensor(w)))
    assert abs(integral - p) < 2e-5


def test_choice_probability_limits():
    w = torch.tensor([0.2, 0.5, 0.8], dtype=torch.float64)
    p0 = choice_probability(torch.zeros_like(w), torch.ones_like(w), w)
    assert torch.allclose(p0, w)
    p_pos = choice_probability(torch.full_like(w, 20.0), torch.ones_like(w), w)
    p_neg = choice_probability(torch.full_like(w, -20.0), torch.ones_like(w), w)
    assert torch.all(p_pos > 0.99)
    assert torch.all(p_neg < 0.01)
    # The complement of the upper-boundary probability is the lower-boundary
    # probability for the *same* drift, not for the reversed drift.
    p_lower = choice_probability(torch.full_like(w, 20.0), torch.ones_like(w), w, upper=False)
    assert torch.allclose(p_pos + p_lower, torch.ones_like(p_pos), atol=1e-12)


def test_invalid_times_return_large_negative_loglik():
    lp = wfpt_logpdf(0.2, 1.0, 1.0, 0.5, t0=0.3)
    assert float(lp) <= -1e29


def test_broadcasting():
    t = torch.linspace(0.4, 2.0, 7, dtype=torch.float64)
    v = torch.tensor([[0.2], [-0.4]], dtype=torch.float64)
    a = torch.tensor([[1.0], [1.4]], dtype=torch.float64)
    w = torch.tensor([[0.5], [0.4]], dtype=torch.float64)
    lp = wfpt_logpdf(t, v, a, w, t0=0.2)
    assert lp.shape == (2, 7)
    assert torch.isfinite(lp).all()


def test_gradcheck():
    t = torch.tensor([0.7, 1.1], dtype=torch.float64)
    v = torch.tensor([0.3, -0.2], dtype=torch.float64, requires_grad=True)
    a = torch.tensor([1.0, 1.3], dtype=torch.float64, requires_grad=True)
    w = torch.tensor([0.5, 0.4], dtype=torch.float64, requires_grad=True)
    t0 = torch.tensor([0.2, 0.2], dtype=torch.float64, requires_grad=True)

    def f(v_, a_, w_, t0_):
        return wfpt_logpdf(t, v_, a_, w_, t0_, boundary="upper").sum()

    assert torch.autograd.gradcheck(f, (v, a, w, t0), eps=1e-6, atol=1e-4, rtol=1e-3)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
def test_mps_matches_cpu_float32():
    g = torch.Generator().manual_seed(0)
    t = 0.2 + 2.5 * torch.rand(200, generator=g)
    v = -2 + 4 * torch.rand(200, generator=g)
    a = 0.6 + 1.2 * torch.rand(200, generator=g)
    w = 0.15 + 0.7 * torch.rand(200, generator=g)
    t0 = 0.15 * torch.rand(200, generator=g)

    cpu = wfpt_logpdf(t.float(), v.float(), a.float(), w.float(), t0.float())
    mps = wfpt_logpdf(
        t.float().to("mps"),
        v.float().to("mps"),
        a.float().to("mps"),
        w.float().to("mps"),
        t0.float().to("mps"),
    ).cpu()
    assert torch.allclose(cpu, mps, atol=3e-3, rtol=3e-3)
