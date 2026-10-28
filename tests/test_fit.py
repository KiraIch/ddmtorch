import numpy as np
import pytest
import torch

from ddmtorch import fit_mle, loglik, sample_wfpt, wfpt_logpdf


def test_loglik_matches_public_density():
    t = torch.tensor([0.6, 0.9, 1.4], dtype=torch.float64)
    v = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float64)
    a = torch.tensor([1.1, 1.1, 1.1], dtype=torch.float64)
    w = torch.tensor([0.4, 0.4, 0.4], dtype=torch.float64)
    t0 = torch.tensor([0.2, 0.2, 0.2], dtype=torch.float64)
    choice = torch.tensor([1.0, 0.0, 1.0], dtype=torch.float64)

    ll = loglik(t, choice, v, a, w, t0)
    up = wfpt_logpdf(t, v, a, w, t0, boundary="upper")
    lo = wfpt_logpdf(t, v, a, w, t0, boundary="lower")
    expected = torch.where(choice > 0.5, up, lo)
    assert torch.allclose(ll, expected)


def test_single_subject_recovery():
    torch.manual_seed(0)
    true = torch.tensor([1.0, 1.0, 0.5, 0.3], dtype=torch.float64)
    rt, choice = sample_wfpt(
        *true,
        n=3000,
        device="cpu",
        dtype=torch.float64,
        t_max=8.0,
        n_grid=8192,
    )
    res = fit_mle(rt, choice, device="cpu", max_iter=80, seed=1)
    est = res.params[0]
    assert abs(est[0] - 1.0) < 0.25
    assert abs(est[1] - 1.0) < 0.15
    assert abs(est[2] - 0.5) < 0.06
    assert abs(est[3] - 0.3) < 0.05
    assert res.success


def test_multi_subject_recovery():
    torch.manual_seed(1)
    true_v = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float64)
    true_a = torch.tensor([0.8, 1.0, 1.3], dtype=torch.float64)
    true_w = torch.tensor([0.4, 0.5, 0.6], dtype=torch.float64)
    true_t0 = torch.tensor([0.2, 0.3, 0.4], dtype=torch.float64)
    rt, choice = sample_wfpt(
        true_v,
        true_a,
        true_w,
        true_t0,
        n=1200,
        device="cpu",
        dtype=torch.float64,
        t_max=8.0,
        n_grid=8192,
    )
    subject = torch.arange(3).repeat_interleave(1200)
    res = fit_mle(
        rt.reshape(-1), choice.reshape(-1), subject, device="cpu", max_iter=80, seed=2
    )
    assert np.max(np.abs(res.params[:, 0] - true_v.numpy())) < 0.3
    assert np.max(np.abs(res.params[:, 1] - true_a.numpy())) < 0.2
    assert np.max(np.abs(res.params[:, 2] - true_w.numpy())) < 0.08
    assert np.max(np.abs(res.params[:, 3] - true_t0.numpy())) < 0.06


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
def test_mps_fit_matches_cpu():
    torch.manual_seed(3)
    rt, choice = sample_wfpt(
        0.8,
        1.0,
        0.5,
        0.25,
        n=1200,
        device="cpu",
        dtype=torch.float64,
        t_max=8.0,
        n_grid=8192,
    )
    res_cpu = fit_mle(rt, choice, device="cpu", max_iter=60, seed=4)
    res_mps = fit_mle(rt.float(), choice.float(), device="mps", max_iter=60, seed=4)
    assert res_mps.success
    assert np.allclose(res_cpu.params[0], res_mps.params[0], atol=5e-2)


def test_fit_rejects_bad_inputs():
    with pytest.raises(ValueError):
        fit_mle(torch.tensor([-1.0]), torch.tensor([1.0]))
    with pytest.raises(ValueError):
        fit_mle(torch.tensor([1.0]), torch.tensor([2.0]))


def test_loglik_mixture_matches_manual():
    t = torch.tensor([0.2, 0.6, 1.0, 1.8], dtype=torch.float64)
    choice = torch.tensor([1.0, 0.0, 1.0, 0.0], dtype=torch.float64)
    v = torch.ones_like(t)
    a = torch.full_like(t, 1.2)
    w = torch.full_like(t, 0.5)
    t0 = torch.full_like(t, 0.25)
    lp_ddm = loglik(t, choice, v, a, w, t0)
    lp_mix = loglik(
        t, choice, v, a, w, t0, p_contam=0.02, contam_min=0.0, contam_max=2.0
    )
    other = torch.full_like(lp_ddm, np.log(0.02) + np.log(0.5 / 2.0))
    expected = torch.logaddexp(np.log(0.98) + lp_ddm, other)
    assert torch.allclose(lp_mix, expected)


def test_fit_mle_mixture_recovers_t0():
    torch.manual_seed(11)
    true_t0 = 0.32
    rt, choice = sample_wfpt(
        1.0,
        1.2,
        0.5,
        true_t0,
        n=1500,
        device="cpu",
        dtype=torch.float64,
        t_max=8.0,
        n_grid=4096,
    )
    g = torch.Generator().manual_seed(12)
    n_cont = 75
    idx = torch.randperm(len(rt), generator=g)[:n_cont]
    rt[idx] = 0.1 + 0.4 * torch.rand(n_cont, generator=g, dtype=torch.float64)
    choice[idx] = torch.randint(0, 2, (n_cont,), generator=g).double()

    res_no = fit_mle(rt, choice, device="cpu", max_iter=100, seed=13)
    res_mix = fit_mle(
        rt,
        choice,
        device="cpu",
        max_iter=100,
        seed=13,
        p_contam=0.05,
        t0_max=0.8,
        t0_frac=0.4,
    )
    assert res_no.params[0, 3] < 0.15
    assert res_mix.params[0, 3] > 0.22
    assert abs(res_mix.params[0, 1] - 1.2) < 0.25
