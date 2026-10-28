"""Vectorized Wiener first-passage time (WFPT) densities.

The implementation follows the hybrid large-time / small-time strategy of
Navarro & Fuss (2009), with the small-time representation written in the
form used by the ``fddm`` package (Foster & Singmann).  All operations are
pure PyTorch so the same code runs on CPU, CUDA, and Apple Silicon MPS.

Conventions
-----------
* ``a`` is the boundary separation (a > 0).
* ``w`` is the relative starting point in (0, 1), measured from the lower
  boundary.  The starting evidence is ``z = a * w``.
* ``v`` is the drift rate; the diffusion coefficient is fixed at 1.
* ``t0`` is the non-decision time.  The density is evaluated at ``rt - t0``.
* The public ``wfpt_logpdf`` returns the density of hitting the *upper*
  boundary at ``a``.  Use ``boundary="lower"`` for the lower boundary, or
  the internal transformation in ``loglik`` for trial-level choices.
"""

from __future__ import annotations

import math

import torch


_LOG_PI = math.log(math.pi)
_LOG_2PI = math.log(2.0 * math.pi)


def _infer_meta(*xs: object) -> tuple[torch.device, torch.dtype]:
    for x in xs:
        if isinstance(x, torch.Tensor):
            return x.device, x.dtype
    return torch.device("cpu"), torch.float64


def _as_tensor(x: object, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    return torch.as_tensor(x, device=device, dtype=dtype)


def _large_time_sum(tau: torch.Tensor, a: torch.Tensor, w: torch.Tensor, k_max: int) -> torch.Tensor:
    """Sum_{j=1}^{k_max} j sin(j pi w) exp(-j^2 pi^2 tau / (2 a^2))."""
    j = torch.arange(1, k_max + 1, device=tau.device, dtype=tau.dtype)
    j = j.view((-1,) + (1,) * tau.ndim)
    terms = j * torch.sin(j * math.pi * w) * torch.exp(-(j * j) * (math.pi**2) * tau / (2.0 * a * a))
    return terms.sum(dim=0)


def _small_time_sum(tau: torch.Tensor, a: torch.Tensor, w: torch.Tensor, k_max: int) -> torch.Tensor:
    """Sum_{j=-k_max}^{k_max} (w + 2j) exp(-a^2 (w + 2j)^2 / (2 tau))."""
    j = torch.arange(-k_max, k_max + 1, device=tau.device, dtype=tau.dtype)
    j = j.view((-1,) + (1,) * tau.ndim)
    x = w + 2.0 * j
    terms = x * torch.exp(-(a * a) * x * x / (2.0 * tau))
    return terms.sum(dim=0)


def _wfpt_lower_logpdf(
    rt: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    w: torch.Tensor,
    t0: torch.Tensor = 0.0,
    *,
    k_max: int = 48,
    switch_thresh: float = 0.8,
) -> torch.Tensor:
    """Log density for hitting the *lower* boundary (fddm convention)."""
    device, dtype = _infer_meta(rt, v, a, w, t0)
    rt = _as_tensor(rt, device, dtype)
    v = _as_tensor(v, device, dtype)
    a = _as_tensor(a, device, dtype)
    w = _as_tensor(w, device, dtype)
    t0 = _as_tensor(t0, device, dtype)

    tau = rt - t0
    valid = tau > 0
    tau_safe = torch.where(valid, tau, torch.ones_like(tau))

    # Keep the parameters inside their mathematical domains.  The clamps are
    # inactive for interior optima but prevent NaNs during line searches.
    a_safe = torch.clamp(a, min=1e-6)
    w_safe = torch.clamp(w, min=1e-6, max=1.0 - 1e-6)
    tau_safe = torch.clamp(tau_safe, min=1e-8)

    # Align all parameter shapes before the series sums so that, for example,
    # a [n_trials] response-time vector and [n_subjects, 1] parameters work.
    tau_safe, a_safe, w_safe, v_safe = torch.broadcast_tensors(
        tau_safe, a_safe, w_safe, v
    )

    scaled_time = tau_safe / (a_safe * a_safe)
    use_large = scaled_time > switch_thresh

    sum_l = _large_time_sum(tau_safe, a_safe, w_safe, k_max)
    sum_s = _small_time_sum(tau_safe, a_safe, w_safe, k_max)

    tiny = torch.finfo(dtype).tiny
    log_mult_l = (
        _LOG_PI
        - 2.0 * torch.log(a_safe)
        - v_safe * a_safe * w_safe
        - 0.5 * v_safe * v_safe * tau_safe
    )
    log_mult_s = (
        torch.log(a_safe)
        - 0.5 * _LOG_2PI
        - 1.5 * torch.log(tau_safe)
        - v_safe * a_safe * w_safe
        - 0.5 * v_safe * v_safe * tau_safe
    )

    # The truncated sums are positive in the valid parameter region; clamping
    # only guards against pathological values during unconstrained search.
    log_dens_l = log_mult_l + torch.log(torch.clamp(sum_l, min=tiny))
    log_dens_s = log_mult_s + torch.log(torch.clamp(sum_s, min=tiny))
    log_dens = torch.where(use_large, log_dens_l, log_dens_s)

    neg_inf = torch.full_like(log_dens, -1e30)
    return torch.where(valid, log_dens, neg_inf)


def wfpt_logpdf(
    rt: object,
    v: object,
    a: object,
    w: object,
    t0: object = 0.0,
    *,
    boundary: str = "upper",
    k_max: int = 48,
    switch_thresh: float = 0.8,
) -> torch.Tensor:
    """Log WFPT density at response time ``rt``.

    Parameters may be Python scalars or broadcastable tensors.  The result is
    the log density of first crossing of the requested boundary.
    """
    if boundary not in {"upper", "lower"}:
        raise ValueError("boundary must be 'upper' or 'lower'")

    device, dtype = _infer_meta(rt, v, a, w, t0)
    v_t = _as_tensor(v, device, dtype)
    w_t = _as_tensor(w, device, dtype)
    if boundary == "upper":
        # f_upper(t | v, a, w) = f_lower(t | -v, a, 1 - w)
        return _wfpt_lower_logpdf(
            rt, -v_t, a, 1.0 - w_t, t0, k_max=k_max, switch_thresh=switch_thresh
        )
    return _wfpt_lower_logpdf(rt, v_t, a, w_t, t0, k_max=k_max, switch_thresh=switch_thresh)


def wfpt_pdf(
    rt: object,
    v: object,
    a: object,
    w: object,
    t0: object = 0.0,
    *,
    boundary: str = "upper",
    k_max: int = 48,
    switch_thresh: float = 0.8,
) -> torch.Tensor:
    """WFPT density (not log) at response time ``rt``."""
    return torch.exp(
        wfpt_logpdf(
            rt,
            v,
            a,
            w,
            t0,
            boundary=boundary,
            k_max=k_max,
            switch_thresh=switch_thresh,
        )
    )


def choice_probability(v: object, a: object, w: object, *, upper: bool = True) -> torch.Tensor:
    """Probability of hitting the upper (or lower) boundary.

    Uses ``P_upper = (1 - exp(-2 v a w)) / (1 - exp(-2 v a))`` with a
    numerically stable limit for ``v -> 0``.
    """
    device, dtype = _infer_meta(v, a, w)
    v_t = _as_tensor(v, device, dtype)
    a_t = _as_tensor(a, device, dtype)
    w_t = _as_tensor(w, device, dtype)

    za = 2.0 * v_t * a_t
    # Keep expm1 finite even for extreme trial parameters.  MPS works in
    # float32, where expm1 overflows around 88.
    limit = 80.0 if dtype == torch.float32 else 700.0
    za = torch.clamp(za, min=-limit, max=limit)
    denom = torch.expm1(-za)
    numer = torch.expm1(-za * w_t)
    tiny = torch.finfo(dtype).eps
    p = numer / torch.where(torch.abs(denom) < tiny, torch.full_like(denom, tiny), denom)
    p = torch.where(torch.abs(za) < 1e-6, w_t, p)
    p = torch.clamp(p, min=tiny, max=1.0 - tiny)
    return p if upper else 1.0 - p
