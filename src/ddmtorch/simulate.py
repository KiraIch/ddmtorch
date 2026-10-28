"""Synthetic data generation for the drift-diffusion model."""

from __future__ import annotations

from typing import Iterable

import torch

from .density import choice_probability, wfpt_logpdf
from .device import resolve_device, resolve_dtype


def _broadcast_params(v: object, a: object, w: object, t0: object, device, dtype):
    def to(x):
        if isinstance(x, torch.Tensor):
            return x.to(device=device, dtype=dtype)
        return torch.as_tensor(x, device=device, dtype=dtype)

    v_t, a_t, w_t, t0_t = torch.broadcast_tensors(to(v), to(a), to(w), to(t0))
    squeeze = v_t.ndim == 0
    return (
        v_t.reshape(-1),
        a_t.reshape(-1),
        w_t.reshape(-1),
        t0_t.reshape(-1),
        squeeze,
    )


def _rand(shape, *, device, dtype, generator):
    try:
        return torch.rand(shape, device=device, dtype=dtype, generator=generator)
    except (RuntimeError, TypeError):
        if device.type == "mps":
            out = torch.rand(shape, dtype=dtype, generator=generator)
            return out.to(device)
        raise


def _randn(shape, *, device, dtype, generator):
    try:
        return torch.randn(shape, device=device, dtype=dtype, generator=generator)
    except (RuntimeError, TypeError):
        if device.type == "mps":
            out = torch.randn(shape, dtype=dtype, generator=generator)
            return out.to(device)
        raise


@torch.no_grad()
def sample_wfpt(
    v: object,
    a: object,
    w: object,
    t0: object = 0.0,
    n: int = 1000,
    *,
    device: str | torch.device = "auto",
    dtype: torch.dtype | None = None,
    generator: torch.Generator | None = None,
    t_max: float = 10.0,
    n_grid: int = 8192,
    chunk_size: int = 64,
    k_max: int = 48,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample ``(rt, choice)`` pairs by inverting the WFPT CDF on a grid.

    Choice is 1 for the upper boundary and 0 for the lower boundary.  The
    returned tensors have shape ``(n,)`` for scalar parameters and ``(B, n)``
    for a batch of ``B`` parameter sets.
    """
    dev = resolve_device(device)
    dt = resolve_dtype(dev, dtype)
    v_t, a_t, w_t, t0_t, squeeze = _broadcast_params(v, a, w, t0, dev, dt)
    n_params = v_t.numel()

    a_t = torch.clamp(a_t, min=1e-4)
    w_t = torch.clamp(w_t, min=1e-5, max=1.0 - 1e-5)
    t0_t = torch.clamp(t0_t, min=0.0)
    p_upper = choice_probability(v_t, a_t, w_t, upper=True)

    eps = 1e-6
    u_grid = torch.linspace(eps, t_max, n_grid, device=dev, dtype=dt)
    du = (t_max - eps) / (n_grid - 1)

    rts = []
    choices = []
    for start in range(0, n_params, chunk_size):
        stop = min(start + chunk_size, n_params)
        vb = v_t[start:stop, None]
        ab = a_t[start:stop, None]
        wb = w_t[start:stop, None]
        tb = t0_t[start:stop, None]
        grid = tb + u_grid[None, :]

        logp_u = wfpt_logpdf(grid, vb, ab, wb, tb, boundary="upper", k_max=k_max)
        logp_l = wfpt_logpdf(grid, vb, ab, wb, tb, boundary="lower", k_max=k_max)
        dens_u = torch.exp(logp_u - logp_u.max(dim=1, keepdim=True).values)
        dens_l = torch.exp(logp_l - logp_l.max(dim=1, keepdim=True).values)

        def cdf_of(dens):
            c = torch.cumsum(dens, dim=1) * du
            c = c - c[:, :1]
            c = torch.clamp(c, min=0.0)
            return torch.cummax(c, dim=1).values / c[:, -1:].clamp_min(torch.finfo(dt).tiny)

        cdf_u = cdf_of(dens_u)
        cdf_l = cdf_of(dens_l)
        B = stop - start
        u = _rand((B, n), device=dev, dtype=dt, generator=generator)
        pick_u = u < p_upper[start:stop, None]
        uu = _rand((B, n), device=dev, dtype=dt, generator=generator)
        rt = torch.empty((B, n), device=dev, dtype=dt)

        # Sampling is done one parameter set at a time.  The expensive part
        # (evaluating the density on the grid) is still batched above; this
        # loop only performs 1-D inverse-CDF interpolation and keeps memory
        # at O(n_grid) rather than O(n_samples * n_grid).
        for b in range(B):
            grid_b = grid[b]
            for mask, cdf_b in ((pick_u[b], cdf_u[b]), (~pick_u[b], cdf_l[b])):
                if not mask.any():
                    continue
                ub = uu[b][mask]
                idx = torch.searchsorted(cdf_b, ub)
                idx = torch.clamp(idx, 0, n_grid - 2)
                c0 = cdf_b[idx]
                c1 = cdf_b[idx + 1]
                g0 = grid_b[idx]
                g1 = grid_b[idx + 1]
                frac = (ub - c0) / (c1 - c0).clamp_min(torch.finfo(dt).tiny)
                rt[b, mask] = g0 + frac * (g1 - g0)

        rts.append(rt)
        choices.append(pick_u.to(dt))

    rt = torch.cat(rts, dim=0)
    choice = torch.cat(choices, dim=0)
    if squeeze:
        return rt[0], choice[0]
    return rt, choice


@torch.no_grad()
def simulate_wfpt_euler(
    v: object,
    a: object,
    w: object,
    t0: object = 0.0,
    n: int = 1000,
    *,
    dt: float = 1e-3,
    max_time: float = 5.0,
    device: str | torch.device = "cpu",
    dtype: torch.dtype | None = None,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Approximate simulator using Euler-Maruyama integration.

    This simulator is intentionally independent of the WFPT density and is
    useful as a sanity check on the likelihood.  It uses a fixed step size,
    so very small non-decision times and large drift rates can show
    discretization bias.
    """
    dev = resolve_device(device)
    dt_t = resolve_dtype(dev, dtype)
    v_t, a_t, w_t, t0_t, squeeze = _broadcast_params(v, a, w, t0, dev, dt_t)
    B = v_t.numel()

    x = (a_t * w_t)[:, None].expand(B, n).clone()
    active = torch.ones((B, n), dtype=torch.bool, device=dev)
    out_t = torch.full((B, n), float("nan"), dtype=dt_t, device=dev)
    out_c = torch.full((B, n), float("nan"), dtype=dt_t, device=dev)

    n_steps = int(max_time / dt)
    sqrt_dt = dt**0.5
    for step in range(1, n_steps + 1):
        noise = _randn((B, n), device=dev, dtype=dt_t, generator=generator)
        x = x + v_t[:, None] * dt + sqrt_dt * noise
        hit_up = active & (x >= a_t[:, None])
        hit_lo = active & (x <= 0.0)
        hit = hit_up | hit_lo
        if hit.any():
            out_t[hit] = step * dt
            out_c[hit] = hit_up[hit].to(dt_t)
            active &= ~hit
        if not active.any():
            break

    out_t = out_t + t0_t[:, None]
    if squeeze:
        return out_t[0], out_c[0]
    return out_t, out_c
