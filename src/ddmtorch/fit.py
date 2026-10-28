"""Maximum-likelihood estimation for the drift-diffusion model."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

from .density import _wfpt_lower_logpdf
from .device import resolve_device, resolve_dtype


@dataclass
class FitResult:
    """Container for a fitted DDM model."""

    params: np.ndarray  # (n_subjects, 4), columns: v, a, w, t0
    nll: float
    n_trials: int
    n_subjects: int
    device: str
    dtype: str
    elapsed_s: float
    n_iter: int
    success: bool
    loss_history: list[float] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "params": self.params.tolist(),
            "nll": self.nll,
            "n_trials": self.n_trials,
            "n_subjects": self.n_subjects,
            "device": self.device,
            "dtype": self.dtype,
            "elapsed_s": self.elapsed_s,
            "n_iter": self.n_iter,
            "success": self.success,
        }


def loglik(
    rt: torch.Tensor,
    choice: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    w: torch.Tensor,
    t0: torch.Tensor,
    *,
    k_max: int = 48,
    switch_thresh: float = 0.8,
    p_contam: float = 0.0,
    contam_min: float = 0.0,
    contam_max: float | None = None,
) -> torch.Tensor:
    """Elementwise log likelihood for observed choices and response times.

    ``choice`` is 1 for the upper boundary and 0 for the lower boundary.
    All arguments must broadcast to the same shape.

    If ``p_contam > 0``, a fixed-proportion uniform contaminant is mixed
    with the DDM likelihood.  The contaminant is uniform on
    ``[contam_min, contam_max]`` and has probability 0.5 for each choice.
    This is a simple lapse model, not a full contaminant distribution.
    """
    if not 0.0 <= p_contam < 1.0:
        raise ValueError("p_contam must be in [0, 1)")
    choice_t = choice.to(dtype=rt.dtype)
    v_eff = torch.where(choice_t > 0.5, -v, v)
    w_eff = torch.where(choice_t > 0.5, 1.0 - w, w)
    logp = _wfpt_lower_logpdf(
        rt, v_eff, a, w_eff, t0, k_max=k_max, switch_thresh=switch_thresh
    )
    if p_contam == 0.0:
        return logp

    if contam_max is None:
        contam_max = float(rt.detach().max()) + 1e-6
    if contam_max <= contam_min:
        raise ValueError("contam_max must be greater than contam_min")
    log_contam = math.log(0.5) - math.log(float(contam_max) - float(contam_min))
    in_range = (rt >= contam_min) & (rt <= contam_max)
    logp_contam = torch.where(
        in_range,
        torch.full_like(logp, log_contam),
        torch.full_like(logp, -1e30),
    )
    return torch.logaddexp(
        math.log1p(-p_contam) + logp,
        math.log(p_contam) + logp_contam,
    )


def _softplus_inv(y: torch.Tensor) -> torch.Tensor:
    return torch.log(torch.expm1(y))


def _sync(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()


def _subject_min_rt(rt: torch.Tensor, subject: torch.Tensor, n_subjects: int) -> torch.Tensor:
    out = torch.full((n_subjects,), float("inf"), device=rt.device, dtype=rt.dtype)
    for s in range(n_subjects):
        mask = subject == s
        if mask.any():
            out[s] = rt[mask].min()
    return out


def fit_mle(
    rt: object,
    choice: object,
    subject: object | None = None,
    *,
    device: str | torch.device = "auto",
    dtype: torch.dtype | None = None,
    n_restarts: int = 1,
    max_iter: int = 200,
    lr: float = 0.5,
    a_init: float = 1.0,
    w_init: float = 0.5,
    t0_frac: float = 0.3,
    a_min: float = 0.05,
    prior_scale: float = 0.0,
    p_contam: float = 0.0,
    contam_min: float = 0.0,
    contam_max: float | None = None,
    t0_max: float | None = None,
    k_max: int = 48,
    switch_thresh: float = 0.8,
    seed: int | None = None,
    verbose: bool = False,
) -> FitResult:
    """Fit one DDM per subject by maximizing the trial-level likelihood.

    This is a pooled independent-subject MLE: every subject has their own
    ``(v, a, w, t0)`` but all subjects are optimized jointly in a single
    vectorized graph.  It is not a hierarchical Bayesian model.

    If ``p_contam > 0``, a fixed-proportion uniform contaminant is added to
    the likelihood.  In that case, pass ``t0_max`` explicitly if the true
    non-decision time may exceed the fastest response time; otherwise the
    default cap ``0.999 * min(RT)`` still applies.
    """
    dev = resolve_device(device)
    dt = resolve_dtype(dev, dtype)
    if seed is not None:
        torch.manual_seed(seed)

    rt_t = torch.as_tensor(rt, dtype=dt, device=dev).reshape(-1)
    choice_t = torch.as_tensor(choice, dtype=dt, device=dev).reshape(-1)
    if rt_t.numel() != choice_t.numel():
        raise ValueError("rt and choice must have the same number of trials")
    if torch.any(rt_t <= 0):
        raise ValueError("all response times must be positive")
    if torch.any((choice_t != 0) & (choice_t != 1)):
        raise ValueError("choice must contain only 0 and 1")

    if subject is None:
        subject_t = torch.zeros_like(rt_t, dtype=torch.long)
        n_subjects = 1
    else:
        subject_raw = torch.as_tensor(subject, dtype=torch.long, device=dev).reshape(-1)
        if subject_raw.numel() != rt_t.numel():
            raise ValueError("subject must have one entry per trial")
        uniq, inverse = torch.unique(subject_raw, return_inverse=True)
        subject_t = inverse.to(torch.long)
        n_subjects = int(uniq.numel())

    min_rt = _subject_min_rt(rt_t, subject_t, n_subjects)
    if torch.any(~torch.isfinite(min_rt)):
        raise ValueError("every subject must contribute at least one trial")
    if t0_max is None:
        t0_max_t = torch.clamp(min_rt * 0.999, min=1e-4)
    else:
        if t0_max <= 0:
            raise ValueError("t0_max must be positive")
        t0_max_t = torch.full((n_subjects,), float(t0_max), device=dev, dtype=dt)
    if p_contam > 0 and contam_max is None:
        contam_max = float(rt_t.max()) + 1e-6

    a_init_t = torch.full((n_subjects,), float(a_init), device=dev, dtype=dt)
    a_raw0 = _softplus_inv(torch.clamp(a_init_t - a_min, min=1e-4))
    w_raw0 = torch.full((n_subjects,), math.log(w_init / (1.0 - w_init)), device=dev, dtype=dt)
    t0_raw0 = torch.full(
        (n_subjects,), math.log(t0_frac / (1.0 - t0_frac)), device=dev, dtype=dt
    )
    v_raw0 = torch.zeros((n_subjects,), device=dev, dtype=dt)

    def unpack(raw: torch.Tensor):
        v = raw[:, 0]
        a = F.softplus(raw[:, 1]) + a_min
        w = torch.sigmoid(raw[:, 2])
        t0 = t0_max_t * torch.sigmoid(raw[:, 3])
        return v, a, w, t0

    def objective(raw: torch.Tensor) -> torch.Tensor:
        v, a, w, t0 = unpack(raw)
        vi = v[subject_t]
        ai = a[subject_t]
        wi = w[subject_t]
        t0i = t0[subject_t]
        lp = loglik(
            rt_t,
            choice_t,
            vi,
            ai,
            wi,
            t0i,
            k_max=k_max,
            switch_thresh=switch_thresh,
            p_contam=p_contam,
            contam_min=contam_min,
            contam_max=contam_max,
        )
        loss = -lp.mean()
        if prior_scale > 0:
            # Weakly regularizing priors.  They are off by default so that
            # recovery experiments estimate the pure MLE.
            log_prior = (
                -0.5 * (v / 3.0) ** 2
                - 0.5 * ((a - 1.0) / 2.0) ** 2
                + torch.log(w) + torch.log1p(-w)
            ).mean()
            loss = loss - prior_scale * log_prior
        return loss

    best_raw = None
    best_loss = float("inf")
    best_history: list[float] = []
    best_iters = 0
    total_start = time.perf_counter()

    for restart in range(n_restarts):
        if restart == 0:
            raw = torch.stack([v_raw0, a_raw0, w_raw0, t0_raw0], dim=1).clone()
        else:
            raw = torch.stack([v_raw0, a_raw0, w_raw0, t0_raw0], dim=1).clone()
            raw = raw + 0.25 * torch.randn_like(raw)
        raw.requires_grad_(True)
        history: list[float] = []
        counter = {"n": 0}

        opt = torch.optim.LBFGS(
            [raw],
            lr=lr,
            max_iter=max_iter,
            history_size=10,
            tolerance_grad=1e-7,
            tolerance_change=1e-10,
            line_search_fn="strong_wolfe",
        )

        def closure():
            opt.zero_grad(set_to_none=True)
            loss = objective(raw)
            if not torch.isfinite(loss):
                loss = torch.nan_to_num(loss, nan=1e30, posinf=1e30, neginf=1e30)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([raw], 1e4)
            counter["n"] += 1
            if not history or abs(float(loss.detach()) - history[-1]) > 1e-9:
                history.append(float(loss.detach()))
            return loss

        _sync(dev)
        try:
            opt.step(closure)
        except RuntimeError as exc:
            if verbose:
                print(f"LBFGS failed ({exc}); falling back to Adam")
            raw = raw.detach().clone().requires_grad_(True)
            opt_adam = torch.optim.Adam([raw], lr=lr * 0.05)
            for _ in range(max_iter * 5):
                opt_adam.zero_grad(set_to_none=True)
                loss = objective(raw)
                loss.backward()
                opt_adam.step()
                counter["n"] += 1
                if not history or abs(float(loss.detach()) - history[-1]) > 1e-9:
                    history.append(float(loss.detach()))
        _sync(dev)

        with torch.no_grad():
            final_loss = float(objective(raw))
        if verbose:
            print(f"restart {restart + 1}/{n_restarts}: nll={final_loss:.6f}")
        if final_loss < best_loss:
            best_loss = final_loss
            best_raw = raw.detach().clone()
            best_history = history
            best_iters = counter["n"]

    _sync(dev)
    elapsed = time.perf_counter() - total_start
    with torch.no_grad():
        v, a, w, t0 = unpack(best_raw)
        params = torch.stack([v, a, w, t0], dim=1).detach().cpu().numpy()

    return FitResult(
        params=params,
        nll=float(best_loss),
        n_trials=int(rt_t.numel()),
        n_subjects=n_subjects,
        device=str(dev),
        dtype=str(dt),
        elapsed_s=elapsed,
        n_iter=best_iters,
        success=bool(np.isfinite(best_loss)),
        loss_history=best_history,
    )
