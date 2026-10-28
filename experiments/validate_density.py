"""Validate the production WFPT density against a high-precision reference."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from ddmtorch import wfpt_logpdf


def main() -> None:
    g = torch.Generator().manual_seed(123)
    n = 20_000
    t0 = 0.1 + 0.4 * torch.rand(n, generator=g, dtype=torch.float64)
    t = t0 + 0.005 + 2.995 * torch.rand(n, generator=g, dtype=torch.float64)
    v = -2.0 + 4.0 * torch.rand(n, generator=g, dtype=torch.float64)
    a = 0.6 + 1.4 * torch.rand(n, generator=g, dtype=torch.float64)
    w = 0.15 + 0.7 * torch.rand(n, generator=g, dtype=torch.float64)

    ref = wfpt_logpdf(t, v, a, w, t0, k_max=512)
    cpu32 = wfpt_logpdf(
        t.float(), v.float(), a.float(), w.float(), t0.float(), k_max=48
    )
    mps32 = wfpt_logpdf(
        t.float().to("mps"),
        v.float().to("mps"),
        a.float().to("mps"),
        w.float().to("mps"),
        t0.float().to("mps"),
        k_max=48,
    ).cpu()

    tau = t - t0
    # Float32 cannot represent densities below ~exp(-87).  Those extreme
    # right-tail values are reported separately rather than mixed into the
    # accuracy summary.
    representable = ref > np.log(np.finfo(np.float32).tiny)
    near_t0 = tau < 0.02
    analysis_mask = representable & (~near_t0)

    def summarize(x, name):
        err = (x.double() - ref).abs()
        err_analysis = err[analysis_mask]
        rel_dens = (torch.exp(x.double()) - torch.exp(ref)).abs() / torch.exp(ref)
        rel_analysis = rel_dens[analysis_mask]
        return {
            "name": name,
            "n_analysis": int(analysis_mask.sum()),
            "n_underflow_ref_below_float32_tiny": int((~representable).sum()),
            "n_tau_below_0.02": int(near_t0.sum()),
            "max_abs_log_error": float(err_analysis.max()),
            "mean_abs_log_error": float(err_analysis.mean()),
            "p99_abs_log_error": float(torch.quantile(err_analysis, 0.99)),
            "max_rel_density_error": float(rel_analysis.max()),
            "mean_rel_density_error": float(rel_analysis.mean()),
            "max_abs_log_error_including_extreme_tails": float(err.max()),
        }

    small_mask = representable & ((tau / (a * a)) <= 0.8)
    small48 = wfpt_logpdf(
        t, v, a, w, t0, k_max=48, switch_thresh=float("inf")
    )
    small256 = wfpt_logpdf(
        t, v, a, w, t0, k_max=256, switch_thresh=float("inf")
    )
    small_err = (small48 - small256).abs()[small_mask]

    out = {
        "n": n,
        "seed": 123,
        "reference": "CPU float64, k_max=512",
        "analysis_rule": "tau >= 0.02 and reference log-density representable in float32",
        "cpu_float32": summarize(cpu32, "cpu_float32_k48"),
        "mps_float32": summarize(mps32, "mps_float32_k48"),
        "cpu_vs_mps_max_abs_log_error": float((cpu32.double() - mps32.double()).abs().max()),
        "small_time_j_truncation": {
            "n": int(small_mask.sum()),
            "k_max_production": 48,
            "k_max_reference": 256,
            "max_abs_log_error": float(small_err.max()),
            "mean_abs_log_error": float(small_err.mean()),
        },
    }
    Path("results").mkdir(exist_ok=True)
    with open("results/density_validation.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
