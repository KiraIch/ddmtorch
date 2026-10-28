"""Illustrative synthetic check: a fixed contaminant mixture recovers t0.

Generates a 2,000-trial DDM dataset, replaces 5% of trials with uniform
contaminants (RT 0.1-0.5 s, random choice), and compares a no-mixture MLE
with a 5% uniform-mixture MLE.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from ddmtorch import fit_mle, sample_wfpt

TRUE = {"v": 1.0, "a": 1.2, "w": 0.5, "t0": 0.32}


def main() -> None:
    torch.manual_seed(0)
    rt, choice = sample_wfpt(
        TRUE["v"],
        TRUE["a"],
        TRUE["w"],
        TRUE["t0"],
        n=2000,
        device="cpu",
        dtype=torch.float64,
        t_max=8.0,
        n_grid=8192,
    )
    g = torch.Generator().manual_seed(1)
    n_cont = 100
    idx = torch.randperm(len(rt), generator=g)[:n_cont]
    rt[idx] = 0.1 + 0.4 * torch.rand(n_cont, generator=g, dtype=torch.float64)
    choice[idx] = torch.randint(0, 2, (n_cont,), generator=g).double()

    no_mix = fit_mle(rt, choice, device="auto", max_iter=120, seed=2)
    mix = fit_mle(
        rt,
        choice,
        device="auto",
        max_iter=120,
        seed=2,
        p_contam=0.05,
        t0_max=0.8,
        t0_frac=0.4,
    )
    out = {
        "true": TRUE,
        "seed": 0,
        "contaminant_seed": 1,
        "fit_seed": 2,
        "n_trials": int(len(rt)),
        "n_contaminants": n_cont,
        "contaminant_fraction": n_cont / len(rt),
        "no_mixture": {
            "v": float(no_mix.params[0, 0]),
            "a": float(no_mix.params[0, 1]),
            "w": float(no_mix.params[0, 2]),
            "t0": float(no_mix.params[0, 3]),
            "nll": float(no_mix.nll),
        },
        "mixture_5pct": {
            "v": float(mix.params[0, 0]),
            "a": float(mix.params[0, 1]),
            "w": float(mix.params[0, 2]),
            "t0": float(mix.params[0, 3]),
            "nll": float(mix.nll),
        },
    }
    Path("results").mkdir(exist_ok=True)
    with open("results/contaminant_check.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
