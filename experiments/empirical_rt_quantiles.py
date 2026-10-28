"""Observed vs predicted RT quantiles for the Roitman-Shadlen application.

Predictions use the ddmtorch empirical fit (drift proportional to coherence,
constant bound, free starting point, non-decision time, fixed 2% uniform
contaminant).  Quantiles are computed for correct and error responses at each
coherence level where enough error trials exist.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch

from ddmtorch import wfpt_pdf
from experiments.empirical_roitman import load_data

QUANTILES = np.array([0.1, 0.3, 0.5, 0.7, 0.9])


def predicted_quantiles(
    beta: float,
    a: float,
    w: float,
    t0: float,
    coh: float,
    boundary: str,
    *,
    p_contam: float = 0.02,
    contam_max: float = 2.0,
    n_grid: int = 12000,
) -> np.ndarray:
    t = torch.linspace(t0 + 1e-4, contam_max, n_grid, dtype=torch.float64)
    dens = wfpt_pdf(t, beta * coh, a, w, t0, boundary=boundary)
    dens = (1.0 - p_contam) * dens + p_contam * 0.5 / contam_max
    cdf = torch.cumsum(dens, dim=0) * (t[1] - t[0])
    cdf = cdf / cdf[-1]
    q = torch.as_tensor(QUANTILES, dtype=torch.float64)
    idx = torch.searchsorted(cdf, q)
    idx = torch.clamp(idx, 1, n_grid - 1)
    c0 = cdf[idx - 1]
    c1 = cdf[idx]
    t0v = t[idx - 1]
    t1v = t[idx]
    frac = (q - c0) / (c1 - c0).clamp_min(1e-12)
    return (t0v + frac * (t1v - t0v)).numpy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/roitman_rts.csv")
    parser.add_argument("--fits", default="results/empirical_roitman.json")
    parser.add_argument("--out", default="results/empirical_rt_quantiles.csv")
    args = parser.parse_args()

    df = load_data(Path(args.data))
    with open(args.fits) as f:
        fits = {int(r["monkey"]): r for r in json.load(f)}

    rows = []
    for monkey in sorted(df["monkey"].unique()):
        d = df[df["monkey"] == monkey]
        fit = fits[int(monkey)]
        for coh in sorted(d["coh"].unique()):
            sub = d[d["coh"] == coh]
            for correct, boundary in [(1, "upper"), (0, "lower")]:
                obs = sub[sub["correct"] == correct]["rt"].to_numpy()
                if len(obs) < 20:
                    continue
                obs_q = np.quantile(obs, QUANTILES)
                pred_q = predicted_quantiles(
                    fit["ddm_beta"],
                    fit["ddm_a"],
                    fit["ddm_w"],
                    fit["ddm_t0"],
                    float(coh),
                    boundary,
                )
                for q, o, p in zip(QUANTILES, obs_q, pred_q):
                    rows.append(
                        {
                            "monkey": int(monkey),
                            "coh": float(coh),
                            "correct": int(correct),
                            "quantile": float(q),
                            "observed": float(o),
                            "predicted": float(p),
                        }
                    )
    out = pd.DataFrame(rows)
    out.to_csv(args.out, index=False)
    rmse = np.sqrt(np.mean((out["predicted"] - out["observed"]) ** 2))
    print(f"n quantile points: {len(out)}")
    print(f"RMSE observed vs predicted RT quantiles: {rmse:.4f} s")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
