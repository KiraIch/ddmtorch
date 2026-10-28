"""Independent simulator check: recover parameters from Euler-Maruyama data.

This guards against an "inverse crime" in which the synthetic data and the
likelihood use exactly the same numerical density.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch

from ddmtorch import fit_mle
from ddmtorch.simulate import simulate_wfpt_euler
from experiments.recovery import recovery_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-datasets", type=int, default=20)
    parser.add_argument("--n-trials", type=int, default=500)
    parser.add_argument("--dt", type=float, default=1e-3)
    parser.add_argument("--max-time", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=77)
    parser.add_argument("--out", default="results/euler_recovery.npz")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    rows = []
    start = time.perf_counter()
    for i in range(args.n_datasets):
        true = np.array(
            [
                rng.uniform(-2.0, 2.0),
                rng.uniform(0.7, 2.0),
                rng.uniform(0.35, 0.65),
                rng.uniform(0.10, 0.50),
            ]
        )
        rt, choice = simulate_wfpt_euler(
            float(true[0]),
            float(true[1]),
            float(true[2]),
            float(true[3]),
            n=args.n_trials,
            dt=args.dt,
            max_time=args.max_time,
            device="cpu",
            dtype=torch.float64,
            generator=torch.Generator().manual_seed(args.seed + i),
        )
        keep = torch.isfinite(rt)
        if keep.sum() < 0.8 * args.n_trials:
            print(f"dataset {i}: only {int(keep.sum())} resolved trials; skipped", flush=True)
            continue
        res = fit_mle(
            rt[keep],
            choice[keep],
            device="auto",
            max_iter=100,
            seed=args.seed + i,
        )
        est = res.params[0]
        rows.append(
            {
                "dataset": i,
                "true_v": true[0],
                "true_a": true[1],
                "true_w": true[2],
                "true_t0": true[3],
                "est_v": est[0],
                "est_a": est[1],
                "est_w": est[2],
                "est_t0": est[3],
                "n_used": int(keep.sum()),
                "nll": res.nll,
                "elapsed_s": res.elapsed_s,
                "success": res.success,
            }
        )
        print(
            f"[{i + 1:2d}/{args.n_datasets}] "
            f"true=({true[0]:+.2f},{true[1]:.2f},{true[2]:.2f},{true[3]:.2f}) "
            f"est=({est[0]:+.2f},{est[1]:.2f},{est[2]:.2f},{est[3]:.2f}) "
            f"used={int(keep.sum())}",
            flush=True,
        )

    df = pd.DataFrame(rows)
    true = df[["true_v", "true_a", "true_w", "true_t0"]].to_numpy()
    est = df[["est_v", "est_a", "est_w", "est_t0"]].to_numpy()
    metrics = recovery_metrics(true, est)
    summary = {
        "n_datasets": len(df),
        "n_trials_requested": args.n_trials,
        "dt": args.dt,
        "max_time": args.max_time,
        "total_elapsed_s": time.perf_counter() - start,
        "metrics": metrics,
    }
    np.savez(args.out, true=true, est=est)
    df.to_csv(Path(args.out).with_suffix(".csv"), index=False)
    with open(Path(args.out).with_suffix(".json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
