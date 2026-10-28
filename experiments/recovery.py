"""Synthetic parameter-recovery experiment.

Run with:
    conda run -n ddm python experiments/recovery.py --n-datasets 100 --n-trials 500
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr

from ddmtorch import fit_mle, sample_wfpt


def recovery_metrics(true: np.ndarray, est: np.ndarray) -> dict:
    out = {}
    names = ["v", "a", "w", "t0"]
    for j, name in enumerate(names):
        err = est[:, j] - true[:, j]
        r = float(pearsonr(true[:, j], est[:, j]).statistic)
        out[name] = {
            "bias": float(np.mean(err)),
            "mae": float(np.mean(np.abs(err))),
            "rmse": float(np.sqrt(np.mean(err**2))),
            "corr": r,
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-datasets", type=int, default=100)
    parser.add_argument("--n-trials", type=int, default=500)
    parser.add_argument("--n-grid", type=int, default=8192)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--fit-device", default="auto")
    parser.add_argument("--out", default="results/recovery.npz")
    parser.add_argument("--csv", default="results/recovery.csv")
    args = parser.parse_args()

    out_path = Path(args.out)
    csv_path = Path(args.csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    rows = []
    start_all = time.perf_counter()

    for i in range(args.n_datasets):
        true = np.array(
            [
                rng.uniform(-2.0, 2.0),
                rng.uniform(0.7, 2.0),
                rng.uniform(0.35, 0.65),
                rng.uniform(0.10, 0.50),
            ]
        )
        rt, choice = sample_wfpt(
            float(true[0]),
            float(true[1]),
            float(true[2]),
            float(true[3]),
            n=args.n_trials,
            device="cpu",
            dtype=torch.float64,
            t_max=10.0,
            n_grid=args.n_grid,
            generator=torch.Generator().manual_seed(args.seed + i),
        )
        res = fit_mle(
            rt,
            choice,
            device=args.fit_device,
            max_iter=args.max_iter,
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
                "nll": res.nll,
                "elapsed_s": res.elapsed_s,
                "n_iter": res.n_iter,
                "success": res.success,
                "device": res.device,
            }
        )
        if (i + 1) % 10 == 0 or i == 0:
            print(
                f"[{i + 1:3d}/{args.n_datasets}] "
                f"true=({true[0]:+.2f},{true[1]:.2f},{true[2]:.2f},{true[3]:.2f}) "
                f"est=({est[0]:+.2f},{est[1]:.2f},{est[2]:.2f},{est[3]:.2f}) "
                f"{res.elapsed_s:.2f}s",
                flush=True,
            )

    df = pd.DataFrame(rows)
    true = df[["true_v", "true_a", "true_w", "true_t0"]].to_numpy()
    est = df[["est_v", "est_a", "est_w", "est_t0"]].to_numpy()
    metrics = recovery_metrics(true, est)
    summary = {
        "n_datasets": args.n_datasets,
        "n_trials": args.n_trials,
        "n_grid": args.n_grid,
        "max_iter": args.max_iter,
        "fit_device": str(df["device"].iloc[0]),
        "success_rate": float(df["success"].mean()),
        "total_elapsed_s": time.perf_counter() - start_all,
        "mean_fit_elapsed_s": float(df["elapsed_s"].mean()),
        "metrics": metrics,
    }

    np.savez(
        out_path,
        true=true,
        est=est,
        nll=df["nll"].to_numpy(),
        elapsed_s=df["elapsed_s"].to_numpy(),
        success=df["success"].to_numpy(),
    )
    df.to_csv(csv_path, index=False)
    with open(out_path.with_suffix(".json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\nRecovery summary")
    print(json.dumps(metrics, indent=2))
    print(f"\nSaved {out_path}, {csv_path}, {out_path.with_suffix('.json')}")


if __name__ == "__main__":
    main()
