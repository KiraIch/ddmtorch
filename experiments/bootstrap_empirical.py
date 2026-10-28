"""Bootstrap percentile intervals for the empirical Roitman-Shadlen fits.

Resamples trials within each monkey, refits the same model with ddmtorch,
and reports 95% percentile intervals.  This is a nonparametric bootstrap
over trials, not over subjects.
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

from experiments.empirical_roitman import fit_ours, load_data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/roitman_rts.csv")
    parser.add_argument("--fits", default="results/empirical_roitman.json")
    parser.add_argument("--n-boot", type=int, default=500)
    parser.add_argument("--max-iter", type=int, default=120)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--out", default="results/empirical_bootstrap.json")
    args = parser.parse_args()

    df = load_data(Path(args.data))
    with open(args.fits) as f:
        fits = {int(r["monkey"]): r for r in json.load(f)}

    rng = np.random.default_rng(args.seed)
    all_samples = []
    summary = {}
    start_all = time.perf_counter()
    for monkey in sorted(df["monkey"].unique()):
        d = df[df["monkey"] == monkey].reset_index(drop=True)
        full = fits[int(monkey)]
        init = {
            "beta": full["ddm_beta"],
            "a": full["ddm_a"],
            "w": full["ddm_w"],
            "t0": full["ddm_t0"],
        }
        rows = []
        print(f"Monkey {monkey}: {len(d)} trials, {args.n_boot} bootstrap fits", flush=True)
        for b in range(args.n_boot):
            idx = rng.integers(0, len(d), size=len(d))
            sample = d.iloc[idx]
            res = fit_ours(
                sample["rt"].to_numpy(),
                sample["correct"].to_numpy(),
                sample["coh"].to_numpy(),
                device=args.device,
                max_iter=args.max_iter,
                n_restarts=1,
                seed=args.seed + b,
                init=init,
            )
            rows.append(
                {
                    "monkey": int(monkey),
                    "b": b,
                    "beta": res["beta"],
                    "a": res["a"],
                    "w": res["w"],
                    "t0": res["t0"],
                    "nll": res["nll"],
                }
            )
            if (b + 1) % 50 == 0 or b == 0:
                print(f"  {b + 1}/{args.n_boot}", flush=True)
        all_samples.extend(rows)
        arr = np.array([[r["beta"], r["a"], r["w"], r["t0"]] for r in rows])
        arr = arr[np.isfinite(arr).all(axis=1)]
        summary[str(int(monkey))] = {
            "n_boot": int(len(arr)),
            "mean": dict(zip(["beta", "a", "w", "t0"], arr.mean(axis=0).tolist())),
            "sd": dict(zip(["beta", "a", "w", "t0"], arr.std(axis=0, ddof=1).tolist())),
            "ci95": {
                name: [
                    float(np.percentile(arr[:, j], 2.5)),
                    float(np.percentile(arr[:, j], 97.5)),
                ]
                for j, name in enumerate(["beta", "a", "w", "t0"])
            },
        }
        print(f"  summary: {summary[str(int(monkey))]}", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_samples).to_csv(out.with_name("empirical_bootstrap_samples.csv"), index=False)
    with open(out, "w") as f:
        json.dump(
            {
                "n_boot_requested": args.n_boot,
                "seed": args.seed,
                "elapsed_s": time.perf_counter() - start_all,
                "summary": summary,
            },
            f,
            indent=2,
        )
    print(f"Saved {out}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
