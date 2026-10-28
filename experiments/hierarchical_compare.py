"""Compare ddmtorch per-subject MLE with the PyMC hierarchical fit.

The comparison uses the Cavanagh et al. (2011) dataset and the same
HDDM-style hierarchical model: drift by stimulus condition, boundary,
non-decision time, w fixed at 0.5, and a fixed 5% uniform contaminant.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr

from ddmtorch import loglik
from ddmtorch.device import resolve_device, resolve_dtype

DATA_URL = "https://raw.githubusercontent.com/hddm-devs/hddm/master/hddm/examples/cavanagh_theta_nn.csv"


def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {DATA_URL} -> {path}")
        urllib.request.urlretrieve(DATA_URL, path)
    return pd.read_csv(path)


def softplus_inv(y: float) -> float:
    return math.log(math.expm1(y))


def logit(p: float) -> float:
    return math.log(p / (1.0 - p))


def fit_subject(
    rt: np.ndarray,
    choice: np.ndarray,
    stim_idx: np.ndarray,
    *,
    device: str = "auto",
    p_contam: float = 0.05,
    contam_max: float = 5.0,
    t0_max: float = 0.6,
    max_iter: int = 200,
    n_restarts: int = 3,
    seed: int = 0,
) -> dict:
    dev = resolve_device(device)
    dt = resolve_dtype(dev)
    rt_t = torch.as_tensor(np.array(rt, copy=True), dtype=dt, device=dev)
    ch_t = torch.as_tensor(np.array(choice, copy=True), dtype=dt, device=dev)
    st_t = torch.as_tensor(np.array(stim_idx, copy=True), dtype=torch.long, device=dev)

    init = torch.tensor(
        [0.20, 0.65, 0.20, softplus_inv(1.95), logit(0.50 / t0_max)],
        dtype=dt,
        device=dev,
    )
    best = None
    start = time.perf_counter()
    g = torch.Generator(device="cpu").manual_seed(seed)
    for restart in range(n_restarts):
        raw = init.clone() if restart == 0 else init + 0.1 * torch.randn(5, generator=g, dtype=dt).to(dev)
        raw.requires_grad_(True)
        opt = torch.optim.LBFGS(
            [raw],
            lr=0.5,
            max_iter=max_iter,
            history_size=10,
            tolerance_grad=1e-7,
            tolerance_change=1e-10,
            line_search_fn="strong_wolfe",
        )

        def closure():
            opt.zero_grad(set_to_none=True)
            v = raw[:3]
            a = F.softplus(raw[3]) + 0.05
            t0 = t0_max * torch.sigmoid(raw[4])
            lp = loglik(
                rt_t,
                ch_t,
                v[st_t],
                a,
                torch.full_like(rt_t, 0.5),
                t0,
                p_contam=p_contam,
                contam_min=0.0,
                contam_max=contam_max,
            )
            loss = -lp.mean()
            loss.backward()
            return loss

        opt.step(closure)
        with torch.no_grad():
            v = raw[:3]
            a = F.softplus(raw[3]) + 0.05
            t0 = t0_max * torch.sigmoid(raw[4])
            lp = loglik(
                rt_t,
                ch_t,
                v[st_t],
                a,
                torch.full_like(rt_t, 0.5),
                t0,
                p_contam=p_contam,
                contam_min=0.0,
                contam_max=contam_max,
            )
            nll_total = float(-lp.sum())
        if best is None or nll_total < best["nll_total"]:
            best = {
                "v_LL": float(v[0]),
                "v_WL": float(v[1]),
                "v_WW": float(v[2]),
                "a": float(a),
                "t0": float(t0),
                "nll_total": nll_total,
                "nll_per_trial": nll_total / len(rt_t),
                "restart": restart,
            }
    best["elapsed_s"] = time.perf_counter() - start
    best["n_trials"] = int(len(rt_t))
    return best


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/cavanagh_theta_nn.csv")
    parser.add_argument("--hier", default="results/hierarchical_cavanagh_subject.csv")
    parser.add_argument("--out", default="results/hierarchical_comparison.json")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--n-restarts", type=int, default=3)
    args = parser.parse_args()

    df = load_data(Path(args.data))
    hier = pd.read_csv(args.hier)
    stim_map = {"LL": 0, "WL": 1, "WW": 2}
    rows = []
    for subj in sorted(df["subj_idx"].unique()):
        d = df[df["subj_idx"] == subj]
        res = fit_subject(
            d["rt"].to_numpy(),
            d["response"].to_numpy(),
            d["stim"].map(stim_map).to_numpy(),
            device=args.device,
            n_restarts=args.n_restarts,
        )
        rows.append({"subj_idx": int(subj), **res})
        print(
            f"subject {subj:2d}: "
            f"v=({res['v_LL']:+.2f},{res['v_WL']:+.2f},{res['v_WW']:+.2f}) "
            f"a={res['a']:.2f} t0={res['t0']:.3f} "
            f"nll/trial={res['nll_per_trial']:.4f}",
            flush=True,
        )
    mle = pd.DataFrame(rows)
    mle.to_csv(args.out.replace(".json", "_mle.csv"), index=False)

    # Merge hierarchical subject means.
    hp = hier.pivot_table(index="subj_idx", columns=["param", "stim"], values="mean")
    hp.columns = [f"{p}_{s}" for p, s in hp.columns]
    hp = hp.reset_index()
    hp = hp.rename(columns={c: f"hier_{c}" for c in hp.columns if c != "subj_idx"})
    merged = mle.merge(hp, on="subj_idx", how="left")
    merged.to_csv(args.out.replace(".json", "_merged.csv"), index=False)

    comparisons = {}
    pairs = {
        "v_LL": ("hier_v_LL", "v_LL"),
        "v_WL": ("hier_v_WL", "v_WL"),
        "v_WW": ("hier_v_WW", "v_WW"),
        "a": ("hier_a_all", "a"),
        "t0": ("hier_t0_all", "t0"),
    }
    for name, (hcol, mcol) in pairs.items():
        x = merged[hcol].to_numpy()
        y = merged[mcol].to_numpy()
        err = y - x
        comparisons[name] = {
            "n": int(len(x)),
            "corr": float(pearsonr(x, y).statistic),
            "rmse": float(np.sqrt(np.mean(err**2))),
            "mean_diff_mle_minus_hier": float(np.mean(err)),
            "mle_mean": float(np.mean(y)),
            "mle_sd": float(np.std(y, ddof=1)),
            "hier_mean": float(np.mean(x)),
            "hier_sd": float(np.std(x, ddof=1)),
        }

    out = {
        "n_subjects": int(len(mle)),
        "n_trials": int(len(df)),
        "comparisons": comparisons,
        "mle_mean_time_s": float(mle["elapsed_s"].mean()),
        "seed": None,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(comparisons, indent=2))
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
