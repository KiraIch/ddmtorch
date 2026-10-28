"""HDDM-style hierarchical DDM in PyMC 5.

HDDM 1.0.1 depends on PyMC 2, which is not installable on Apple
Silicon/macOS.  This script fits the same model class (hierarchical
DDM with subject-specific drift-by-condition, boundary, and non-decision time)
using PyMC 5.  It is explicitly a reimplementation, not the HDDM package.

Dataset: Cavanagh et al. (2011) reward-based decision task, distributed as
``cavanagh_theta_nn.csv`` with the HDDM examples.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
from arviz import summary

DATA_URL = "https://raw.githubusercontent.com/hddm-devs/hddm/master/hddm/examples/cavanagh_theta_nn.csv"


def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {DATA_URL} -> {path}")
        urllib.request.urlretrieve(DATA_URL, path)
    return pd.read_csv(path)


def wfpt_logp(rt, choice, v, a, t0, w=0.5, k_max: int = 24):
    """Lower-boundary WFPT log density in PyTensor.

    ``choice`` is 1 for the upper boundary and 0 for the lower boundary;
    ``rhs`` is transformed by ``v -> -v`` for upper responses.  The starting
    point is fixed at 0.5 (unbiased).
    """
    tau = rt - t0
    valid = tau > 1e-6
    tau = pt.clip(tau, 1e-6, np.inf)
    a = pt.clip(a, 1e-4, np.inf)
    w = pt.clip(w, 1e-5, 1.0 - 1e-5)
    v_eff = pt.where(pt.eq(choice, 1), -v, v)
    scaled = tau / (a * a)
    use_large = scaled > 0.8

    j = pt.arange(1, k_max + 1, dtype="float64")
    j_ = j[:, None]
    large_terms = (
        j_
        * pt.sin(j_ * np.pi * w)
        * pt.exp(-(j_ * j_) * (np.pi**2) * tau[None, :] / (2 * a[None, :] ** 2))
    )
    large_sum = pt.sum(large_terms, axis=0)

    jj = pt.arange(-k_max, k_max + 1, dtype="float64")
    x = w + 2 * jj[:, None]
    small_terms = x * pt.exp(-(a[None, :] ** 2) * x * x / (2 * tau[None, :]))
    small_sum = pt.sum(small_terms, axis=0)

    log_mult_l = pt.log(np.pi) - 2 * pt.log(a) - v_eff * a * w - 0.5 * v_eff**2 * tau
    log_mult_s = (
        pt.log(a)
        - 0.5 * pt.log(2 * np.pi)
        - 1.5 * pt.log(tau)
        - v_eff * a * w
        - 0.5 * v_eff**2 * tau
    )
    tiny = 1e-300
    logp = pt.where(
        use_large,
        log_mult_l + pt.log(pt.maximum(large_sum, tiny)),
        log_mult_s + pt.log(pt.maximum(small_sum, tiny)),
    )
    return pt.where(valid, logp, -1e30)


def mixture_logp(rt, choice, v, a, t0, *, w=0.5, k_max=24,
                 p_contam=0.05, contam_min=0.0, contam_max=5.0):
    """WFPT log density mixed with a fixed-proportion uniform contaminant."""
    logp_ddm = wfpt_logp(rt, choice, v, a, t0, w=w, k_max=k_max)
    if p_contam <= 0:
        return logp_ddm
    logp_contam = np.log(0.5) - np.log(contam_max - contam_min)
    in_range = (rt >= contam_min) & (rt <= contam_max)
    logp_contam = pt.where(in_range, logp_contam, -1e30)
    return pt.logaddexp(
        np.log1p(-p_contam) + logp_ddm,
        np.log(p_contam) + logp_contam,
    )


def fit_hierarchical(
    df: pd.DataFrame,
    subjects: list[int],
    *,
    draws: int = 1000,
    tune: int = 1000,
    chains: int = 4,
    t0_max: float = 0.6,
    k_max: int = 24,
    p_contam: float = 0.05,
    contam_max: float = 5.0,
    seed: int = 2026,
):
    d = df[df["subj_idx"].isin(subjects)].copy()
    subject_map = {s: i for i, s in enumerate(subjects)}
    stim_map = {"LL": 0, "WL": 1, "WW": 2}
    sidx = d["subj_idx"].map(subject_map).to_numpy(dtype=int)
    cidx = d["stim"].map(stim_map).to_numpy(dtype=int)
    rt = d["rt"].to_numpy(dtype=float)
    choice = d["response"].to_numpy(dtype=float)

    S = len(subjects)
    C = 3
    with pm.Model() as model:
        mu_v = pm.Normal("mu_v", 0.0, 2.0, shape=C)
        sigma_v = pm.HalfNormal("sigma_v", 1.0, shape=C)
        v_raw = pm.Normal("v_raw", 0.0, 1.0, shape=(S, C))
        v = pm.Deterministic("v", mu_v[None, :] + sigma_v[None, :] * v_raw)

        mu_loga = pm.Normal("mu_loga", np.log(1.2), 0.5)
        sigma_loga = pm.HalfNormal("sigma_loga", 0.5)
        a_raw = pm.Normal("a_raw", 0.0, 1.0, shape=S)
        a = pm.Deterministic("a", pt.exp(mu_loga + sigma_loga * a_raw))

        mu_t0 = pm.Normal("mu_t0", 0.0, 1.0)
        sigma_t0 = pm.HalfNormal("sigma_t0", 0.5)
        t0_raw = pm.Normal("t0_raw", 0.0, 1.0, shape=S)
        t0 = pm.Deterministic("t0", t0_max * pt.sigmoid(mu_t0 + sigma_t0 * t0_raw))

        v_i = v[sidx, cidx]
        a_i = a[sidx]
        t0_i = t0[sidx]
        pm.Potential(
            "lik",
            mixture_logp(
                rt,
                choice,
                v_i,
                a_i,
                t0_i,
                k_max=k_max,
                p_contam=p_contam,
                contam_max=contam_max,
            ).sum(),
        )

        start = time.perf_counter()
        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            cores=chains,
            target_accept=0.9,
            random_seed=seed,
            nuts_sampler="nutpie",
            progressbar=False,
        )
        elapsed = time.perf_counter() - start
    return idata, d, subject_map, elapsed


def summarize(idata, d, subject_map, elapsed):
    vars_ = ["mu_v", "sigma_v", "mu_loga", "sigma_loga", "mu_t0", "sigma_t0"]
    sm = summary(idata, var_names=vars_, round_to="none")
    S = len(subject_map)
    C = 3
    v_post = idata.posterior["v"].values.reshape(-1, S, C)
    a_post = idata.posterior["a"].values.reshape(-1, S)
    t0_post = idata.posterior["t0"].values.reshape(-1, S)

    subject_rows = []
    for s, i in subject_map.items():
        for c, name in enumerate(["LL", "WL", "WW"]):
            post = v_post[:, i, c]
            subject_rows.append(
                {
                    "subj_idx": int(s),
                    "stim": name,
                    "param": "v",
                    "mean": float(np.mean(post)),
                    "sd": float(np.std(post)),
                    "hdi_low": float(np.quantile(post, 0.025)),
                    "hdi_high": float(np.quantile(post, 0.975)),
                }
            )
        for param, post in [("a", a_post[:, i]), ("t0", t0_post[:, i])]:
            subject_rows.append(
                {
                    "subj_idx": int(s),
                    "stim": "all",
                    "param": param,
                    "mean": float(np.mean(post)),
                    "sd": float(np.std(post)),
                    "hdi_low": float(np.quantile(post, 0.025)),
                    "hdi_high": float(np.quantile(post, 0.975)),
                }
            )

    group_summary = {}
    for var in vars_:
        arr = idata.posterior[var].values
        flat = arr.reshape(arr.shape[0] * arr.shape[1], -1)
        if flat.shape[1] == 1:
            group_summary[var] = {
                "mean": float(np.mean(flat)),
                "sd": float(np.std(flat)),
                "hdi_low": float(np.quantile(flat, 0.025)),
                "hdi_high": float(np.quantile(flat, 0.975)),
            }
        else:
            group_summary[var] = [
                {
                    "index": int(j),
                    "mean": float(np.mean(flat[:, j])),
                    "sd": float(np.std(flat[:, j])),
                    "hdi_low": float(np.quantile(flat[:, j], 0.025)),
                    "hdi_high": float(np.quantile(flat[:, j], 0.975)),
                }
                for j in range(flat.shape[1])
            ]

    return {
        "elapsed_s": elapsed,
        "n_subjects": S,
        "n_trials": int(len(d)),
        "max_rhat": float(sm["r_hat"].max()),
        "min_ess_bulk": float(sm["ess_bulk"].min()),
        "group_summary": group_summary,
        "subject_rows": subject_rows,
    }, sm

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/cavanagh_theta_nn.csv")
    parser.add_argument("--subjects", default="all", help="'all' or comma-separated ids")
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--tune", type=int, default=1000)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--out", default="results/hierarchical_cavanagh.json")
    args = parser.parse_args()

    df = load_data(Path(args.data))
    all_subjects = sorted(df["subj_idx"].unique().tolist())
    if args.subjects == "all":
        subjects = all_subjects
    else:
        subjects = [int(x) for x in args.subjects.split(",")]

    print(f"Fitting {len(subjects)} subjects, {len(df[df.subj_idx.isin(subjects)])} trials", flush=True)
    idata, d, subject_map, elapsed = fit_hierarchical(
        df,
        subjects,
        draws=args.draws,
        tune=args.tune,
        chains=args.chains,
        seed=args.seed,
    )
    out, sm = summarize(idata, d, subject_map, elapsed)
    out.update(
        {
            "subjects": subjects,
            "seed": args.seed,
            "draws": args.draws,
            "tune": args.tune,
            "chains": args.chains,
            "t0_max": 0.6,
            "k_max": 24,
            "p_contam": 0.05,
        }
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    sm.to_csv(out_path.with_name("hierarchical_cavanagh_summary.csv"))
    pd.DataFrame(out["subject_rows"]).to_csv(
        out_path.with_name("hierarchical_cavanagh_subject.csv"), index=False
    )
    print(json.dumps({k: v for k, v in out.items() if k != "subject_rows"}, indent=2))
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
