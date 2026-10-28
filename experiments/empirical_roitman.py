"""Empirical application: Roitman & Shadlen (2002) motion discrimination data.

Model: drift proportional to motion coherence, constant bound, unbiased
starting point (w = 0.5), non-decision time, and a fixed 2% uniform
contaminant.  This mirrors the PyDDM tutorial model for the same data, so
the two toolboxes can be compared on the same likelihood.

Data source:
https://github.com/mwshinn/PyDDM/blob/master/doc/downloads/roitman_rts.csv
(processed from the public Roitman & Shadlen dataset).
"""

from __future__ import annotations

import argparse
import json
import math
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from ddmtorch import choice_probability, loglik
from ddmtorch.device import resolve_device, resolve_dtype

DATA_URL = "https://raw.githubusercontent.com/mwshinn/PyDDM/master/doc/downloads/roitman_rts.csv"


def softplus_inv(y: float) -> float:
    return math.log(math.expm1(y))


def logit(p: float) -> float:
    return math.log(p / (1.0 - p))


def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {DATA_URL} -> {path}")
        urllib.request.urlretrieve(DATA_URL, path)
    df = pd.read_csv(path)
    df = df[(df["rt"] > 0.1) & (df["rt"] < 1.65)].copy()
    return df


def unpack(
    raw: torch.Tensor, t0_max: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    beta = F.softplus(raw[0])
    a = F.softplus(raw[1]) + 0.05
    t0 = t0_max * torch.sigmoid(raw[2])
    w = torch.sigmoid(raw[3])
    return beta, a, t0, w


def fit_ours(
    rt: np.ndarray,
    choice: np.ndarray,
    coh: np.ndarray,
    *,
    device: str = "auto",
    p_contam: float = 0.02,
    contam_max: float = 2.0,
    t0_max: float = 0.6,
    max_iter: int = 200,
    n_restarts: int = 3,
    seed: int = 0,
    init: dict | None = None,
) -> dict:
    dev = resolve_device(device)
    dt = resolve_dtype(dev)
    rt_t = torch.as_tensor(np.array(rt, copy=True), dtype=dt, device=dev)
    ch_t = torch.as_tensor(np.array(choice, copy=True), dtype=dt, device=dev)
    coh_t = torch.as_tensor(np.array(coh, copy=True), dtype=dt, device=dev)

    # Plausible initial values from the DDM literature / PyDDM tutorial.
    if init is None:
        init_vec = [softplus_inv(10.0), softplus_inv(1.45), logit(0.31 / t0_max), 0.0]
    else:
        init_vec = [
            softplus_inv(float(init["beta"])),
            softplus_inv(float(init["a"]) - 0.05),
            logit(float(init["t0"]) / t0_max),
            logit(float(init["w"])),
        ]
    init = torch.tensor(init_vec, dtype=dt, device=dev)
    best = None
    start = time.perf_counter()
    g = torch.Generator(device="cpu").manual_seed(seed)
    for restart in range(n_restarts):
        if restart == 0:
            raw = init.clone()
        else:
            raw = init + 0.15 * torch.randn(4, generator=g, dtype=dt).to(dev)
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
            beta, a, t0, w = unpack(raw, t0_max)
            v = beta * coh_t
            lp = loglik(
                rt_t,
                ch_t,
                v,
                a,
                w,
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
            beta, a, t0, w = unpack(raw, t0_max)
            v = beta * coh_t
            lp = loglik(
                rt_t,
                ch_t,
                v,
                a,
                w,
                t0,
                p_contam=p_contam,
                contam_min=0.0,
                contam_max=contam_max,
            )
            nll = float(-lp.sum())
        if best is None or nll < best["nll"]:
            best = {
                "nll": nll,
                "nll_per_trial": nll / len(rt_t),
                "beta": float(beta),
                "a": float(a),
                "t0": float(t0),
                "w": float(w),
                "p_contam": p_contam,
                "contam_max": contam_max,
                "restart": restart,
            }
    best["elapsed_s"] = time.perf_counter() - start
    best["n_trials"] = int(len(rt_t))
    best["device"] = str(dev)
    return best


def fit_pyddm(df: pd.DataFrame, monkey: int, *, seed: int = 0) -> dict:
    import pyddm
    from pyddm import Sample
    from pyddm.functions import get_model_loss

    d = df[df["monkey"] == monkey].copy()
    sample = Sample.from_pandas_dataframe(
        d, rt_column_name="rt", choice_column_name="correct"
    )
    model = pyddm.gddm(
        drift=lambda coh, driftcoh: driftcoh * coh,
        noise=1,
        bound="b",
        starting_position="x0",  # PyDDM x0 in [-1, 1]; 0 is unbiased
        nondecision="ndtime",
        mixture_coef=0.02,
        mixture_distribution=None,  # uniform mixture, matching ddmtorch
        T_dur=2.0,
        parameters={
            "driftcoh": (0, 20),
            "b": (0.4, 3),
            "x0": (-0.9, 0.9),
            "ndtime": (0, 0.5),
        },
        conditions=["coh"],
    )
    start = time.perf_counter()
    model.fit(sample=sample, verbose=False, fitparams={"seed": seed})
    elapsed = time.perf_counter() - start
    p = model.parameters()
    nll = float(get_model_loss(model, sample))
    return {
        "nll": nll,
        "nll_per_trial": nll / len(d),
        "beta": float(p["drift"]["driftcoh"]),
        "B": float(p["bound"]["B"]),
        "a": 2.0 * float(p["bound"]["B"]),  # convert PyDDM bound to separation
        "t0": float(p["overlay"]["nondectime"]),
        "w": float(p["IC"]["x0"]) / 2.0 + 0.5,
        "elapsed_s": elapsed,
        "n_trials": int(len(d)),
        "model": model,
        "sample": sample,
    }


def psychometric(model, df: pd.DataFrame) -> dict:
    d = df.copy()
    obs = d.groupby("coh")["correct"].agg(["mean", "count"]).reset_index()
    preds = []
    for coh in obs["coh"]:
        sol = model.solve(conditions={"coh": float(coh)})
        preds.append(float(sol.prob("upper")))
    obs["pred_pyddm"] = preds
    return obs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/roitman_rts.csv")
    parser.add_argument("--out", default="results/empirical_roitman.json")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-iter", type=int, default=200)
    parser.add_argument("--restarts", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    df = load_data(Path(args.data))
    rows = []
    plot_rows = []
    for monkey in sorted(df["monkey"].unique()):
        d = df[df["monkey"] == monkey]
        print(f"\nMonkey {monkey}: {len(d)} trials", flush=True)
        ours = fit_ours(
            d["rt"].to_numpy(),
            d["correct"].to_numpy(),
            d["coh"].to_numpy(),
            device=args.device,
            max_iter=args.max_iter,
            n_restarts=args.restarts,
            seed=args.seed,
        )
        print("ddmtorch", {k: round(v, 4) if isinstance(v, float) else v for k, v in ours.items()}, flush=True)

        pyddm_res = fit_pyddm(df, monkey, seed=args.seed)
        print(
            "PyDDM",
            {
                k: round(v, 4) if isinstance(v, float) else v
                for k, v in pyddm_res.items()
                if k not in {"model", "sample"}
            },
            flush=True,
        )
        rows.append(
            {
                "monkey": int(monkey),
                "n_trials": ours["n_trials"],
                "ddm_beta": ours["beta"],
                "ddm_a": ours["a"],
                "ddm_w": ours["w"],
                "ddm_t0": ours["t0"],
                "ddm_nll": ours["nll"],
                "ddm_nll_per_trial": ours["nll_per_trial"],
                "ddm_elapsed_s": ours["elapsed_s"],
                "pyddm_beta": pyddm_res["beta"],
                "pyddm_a": pyddm_res["a"],
                "pyddm_w": pyddm_res["w"],
                "pyddm_t0": pyddm_res["t0"],
                "pyddm_nll": pyddm_res["nll"],
                "pyddm_nll_per_trial": pyddm_res["nll_per_trial"],
                "pyddm_elapsed_s": pyddm_res["elapsed_s"],
                "seed": args.seed,
            }
        )
        obs = psychometric(pyddm_res["model"], d)
        obs["monkey"] = monkey
        pred_ours = choice_probability(
            ours["beta"] * obs["coh"].to_numpy(),
            ours["a"],
            ours["w"],
        )
        obs["pred_ours"] = pred_ours.detach().cpu().numpy()
        plot_rows.append(obs)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out.with_suffix(".csv"), index=False)
    with open(out, "w") as f:
        json.dump(rows, f, indent=2)
    pd.concat(plot_rows, ignore_index=True).to_csv(
        out.with_name("empirical_roitman_psychometric.csv"), index=False
    )
    print(f"\nSaved {out} and {out.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
