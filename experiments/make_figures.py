"""Create paper figures from saved experiment results."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from ddmtorch import wfpt_pdf

FIGDIR = Path("paper/figures")
FIGDIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(
    {
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "figure.dpi": 150,
        "savefig.bbox": "tight",
    }
)


def recovery_scatter() -> None:
    d = np.load("results/recovery.npz")
    true, est = d["true"], d["est"]
    names = ["Drift rate $v$", "Boundary $a$", "Start point $w$", "Non-decision $t_0$"]
    units = ["", "", "", " s"]
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 6.2))
    for j, ax in enumerate(axes.ravel()):
        t = true[:, j]
        e = est[:, j]
        err = e - t
        r = np.corrcoef(t, e)[0, 1]
        rmse = np.sqrt(np.mean(err**2))
        ax.scatter(t, e, s=12, alpha=0.7, edgecolors="none", color="#2a6f97")
        lo = min(t.min(), e.min())
        hi = max(t.max(), e.max())
        pad = 0.05 * (hi - lo)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", lw=0.8, alpha=0.7)
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_xlabel(f"True {names[j]}{units[j]}")
        ax.set_ylabel(f"Estimated {names[j]}{units[j]}")
        ax.set_title(f"{names[j]}: r = {r:.3f}, RMSE = {rmse:.3f}")
        ax.set_aspect("equal", adjustable="box")
    fig.suptitle("Synthetic parameter recovery (100 datasets, 500 trials each)", y=1.02)
    fig.tight_layout()
    fig.savefig(FIGDIR / "recovery_scatter.pdf")
    fig.savefig(FIGDIR / "recovery_scatter.png")
    plt.close(fig)


def benchmark_scaling() -> None:
    df = pd.read_csv("results/benchmark.csv")
    d = df[df["task"] == "density_fwd_bwd"].dropna(subset=["elapsed_s"])
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    styles = {
        ("cpu", "float64"): ("CPU float64", "#1f4e79", "o", "-"),
        ("cpu", "float32"): ("CPU float32", "#6c757d", "s", "--"),
        ("mps", "float32"): ("Apple M5 MPS (float32)", "#c1121f", "^", "-"),
    }
    for key, (label, color, marker, ls) in styles.items():
        sub = d[(d["device"] == key[0]) & (d["dtype"] == key[1])].sort_values("n_trials")
        if len(sub):
            ax.plot(
                sub["n_trials"],
                sub["elapsed_s"],
                marker=marker,
                color=color,
                linestyle=ls,
                label=label,
                ms=4,
            )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of trials")
    ax.set_ylabel("Forward + backward time (s)")
    ax.set_title("Vectorized WFPT likelihood evaluation")
    ax.grid(True, which="both", alpha=0.25, lw=0.5)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGDIR / "benchmark_scaling.pdf")
    fig.savefig(FIGDIR / "benchmark_scaling.png")
    plt.close(fig)


def benchmark_mle() -> None:
    df = pd.read_csv("results/benchmark.csv")
    d = df[df["task"] == "mle"].dropna(subset=["elapsed_s"])
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    labels = []
    width = 0.26
    groups = [(2000, 1), (10000, 1), (50000, 20)]
    colors = {
        ("cpu", "float64"): "#1f4e79",
        ("cpu", "float32"): "#6c757d",
        ("mps", "float32"): "#c1121f",
    }
    for i, (n, s) in enumerate(groups):
        labels.append(f"{n // 1000}k\n({s} subj.)" if s > 1 else f"{n // 1000}k")
        for j, ((dev, dtype), color) in enumerate(colors.items()):
            sub = d[
                (d["device"] == dev)
                & (d["dtype"] == dtype)
                & (d["n_trials"] == n)
                & (d["notes"].str.contains(f"n_subjects={s},", regex=False))
            ]
            if len(sub):
                ax.bar(
                    i + (j - 1) * width,
                    float(sub["elapsed_s"].iloc[0]),
                    width=width,
                    color=color,
                    label={
                        ("cpu", "float64"): "CPU float64",
                        ("cpu", "float32"): "CPU float32",
                        ("mps", "float32"): "MPS float32",
                    }[ (dev, dtype) ]
                    if i == 0
                    else None,
                )
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("MLE wall time (s)")
    ax.set_title("Joint maximum-likelihood fits")
    ax.grid(True, axis="y", alpha=0.25, lw=0.5)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGDIR / "benchmark_mle.pdf")
    fig.savefig(FIGDIR / "benchmark_mle.png")
    plt.close(fig)


def model_illustration() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))
    a_values = [0.8, 1.2, 1.6]
    for a in a_values:
        t = torch.linspace(0.31, 3.0, 600, dtype=torch.float64)
        dens = wfpt_pdf(t, 0.8, a, 0.5, t0=0.3, boundary="upper")
        axes[0].plot(t.numpy(), dens.numpy(), label=f"a = {a}")
    axes[0].set_xlabel("Response time (s)")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Boundary separation")
    axes[0].legend(frameon=False)

    for v in [-1.0, 0.0, 1.0]:
        t = torch.linspace(0.31, 3.0, 600, dtype=torch.float64)
        dens = wfpt_pdf(t, v, 1.2, 0.5, t0=0.3, boundary="upper")
        axes[1].plot(t.numpy(), dens.numpy(), label=f"v = {v:+.0f}")
    axes[1].set_xlabel("Response time (s)")
    axes[1].set_ylabel("Density")
    axes[1].set_title("Drift rate")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGDIR / "model_illustration.pdf")
    fig.savefig(FIGDIR / "model_illustration.png")
    plt.close(fig)



def empirical_psychometric() -> None:
    df = pd.read_csv("results/empirical_roitman_psychometric.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9), sharey=True)
    for ax, monkey in zip(axes, sorted(df["monkey"].unique())):
        d = df[df["monkey"] == monkey].sort_values("coh")
        p = d["mean"].to_numpy()
        n = d["count"].to_numpy()
        se = np.sqrt(p * (1 - p) / n)
        ax.errorbar(
            d["coh"],
            p,
            yerr=1.96 * se,
            fmt="o",
            color="#1f4e79",
            ms=4,
            capsize=2,
            label="Observed",
        )
        ax.plot(
            d["coh"],
            d["pred_ours"],
            "-",
            color="#c1121f",
            lw=1.4,
            label="ddmtorch",
        )
        ax.plot(
            d["coh"],
            d["pred_pyddm"],
            "--",
            color="#6c757d",
            lw=1.2,
            label="PyDDM",
        )
        ax.set_xlabel("Motion coherence")
        ax.set_title(f"Monkey {int(monkey)}")
        ax.grid(True, alpha=0.25, lw=0.5)
    axes[0].set_ylabel("P(correct)")
    axes[0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGDIR / "empirical_psychometric.pdf")
    fig.savefig(FIGDIR / "empirical_psychometric.png")
    plt.close(fig)


def empirical_rt_quantiles() -> None:
    df = pd.read_csv("results/empirical_rt_quantiles.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2), sharex=True, sharey=True)
    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(df["coh"].min(), df["coh"].max())
    for ax, monkey in zip(axes, sorted(df["monkey"].unique())):
        d = df[df["monkey"] == monkey]
        for correct, marker, label in [(1, "o", "Correct"), (0, "x", "Error")]:
            sub = d[d["correct"] == correct]
            ax.scatter(
                sub["observed"],
                sub["predicted"],
                c=sub["coh"],
                cmap=cmap,
                norm=norm,
                marker=marker,
                s=16,
                alpha=0.8,
                label=label,
                linewidths=0.6,
            )
        lo = min(d["observed"].min(), d["predicted"].min()) - 0.05
        hi = max(d["observed"].max(), d["predicted"].max()) + 0.05
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.7)
        rmse = np.sqrt(np.mean((d["predicted"] - d["observed"]) ** 2))
        ax.set_title(f"Monkey {int(monkey)} (RMSE = {rmse:.3f} s)")
        ax.set_xlabel("Observed RT quantile (s)")
        ax.grid(True, alpha=0.25, lw=0.5)
        ax.set_aspect("equal", adjustable="box")
    axes[0].set_ylabel("Predicted RT quantile (s)")
    axes[0].legend(frameon=False, fontsize=7)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(sm, ax=axes, fraction=0.025, pad=0.02)
    cbar.set_label("Coherence")
    fig.savefig(FIGDIR / "empirical_rt_quantiles.pdf")
    fig.savefig(FIGDIR / "empirical_rt_quantiles.png")
    plt.close(fig)


def hierarchical_comparison() -> None:
    df = pd.read_csv("results/hierarchical_comparison_merged.csv")
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.8))
    colors = {"LL": "#1f4e79", "WL": "#e07a1f", "WW": "#2a9d8f"}
    for name, color in colors.items():
        x = df[f"hier_v_{name}"]
        y = df[f"v_{name}"]
        axes[0].scatter(x, y, s=18, alpha=0.8, color=color, label=name, edgecolors="none")
    r = np.corrcoef(df[["hier_v_LL", "hier_v_WL", "hier_v_WW"]].to_numpy().ravel(),
                    df[["v_LL", "v_WL", "v_WW"]].to_numpy().ravel())[0, 1]
    rmse = np.sqrt(np.mean((df[["v_LL", "v_WL", "v_WW"]].to_numpy()
                            - df[["hier_v_LL", "hier_v_WL", "hier_v_WW"]].to_numpy()) ** 2))
    axes[0].set_title(f"Drift $v$\n$r={r:.3f}$, RMSE={rmse:.3f}", fontsize=8)
    axes[0].set_xlabel("Hierarchical posterior mean")
    axes[0].set_ylabel("ddmtorch MLE")
    axes[0].legend(frameon=False, fontsize=7, title="Stimulus", title_fontsize=7)

    for ax, hcol, mcol, label in [
        (axes[1], "hier_a_all", "a", "Boundary $a$"),
        (axes[2], "hier_t0_all", "t0", "Non-decision $t_0$"),
    ]:
        x = df[hcol]
        y = df[mcol]
        ax.scatter(x, y, s=18, color="#2a6f97", edgecolors="none")
        r = np.corrcoef(x, y)[0, 1]
        rmse = np.sqrt(np.mean((y - x) ** 2))
        ax.set_title(f"{label}\n$r={r:.3f}$, RMSE={rmse:.3f}", fontsize=8)
        ax.set_xlabel("Hierarchical posterior mean")

    for ax in axes:
        lo = min(ax.get_xlim()[0], ax.get_ylim()[0])
        hi = max(ax.get_xlim()[1], ax.get_ylim()[1])
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.7)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.25, lw=0.5)
    fig.tight_layout()
    fig.savefig(FIGDIR / "hierarchical_comparison.pdf")
    fig.savefig(FIGDIR / "hierarchical_comparison.png")
    plt.close(fig)

def main() -> None:
    recovery_scatter()
    benchmark_scaling()
    benchmark_mle()
    model_illustration()
    empirical_psychometric()
    empirical_rt_quantiles()
    hierarchical_comparison()
    print(f"Figures written to {FIGDIR.resolve()}")


if __name__ == "__main__":
    main()
