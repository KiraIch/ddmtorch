"""CPU vs Apple Silicon MPS benchmarks for density and MLE.

Run with:
    conda run -n ddm python experiments/benchmark.py
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ddmtorch import fit_mle, sample_wfpt, wfpt_logpdf
from ddmtorch.device import resolve_device, resolve_dtype


def _sync(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()


def bench_density(
    n: int,
    device: torch.device,
    dtype: torch.dtype,
    *,
    warmup: int = 3,
    iters: int = 10,
    k_max: int = 48,
) -> dict:
    g = torch.Generator(device="cpu").manual_seed(1234)
    t = 0.3 + 2.5 * torch.rand(n, generator=g)
    v = -2.0 + 4.0 * torch.rand(n, generator=g)
    a = 0.7 + 1.3 * torch.rand(n, generator=g)
    w = 0.25 + 0.5 * torch.rand(n, generator=g)
    t0 = 0.2 * torch.rand(n, generator=g)
    t, v, a, w, t0 = [x.to(device=device, dtype=dtype) for x in (t, v, a, w, t0)]
    params = [v, a, w, t0]
    for p in params:
        p.requires_grad_(True)

    def step():
        for p in params:
            p.grad = None
        loss = -wfpt_logpdf(t, v, a, w, t0, k_max=k_max).mean()
        loss.backward()

    try:
        for _ in range(warmup):
            step()
        _sync(device)
        start = time.perf_counter()
        for _ in range(iters):
            step()
        _sync(device)
        elapsed = (time.perf_counter() - start) / iters
        return {
            "task": "density_fwd_bwd",
            "n_trials": n,
            "device": device.type,
            "dtype": str(dtype).replace("torch.", ""),
            "elapsed_s": elapsed,
            "trials_per_s": n / elapsed,
            "n_iter": iters,
            "notes": "ok",
        }
    except (RuntimeError, MemoryError) as exc:
        return {
            "task": "density_fwd_bwd",
            "n_trials": n,
            "device": device.type,
            "dtype": str(dtype).replace("torch.", ""),
            "elapsed_s": float("nan"),
            "trials_per_s": float("nan"),
            "n_iter": 0,
            "notes": f"failed: {type(exc).__name__}: {exc}"[:250],
        }


def bench_mle(
    n_trials: int,
    n_subjects: int,
    device: torch.device,
    dtype: torch.dtype,
    *,
    max_iter: int = 60,
    seed: int = 7,
) -> dict:
    rng = np.random.default_rng(seed)
    per = max(1, n_trials // n_subjects)
    v = torch.tensor(rng.uniform(-2, 2, n_subjects), dtype=torch.float64)
    a = torch.tensor(rng.uniform(0.8, 1.8, n_subjects), dtype=torch.float64)
    w = torch.tensor(rng.uniform(0.35, 0.65, n_subjects), dtype=torch.float64)
    t0 = torch.tensor(rng.uniform(0.15, 0.45, n_subjects), dtype=torch.float64)
    rt, choice = sample_wfpt(
        v,
        a,
        w,
        t0,
        n=per,
        device="cpu",
        dtype=torch.float64,
        n_grid=4096,
        generator=torch.Generator().manual_seed(seed),
    )
    subject = torch.arange(n_subjects).repeat_interleave(per)
    rt = rt.reshape(-1)
    choice = choice.reshape(-1)
    if device.type == "mps":
        rt = rt.float()
        choice = choice.float()
    try:
        res = fit_mle(
            rt,
            choice,
            subject,
            device=device,
            dtype=dtype,
            max_iter=max_iter,
            seed=seed,
        )
        return {
            "task": "mle",
            "n_trials": int(rt.numel()),
            "device": device.type,
            "dtype": str(dtype).replace("torch.", ""),
            "elapsed_s": res.elapsed_s,
            "trials_per_s": rt.numel() / res.elapsed_s,
            "n_iter": res.n_iter,
            "notes": f"n_subjects={n_subjects}, success={res.success}",
        }
    except (RuntimeError, MemoryError) as exc:
        return {
            "task": "mle",
            "n_trials": int(rt.numel()),
            "device": device.type,
            "dtype": str(dtype).replace("torch.", ""),
            "elapsed_s": float("nan"),
            "trials_per_s": float("nan"),
            "n_iter": 0,
            "notes": f"failed: {type(exc).__name__}: {exc}"[:250],
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results/benchmark.csv")
    parser.add_argument("--max-n", type=int, default=1_000_000)
    parser.add_argument("--iters", type=int, default=10)
    args = parser.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    cpu = resolve_device("cpu")
    devices = [(cpu, torch.float64), (cpu, torch.float32)]
    if torch.backends.mps.is_available():
        devices.append((resolve_device("mps"), torch.float32))

    n_values = [1_000, 10_000, 100_000, args.max_n]
    n_values = sorted(set(n_values))
    rows = []
    for dev, dtype in devices:
        for n in n_values:
            print(f"density n={n:,} device={dev.type} dtype={dtype}", flush=True)
            rows.append(bench_density(n, dev, dtype, iters=args.iters))
            print(f"  {rows[-1]['elapsed_s']:.3f}s", flush=True)

    for dev, dtype in devices:
        for n_trials, n_subjects in [(2_000, 1), (10_000, 1), (50_000, 20)]:
            print(f"mle n={n_trials:,} subjects={n_subjects} device={dev.type} dtype={dtype}", flush=True)
            rows.append(
                bench_mle(n_trials, n_subjects, dev, dtype, max_iter=60, seed=11)
            )
            print(f"  {rows[-1]['elapsed_s']:.3f}s", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    print(df.to_string(index=False))
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
