# ddmtorch

**Repository:** <https://github.com/KiraIch/ddmtorch>  
**Technical report:** [paper/main.pdf](paper/main.pdf)  
**Archived code DOI:** [10.5281/zenodo.23721071](https://doi.org/10.5281/zenodo.23721071)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23721071.svg)](https://doi.org/10.5281/zenodo.23721071)

A differentiable, PyTorch implementation of the drift-diffusion model (DDM) with first-class support for Apple Silicon MPS.

`ddmtorch` vectorizes the Wiener first-passage time (WFPT) likelihood across trials and subjects, runs on CPU or MPS, and provides batched maximum-likelihood estimation, synthetic data generation, and a reproducible recovery/benchmark suite.

> **Scope.** This is an engineering toolbox for the standard four-parameter DDM with fixed boundaries, constant drift, and no across-trial variability. It is not a replacement for HDDM, fast-dm, PyDDM, or fddm, and it does not yet implement hierarchical priors, variational inference, or the full Ratcliff model. See [Limitations](#limitations).

## Highlights

- **Vectorized WFPT likelihood** using the hybrid large-time/small-time expansions of Navarro & Fuss (2009), following the formulation used by `fddm`.
- **CPU float64 and Apple Silicon MPS float32** backends; the same code is device-agnostic.
- **Autograd-friendly**: the log likelihood is differentiable with respect to `v, a, w, t0`.
- **Batched multi-subject MLE** with LBFGS and numerically stable parameter transformations.
- **Synthetic recovery**: grid-based inverse-CDF sampling plus an independent Euler–Maruyama simulator.
- **Empirical applications** to the public Roitman & Shadlen (2002) motion-discrimination dataset (PyDDM comparison) and the 14-subject Cavanagh et al. (2011) dataset (HDDM-style hierarchical comparison).
- **Reproducible benchmarks** comparing CPU float64, CPU float32, and MPS float32.

## Install

This project is developed with Miniforge. Create the environment from the repository root:

```bash
conda env create -f environment.yml
conda activate ddm
python -m pip install -e ".[dev]"
```

Or, if you already have PyTorch installed:

```bash
python -m pip install -e .
```

Requirements: Python ≥ 3.10 and PyTorch ≥ 2.2. MPS requires macOS with an Apple Silicon GPU and a PyTorch build with MPS enabled.

## Quick start

```python
import torch
from ddmtorch import fit_mle, sample_wfpt, wfpt_logpdf, choice_probability

# Evaluate the WFPT log density for the upper boundary.
rt = torch.tensor([0.6, 0.9, 1.4], dtype=torch.float64)
logp = wfpt_logpdf(rt, v=1.0, a=1.2, w=0.5, t0=0.3, boundary="upper")
print(logp)

# Choice probability.
print(choice_probability(1.0, 1.2, 0.5))  # tensor(0.70...)

# Generate synthetic data.
rt, choice = sample_wfpt(
    v=1.0, a=1.2, w=0.5, t0=0.3,
    n=2000, device="cpu", dtype=torch.float64,
)

# Fit on MPS (falls back automatically if MPS is unavailable).
fit = fit_mle(rt, choice, device="auto")
print(fit.params)   # columns: v, a, w, t0
print(fit.nll, fit.elapsed_s, fit.success)
```

For multiple subjects:

```python
subject = torch.arange(3).repeat_interleave(1000)
fit = fit_mle(rt, choice, subject=subject, device="mps", max_iter=100)
print(fit.params.shape)  # (3, 4)
```

## API

| Function | Description |
|---|---|
| `wfpt_logpdf(rt, v, a, w, t0, boundary="upper")` | Log WFPT density; broadcasts over trials, subjects, and parameters. |
| `wfpt_pdf(...)` | WFPT density (not log). |
| `choice_probability(v, a, w, upper=True)` | Analytic probability of hitting the upper/lower boundary. |
| `sample_wfpt(v, a, w, t0, n, ...)` | Sample `(rt, choice)` by inverting the WFPT CDF on a grid. |
| `simulate_wfpt_euler(v, a, w, t0, n, dt=1e-3, ...)` | Independent Euler–Maruyama simulator (approximate). |
| `loglik(rt, choice, v, a, w, t0)` | Elementwise log likelihood for observed choices. |
| `fit_mle(rt, choice, subject=None, ...)` | Pooled independent-subject MLE; returns `FitResult`. |

### Conventions

- `a` is the boundary separation; the lower boundary is at 0 and the upper boundary at `a`.
- `w` is the relative starting point measured from the lower boundary (`z = a*w`); `w = 0.5` is unbiased.
- `v` is the drift rate; the diffusion coefficient is fixed at 1.
- `t0` is non-decision time; the density is evaluated at `rt - t0`.
- `choice = 1` means the upper boundary, `choice = 0` the lower boundary.

## Results

All results below were generated on a MacBook Air with Apple M5, 24 GB unified memory, macOS 26.5, Python 3.12.16, and PyTorch 2.14.1.

### Density accuracy

Compared with a CPU float64 reference (`k_max=512`), the production float32 implementation (`k_max=48`) had a maximum absolute log-density error of `3.51e-5` on CPU and `3.13e-5` on MPS, for 19,918 of 20,000 random parameter sets with `tau >= 0.02` and representable float32 densities. CPU float32 and MPS float32 agreed to `7.6e-6`. The series sum is clamped at the smallest positive float32 value before taking the log, so extreme right-tail densities below the float32 range are not resolved; use CPU float64 for those cases.

### Synthetic recovery

100 datasets × 500 trials, grid sampler, MPS fit:

| Parameter | RMSE | Correlation |
|---|---:|---:|
| Drift rate `v` | 0.097 | 0.996 |
| Boundary `a` | 0.034 | 0.995 |
| Start point `w` | 0.018 | 0.981 |
| Non-decision `t0` | 0.007 | 0.998 |

An independent Euler–Maruyama check (100 datasets, `dt=1e-3`) showed similar recovery, with a small positive bias in `a` (+0.033) and essentially no bias in `v` (+0.0004), `w` (+0.001), and `t0` (+0.002). The boundary bias is consistent with boundary-crossing discretization.

### Empirical application

We also fit the public Roitman & Shadlen (2002) motion-discrimination dataset (two monkeys, six coherence levels) with drift proportional to coherence, a constant bound, a free starting point, non-decision time, and a fixed 2% uniform contaminant. The same model was fit with PyDDM 0.9.0 (Fokker–Planck solver).

| Monkey | Method | Drift slope | Boundary | Start | `t0` | NLL |
|---|---:|---:|---:|---:|---:|---:|
| 1 | ddmtorch | 10.494 | 1.489 | 0.488 | 0.307 | 204.30 |
| 1 | PyDDM | 10.536 | 1.486 | 0.483 | 0.309 | 206.66 |
| 2 | ddmtorch | 9.495 | 1.749 | 0.501 | 0.193 | 1254.22 |
| 2 | PyDDM | 9.602 | 1.754 | 0.494 | 0.194 | 1254.61 |

The two implementations recover nearly identical parameters and similar likelihoods. The simple constant-bound model under-predicts accuracy at intermediate coherences and over-disperses RT quantiles; that is a property of the model, not of either implementation. See `paper/figures/empirical_psychometric.pdf` and `paper/figures/empirical_rt_quantiles.pdf`.

### Hierarchical comparison

On the 14-subject Cavanagh et al. (2011) reward-based decision dataset, we compared per-subject `ddmtorch` MLE with a PyMC 5 HDDM-style hierarchical fit (subject-specific drift by condition, boundary, non-decision time; fixed `w=0.5`; 5% uniform contaminant). As of October 2026, HDDM 1.0.1 depends on PyMC 2, so this is a reimplementation of the model class, not the HDDM package. Correlations: `v_LL=0.997`, `v_WL=0.9995`, `v_WW=0.997`, `a=0.999`, `t0=0.996`; the hierarchical estimates show the expected shrinkage. See `paper/figures/hierarchical_comparison.pdf`.

### CPU vs MPS

| Workload | CPU float64 | CPU float32 | MPS float32 |
|---|---:|---:|---:|
| Density fwd+bwd, 1M trials | 3.21 s | 1.22 s | 0.62 s |
| MLE, 10k trials, 1 subject | 0.66 s | 0.46 s | 0.28 s |
| MLE, 50k trials, 20 subjects | 7.30 s | 4.10 s | 2.32 s |
| MLE, 2k trials, 1 subject | 1.04 s | 0.12 s | 0.26 s |

MPS wins for large batches (roughly ≥10k trials), but CPU float32 is faster for small batches because kernel-launch and synchronization overhead dominate. See `results/benchmark.csv` and `paper/figures/` for the full data.

## Reproduce

```bash
conda activate ddm

# Tests
python -m pytest -q

# Synthetic recovery (100 datasets x 500 trials)
python experiments/recovery.py --n-datasets 100 --n-trials 500

# Independent Euler-Maruyama check
python experiments/euler_recovery.py --n-datasets 100 --n-trials 500

# Density validation
python experiments/validate_density.py

# CPU/MPS benchmarks
python experiments/benchmark.py --max-n 1000000 --iters 5

# Empirical application (downloads the public dataset on first run)
python experiments/empirical_roitman.py --seed 0

# Bootstrap confidence intervals (500 resamples per monkey)
python experiments/bootstrap_empirical.py --n-boot 500

# Observed vs predicted RT quantiles
python experiments/empirical_rt_quantiles.py

# Hierarchical comparison (requires the separate `pymc` environment)
conda run -n pymc python experiments/hierarchical_pymc.py --subjects all --draws 1000 --tune 1000 --chains 4 --seed 2026
python experiments/hierarchical_compare.py --n-restarts 2 --seed 0

# Illustrative contaminant check
python experiments/contaminant_check.py

# Figures
python experiments/make_figures.py

# Technical report
cd paper && tectonic -X compile main.tex
```

Or use the Makefile:

```bash
make test
make experiments
make paper
```

### Fixed seeds

The scripts expose their seeds as command-line options and use these defaults: density validation `123`; recovery `2026` (dataset `i` uses `2026+i`); Euler check `77` (dataset `i` uses `77+i`); Roitman bootstrap `2026` (resample `b` uses `2026+b`); Roitman `ddmtorch`/PyDDM restart seed `0`; hierarchical PyMC `2026`; Cavanagh per-subject MLE restart seed `0`; density benchmark `1234`; MLE benchmark `11`; contaminant check `0`/`1`/`2`. The generated `results/` files record these values where applicable.

## Repository layout

```
src/ddmtorch/          # package
  density.py           # vectorized WFPT density, choice probability
  simulate.py          # inverse-CDF and Euler-Maruyama samplers
  fit.py               # pooled multi-subject MLE
  device.py            # CPU/MPS device and dtype helpers
tests/                 # correctness, gradients, recovery, MPS agreement
experiments/           # recovery, Euler check, density validation, benchmarks, empirical application, figures
data/                  # downloaded empirical data (not versioned)
environment-pymc.yml   # separate PyMC 5 environment for the hierarchical comparison
paper/                 # LaTeX technical report, references, figures
results/               # generated CSV/JSON/NPZ outputs
```

## Limitations

- **Model scope**: standard four-parameter DDM; a fixed-proportion uniform contaminant is available but its probability is not estimated. No across-trial variability (`sv`, `sz`, `st0`), full contaminant distribution, collapsing bounds, or time-varying drift.
- **Inference scope**: the core API provides MLE point estimates only; it does not compute standard errors, hierarchical shrinkage, posterior uncertainty, or model comparison. Bootstrap confidence intervals and a separate PyMC 5 hierarchical comparison are provided as analysis scripts.
- **Numerical scope**: MPS uses float32; extreme right-tail densities can underflow. Use CPU float64 for publication-grade numerical work or responses very close to `t0`.
- **Evidence scope**: synthetic recovery plus two public empirical datasets; PyDDM comparison and an HDDM-style PyMC 5 hierarchical comparison. The HDDM package itself was not run because its current release depends on PyMC 2; no fast-dm comparison.
- **Hardware scope**: benchmarked on one Apple M5 machine; no CUDA benchmark.

## Roadmap

- Across-trial variability and contaminant mixtures.
- Bootstrap/Hessian standard errors.
- Mean-field and hierarchical variational inference.
- Amortized Bayesian inference with simulation-based calibration.
- Trial-level regression through `v`, `a`, or `t0`.
- CUDA and `torch.compile` benchmarks.
- Direct comparison with fddm, fast-dm, HDDM, and PyDDM on shared data.

## Data

The empirical examples use the public Roitman & Shadlen (2002) dataset (processed by the PyDDM project) and the Cavanagh et al. (2011) dataset distributed with the HDDM examples. The scripts download the processed CSVs on first run; the data are not redistributed in this repository. Please cite the original papers if you use the datasets.

## Documentation

- [Technical report source](paper/main.tex)
- [Experiment scripts](experiments/)
- [Generated results](results/README.md)

Personal literature and knowledge notes are kept locally; they are not part of this public repository.

## Citation

If you use this code, please cite the archived software release and the underlying WFPT work:

```bibtex
@software{Lin2026ddmtorch,
  author  = {Lin, ChiaAn},
  title   = {ddmtorch: A Differentiable, MPS-Accelerated Toolbox for Drift-Diffusion Model Estimation on Apple Silicon},
  year    = {2026},
  version = {0.1.0},
  doi     = {10.5281/zenodo.23721071},
  url     = {https://github.com/KiraIch/ddmtorch}
}

@article{NavarroFuss2009,
  author  = {Navarro, Daniel J. and Fuss, Ian G.},
  title   = {Fast and accurate calculations for first-passage times in Wiener diffusion models},
  journal = {Journal of Mathematical Psychology},
  year    = {2009},
  volume  = {53},
  number  = {4},
  pages   = {222--230},
  doi     = {10.1016/j.jmp.2009.02.003}
}
```

## License

The code is released under the MIT license (see `LICENSE`). The manuscript and figures are released under CC BY 4.0 (see `paper/LICENSE`).
