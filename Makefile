PY := conda run -n ddm python

.PHONY: test recovery euler validate benchmark empirical bootstrap quantiles contaminant hierarchical hier_compare figures paper experiments all

test:
	$(PY) -m pytest -q

recovery:
	$(PY) experiments/recovery.py --n-datasets 100 --n-trials 500

euler:
	PYTHONPATH=. $(PY) experiments/euler_recovery.py --n-datasets 100 --n-trials 500

validate:
	$(PY) experiments/validate_density.py

benchmark:
	$(PY) experiments/benchmark.py --max-n 1000000 --iters 5

empirical:
	PYTHONPATH=. $(PY) experiments/empirical_roitman.py

bootstrap:
	PYTHONPATH=. $(PY) experiments/bootstrap_empirical.py --n-boot 500

quantiles:
	PYTHONPATH=. $(PY) experiments/empirical_rt_quantiles.py

contaminant:
	$(PY) experiments/contaminant_check.py

hierarchical:
	conda run -n pymc python experiments/hierarchical_pymc.py --subjects all --draws 1000 --tune 1000 --chains 4 --seed 2026

hier_compare:
	$(PY) experiments/hierarchical_compare.py --n-restarts 2 --seed 0

figures:
	$(PY) experiments/make_figures.py

experiments: recovery euler validate benchmark empirical bootstrap quantiles contaminant figures

paper:
	cd paper && conda run -n ddm tectonic -X compile main.tex

all: test experiments paper
