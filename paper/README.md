# Technical report

Build with Tectonic:

```bash
conda activate ddm
cd paper
tectonic -X compile main.tex
```

This produces `main.pdf` and `main.log`; BibTeX is run automatically.  Line numbers are enabled with `\linenumbers` for review; remove that command for the final camera-ready/preprint version if you prefer unnumbered lines.

Author: ChiaAn Lin, Master's Program in Precision Health and Intelligent Medicine, Graduate School of Advanced Technology, National Taiwan University, Taipei, Taiwan. Correspondence: ntuhibiki@gmail.com.

Before submitting to PsyArXiv:

1. Confirm the official program name and affiliation formatting in the title-page footnote.
2. Archived code DOI is set: https://doi.org/10.5281/zenodo.23721071.
3. License selected: CC BY 4.0 for the manuscript; MIT for the code.
4. Re-run the experiments that produce the numbers in Tables 1–2, or verify that the committed `results/` match the paper.
5. Check the reference list against the PDFs in `docs/literature/pdfs/` (especially the Fengler et al. published version, DOI `10.1162/jocn_a_01902`).
6. Re-run `experiments/empirical_roitman.py`, `experiments/bootstrap_empirical.py`, and `experiments/empirical_rt_quantiles.py` if the empirical numbers change; the first script downloads the public Roitman & Shadlen (2002) data on first run.
7. The hierarchical comparison uses the separate `pymc` environment (`environment-pymc.yml`) and `experiments/hierarchical_pymc.py`; HDDM itself is not used because its current release depends on PyMC 2.
8. Re-run `experiments/make_figures.py` to regenerate all figures.

The report intentionally describes the standard four-parameter DDM only. Do not add claims about hierarchical models, inter-trial variability, or CUDA performance unless those experiments are actually added.
