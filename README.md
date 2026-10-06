# ESR_weak: dark-matter halo profiles from weak lensing with Exhaustive Symbolic Regression

Code for constraining the **density profiles of dark-matter haloes directly from weak
lensing**, without assuming a functional form such as NFW.

The code builds on [Exhaustive Symbolic Regression (ESR)](https://github.com/DeaglanBartlett/ESR).
ESR enumerates every analytic expression up to a given complexity, fits each one as a
3D density profile `ρ(r)` to the measured **excess surface density** `ΔΣ(R)` of galaxy
clusters, and ranks the expressions by **minimum description length**. That ranking
trades accuracy against simplicity, so it identifies the profiles the data prefer.
The analysis compares them with NFW.

This repository extends ESR with:
- a 3D-density to ESD forward model with checks for poles and quadrature error
  (`esr/esd.py`, `analysis/esd_fixed.py`);
- weak-lensing likelihoods for the HSC and CLASH cluster samples, the CLASH one with a
  full covariance (`esr/fitting/WL_likelihood.py`, `esr/fitting/WL_likelihood_CLASH.py`);
- multi-cluster ranking that combines description lengths across clusters, plus global
  versus mixture-of-profiles assignment (`analysis/combine_DL_galaxies.py`,
  `analysis/mixture_global.py`);
- M200 posteriors and mass comparisons with NFW (`analysis/calculate_M200*.py`);
- numerical-stability and quadrature audits (`analysis/check_*.py`, `*_audit*.py`,
  `quad_verify.py`).

## Papers

- A. Martín, T. Yasin, D. J. Bartlett, H. Desmond, P. G. Ferreira,
  **Symbolically regressing dark matter halo profiles using weak lensing**,
  MNRAS (2026), [doi:10.1093/mnras/stag1394](https://doi.org/10.1093/mnras/stag1394),
  [arXiv:2601.05203](https://arxiv.org/abs/2601.05203). This is the HSC analysis.
- A. Martín, T. Yasin, D. J. Bartlett, H. Desmond, P. G. Ferreira,
  **Constraining dark matter halo profiles with symbolic regression**,
  Phil. Trans. R. Soc. A 384, 20250090 (2026),
  [doi:10.1098/rsta.2025.0090](https://doi.org/10.1098/rsta.2025.0090),
  [arXiv:2511.23073](https://arxiv.org/abs/2511.23073).
- The CLASH analysis is **in preparation**.

This version of the code contains the pipeline as developed for the CLASH analysis.
It extends the code used for the published HSC paper and is very similar to it. The
earlier state of the repository (April 2025) is kept under the git tag
[`hsc-2025-04`](https://github.com/Alicia-Martin/ESR_weak/tree/hsc-2025-04).

## Installation

```bash
git clone https://github.com/Alicia-Martin/ESR_weak.git
cd ESR_weak
pip install -e .
pip install -r requirements.txt
```

You also need an ESR **function library**: the pre-computed candidate functions at each
complexity. It's available from [Zenodo](https://doi.org/10.5281/zenodo.7339113); see
[`README_ESR.rst`](README_ESR.rst). Put it in `esr/function_library/`.

## Usage

The ESR stages run for each cluster and complexity:

```python
from esr.fitting.WL_likelihood_CLASH import WLLikelihood
import esr.fitting.test_all, esr.fitting.test_all_Fisher, esr.fitting.match

like = WLLikelihood(data_file, cov_file, run_name, fn_set="core_maths")
esr.fitting.test_all.main(comp, like)          # fit every function
esr.fitting.test_all_Fisher.main(comp, like)   # Fisher matrix + description length
esr.fitting.match.main(comp, like)             # extend to all equivalent forms
```

`esr/esr_WL.py` is an example driver, and it runs under MPI
(`mpirun -n N python esr/esr_WL.py`). Results go to `esr/fitting/output/`. Then:

- `analysis/combine_DL_galaxies.py` ranks functions across clusters;
- `analysis/combine_final_results.py` and `analysis/mixture_global.py` build the
  final tables (see `analysis/mixture_global_README.md`);
- `analysis/calculate_M200*.py` compute masses.

The scripts in `analysis/` and `notebooks/` are research scripts. Their input and output
paths are set near the top of each file and point at `data/` and `results/` folders.
Edit them to match where your data and fit outputs are.

## Data

The cluster lensing data are **not included**, in line with the papers' data-availability
statements; they are available on reasonable request. For CLASH, each cluster's input is
a text file with three rows (radii `R`, `ΔΣ`, and its uncertainty), plus a covariance
matrix file. `analysis/cluster_names.txt` and `cluster_redshifts.txt` list the CLASH
clusters used.

## Credits

ESR is by Deaglan J. Bartlett and Harry Desmond; please also cite
[Bartlett, Desmond & Ferreira (2023)](https://arxiv.org/abs/2211.11461) and the other ESR
papers listed in [`README_ESR.rst`](README_ESR.rst). The prior on functions uses
[katz](https://github.com/DeaglanBartlett/katz).

## Citation

Please cite the papers above. GitHub's "Cite this repository" button uses
[`CITATION.cff`](CITATION.cff).

## License

MIT. See [LICENSE](LICENSE).
