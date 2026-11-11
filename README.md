# Code and data for "Mean span dimension of an ideal wormlike chain"

Yanwei Wang, Nazarbayev University. Prepared for submission to *The Journal of Chemical Physics* (2026).

This repository contains the numerical programs, the generated data, the figures, a
chapter-by-chapter tutorial that derives and verifies the article's results, and one explanatory
notebook per main-text figure, for the theory of the mean span dimension of the ideal three-dimensional wormlike
chain (WLC), without excluded volume.

**Start with `tutorial/00_overview.ipynb`.**

The article and its supplementary material are not part of this repository; they will be available from
the journal. The notebooks and data files refer to the article's results by name (for example "the long-chain
expansion"), not by equation or section number; only the figure numbers (Figs. 1–4 and Fig. S1) are used, because
the figure files and notebooks are named after them. `tutorial/README.md` lists where each result is derived, with its equation number in the submitted article.

## Contents

| Path | What it is |
|---|---|
| `generated/`, `figures/` | LaTeX tables generated from the data (the two reference tables of the article and the supplement among them) and the figure files (Figs. 1–4 and Fig. S1, PDF and PNG). |
| `code/` | Numerical programs: the spectral first-return solver, the finite-volume solver, the chain-path samplers, symbolic audits, table generators and the figure module. |
| `scripts/` | One short script per figure (`make_fig1.py` … `make_fig4.py`), `build_notebooks.py` and `build_tutorial.py`. |
| `tutorial/` | Ten executed notebooks (an overview and nine chapters) that walk through and verify the article topic by topic: every derivation step by step with a symbolic or numerical check, live recomputations, an index from each result to its notebook section, and a final chapter that recomputes the article's quantitative statements (117 checks; values to their printed rounding, bounds one-sidedly). See `tutorial/README.md`. |
| `notebooks/` | One executed Jupyter notebook per main-text figure (`fig1.ipynb` … `fig4.ipynb`) explaining what is plotted, which data file it comes from, and how the quantities were obtained. |
| `data/` | All computed data (CSV, JSON, retained sample records). See `data/README.md`. |
| `validation/` | Run logs, audit outputs, and the earlier versions of the reference table with their comparison (`previous_tables/`, `reference_table_versions.csv`, `inversion_noise_audit.csv`). |

## The figures

Each figure is produced by `code/plot_main_figures.py` from archived data only; no value is refitted
at plotting time. `scripts/make_figN.py` regenerates figure *N*; `notebooks/figN.ipynb` walks through
the same data and reproduces the numbers quoted in the text before rendering the figure.

| Figure | Content | Data |
|---|---|---|
| 1 | Concept: an illustrative conformation and its span; the record segments along the contour; the same trace read backward from a growing record; the angle-averaged Milne solution and its extrapolation length | seeded simulation (illustrative only); stationary Milne solution at m = 128 (`build_boundary_moments.py`) |
| 2 | Compensated mean span and the span-to-gyration ratio over 10⁻² ≤ L/p ≤ 10⁵ with the short- and long-chain expansions | `wlc_mean_span_reference.csv` |
| 3 | Asymptotic coefficients 37/768, 9/256 and 2z_M, and the wall offset z_M of the depletion layer | `stiff_alias_summary.csv`, `endpoint_excursion_diagnostic.csv`, reference CSV |
| 4 | The practical formula and its relative error at every node, with calibration, interleaved withheld and extrapolation nodes distinguished | reference CSV, `rational_fit_coefficients.json`, `reference_nodes_with_split.csv` |
| S1 (supplement) | Tests independent of the spectral solver: heat-kernel paths (L/p = 0.1–10⁴) and finite-volume solutions against the spectral solution; paired step-size differences of the paths (`figures/figSM1_independent_checks.pdf`, `pmf.figure_checks()`) | `heat_path_coupled.csv`, `heat_path_long.csv`, `heat_path_long_pooled.csv`, `finite_volume_comparison.csv`, `finite_volume_order_analysis.csv`, `finite_volume_long_summary.json` |

## Reproduction

Python 3.12 or later with the package versions in `requirements.txt` (the notebooks additionally need `jupyter`, `nbformat`, `nbconvert`, `pandas` and `sympy`):

```bash
python -m pip install -r requirements.txt
```

Analysis, tables, figures and notebooks from the archived data (minutes):

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python code/evaluate_formula_metrics.py # error of the practical formula by node set (no refit)
python code/refresh_spectral_columns.py  # spectral values stored in the finite-volume comparison files
python code/finite_volume_order_analysis.py
python code/pool_long_paths.py          # pooled path estimate at L/p = 1e4 from the retained samples
python code/compare_table_versions.py
python code/telegraph_inversion_test.py # exact test of the production inversion at all 67 lengths (seconds)
python code/long_chain_remainder_check.py # table vs the long-chain expansion + remainder; inversion-parameter sensitivity (about 30 s)
python code/generate_latex_tables.py
python code/apply_supplement_alt_text.py # alt-text comments in the generated tables
python code/plot_span_to_rg.py
python code/plot_main_figures.py        # Figs. 1-4 and supplementary Fig. S1
python scripts/build_notebooks.py       # executes notebooks/fig*.ipynb
python scripts/build_tutorial.py        # executes tutorial/*.ipynb (one to two minutes); chapter 9 runs the article's checks
python code/validate_span_to_rg.py
```

The reference curve itself (`data/wlc_mean_span_reference.csv`, 67 nodes):

```bash
python code/build_reference_curve.py --fresh --workers 3     # 67 nodes, about 6 s each (ordered Schur, alias-corrected A=12)
python code/sphere_heat_path_validation.py --long --workers 2 # path sampling at L/p = 1e3, 1e4 (about 30 min on two cores)
python code/sphere_heat_path_validation.py --long-extra --workers 2 # three more batches at L/p = 1e4 (about 30 min)
python code/build_finite_volume_long.py                      # finite-volume check at L/p = 100 (about 1.5 h on two cores)
python code/conditioning_audit.py                            # cond(E+ Z-) at every production transform point, orders 8-128
python code/solver_regression_audit.py                       # eigendecomposition route as regression check
python code/high_order_spot_checks.py                        # orders 128, 192, 256 and their own extrapolation
```

The files `generated/*.tex` that the article and its supplement do not read are written by the same
table generator (`code/generate_latex_tables.py`) and are retained as tabulated diagnostics.

The remaining campaigns (short-chain refinement to angular order 512, finite-volume solutions,
heat-kernel and polygonal path sampling, Milne boundary moments) take hours and
are listed with their options in the docstrings of the corresponding files in `code/`. Final digits can
vary with libraries and hardware. Sampling standard errors exclude path discretization and
heat-kernel inversion errors; the spectral sensitivities are numerical diagnostics, not certified
error bounds.

## Licences

* Code, including the code cells of the notebooks: MIT licence (`LICENSE`).
* Data generated in this work: the CSV and JSON data files, the figures and the explanatory text of the
  notebooks are released under CC BY 4.0 (`DATA_LICENSE.md`).
* Third-party material: the published constants used as inputs (the Milne extrapolation length of
  Amic, Luck and Nieuwenhuizen; the random-acceleration return kernel of McKean and Burkhardt) remain
  the results of those articles (see `DATA_LICENSE.md`).

## Citation

If you use this code or data, please cite the paper above and this repository (`CITATION.cff`):

Y. Wang, Code and data for "Mean span dimension of an ideal wormlike chain", GitHub repository, https://github.com/ktwyw/wlc-mean-span (2026).

## Contact

Yanwei Wang, Nazarbayev University: [yanwei.wang@nu.edu.kz](mailto:yanwei.wang@nu.edu.kz),
ORCID [0000-0002-8488-9833](https://orcid.org/0000-0002-8488-9833).
