# Trust-Region vs Line Search for Unconstrained Optimization — code and data

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This repository contains the code and the data that accompany the manuscript

> J. Slimani and K. Ibn Batouta, *Trust-Region vs Line Search for Unconstrained Optimization: A Fair-Cost Study on
> Ill-Conditioned, Non-Convex, and Indefinite-Hessian Problems*, manuscript under review (the DOI will be added after
> publication).

The tag `v0.5.2` contains the code that produced all the results reported in the manuscript, unchanged, together with the
result files and the scripts that reproduce the tables and figures.

---

## Repository structure

```
trust-region-vs-line-search/
├── framework_united_final.py                       # benchmark engine: oracle counting, unified budget, run_one()
├── trust_region.py                                 # Cauchy Point, Dogleg, Steihaug-CG
├── implementation_methode_lineaires.py             # BFGS-Armijo, BFGS-Wolfe, Newton-LM
├── benchmark_all_categories_no_levy_validation.py  # the 36 test problems (categories A, B, C)
├── run_multistart.py                               # multiple-start protocol
├── run_G3.py, plot_G3.py                           # controlled conditioning experiment on rotated quadratics
├── analysis/data_profiles.py                       # data profiles (Moré and Wild)
├── tests/                                          # unit tests
├── results/                                        # result files used by the manuscript (see "Data files")
│   ├── timing/                                     # wall-clock measurements (passes A and B)
│   └── reference/                                  # outputs of earlier versions, used only for comparisons
└── scripts/
    ├── run/                                        # execution of the benchmark, multiple starts and conditioning experiment
    ├── timing/                                     # wall-clock measurement harness
    └── analysis/                                   # statistics, tables and figures; sensitivity/ for the sensitivity analysis
```

In the code and in the result files, Newton-LM is named `Newton_BT`, and problem identifiers carry the name of the
function (for example `B6_Broydn3D` for problem B6 of the manuscript).

---

## Solvers and benchmark

Six solvers are compared: three trust-region solvers (Cauchy Point, Dogleg, Steihaug-CG) and three line-search methods
(BFGS-Armijo, BFGS-Wolfe, and Newton with Levenberg–Marquardt regularization, Newton-LM).

The benchmark has 36 problems in three categories: (A) ill-conditioned problems, 15 problems (7 quadratics with a
prescribed spectrum and 8 smooth test functions); (B) non-convex functions, 11 problems; (C) problems with an indefinite
Hessian at the starting point, 10 problems. Scalable problems are run at n ∈ {10, 50, 100, 500, 1000}; fixed-dimension
problems are run at their natural size (n = 2, 3 or 4), and the two problems that require n to be a multiple of 4 are
run at n = 8 and n = 48 instead of 10 and 50. This gives 156 distinct instances (180 runs per solver, some of which
repeat a fixed-dimension instance). A run is a success if ‖∇f(x_k)‖ < 10⁻⁶ is reached within K_max = 1000 iterations.
Appendix A of the manuscript lists the 36 problems with their dimensions, starting points, derivative oracles and
sources; Appendix B specifies the six solvers.

## Main results (tested domain)

The results hold in the tested domain: dense explicit Hessians for all solvers, n ≤ 1000, and wall-clock times measured
on Category A on a single machine.

- Newton-LM has the highest success rate in every category and overall (97.4 % of the 156 distinct instances); its
  advantage over Dogleg is no longer significant when the four problems with finite-difference derivatives are
  excluded.
- Up to n = 1000, Newton-LM is not slower than Steihaug-CG in wall-clock time (ratio of wall-clock times 0.65
  [0.48, 1.02] at n = 1000).
- The Pareto front of robustness against the slope of the wall-clock time per iteration consists of two solvers,
  Steihaug-CG and Newton-LM.
- A decision rule with a single node (Newton-LM if an explicit Hessian is available, BFGS-Wolfe otherwise) loses 0.006
  [0.000, 0.017] of success in leave-one-problem-out validation, against 0.050 [0.022, 0.078] for a threshold rule on
  the dimension.

See the manuscript for the statistical analysis, the intervals and the limitations.

---

## Installation

```bash
git clone https://github.com/jslimani1989/trust-region-vs-line-search.git
cd trust-region-vs-line-search
git checkout v0.5.2
pip install -r requirements.txt
```

The results were produced with Python 3.13.13, NumPy 2.4.6 and SciPy 1.18.0 (see `requirements.txt`). The figure scripts
also use Matplotlib and pandas. The scripts in `scripts/run/` and `scripts/timing/` call `git` to check the code (see
below), so the repository must be a git clone. The timing harness (`scripts/timing/`) uses Windows process affinity and
priority.

---

## Reproducing the results

All commands are run from the root of the repository. Re-executions write to `rerun/`; analysis outputs go to the
directory given by the environment variable `ANALYSIS_OUT` (default: `analysis_output/`). The distributed files in
`results/` are not modified, except by `p0_fichiers_complets.py` (step 2 below), which rebuilds the two complete files
of `results/` from a re-execution.

Before running, each script of `scripts/run/` and the timing harness check that the four code files
(`framework_united_final.py`, `trust_region.py`, `implementation_methode_lineaires.py`,
`benchmark_all_categories_no_levy_validation.py`) have the same git hashes as in the original run, recorded in
`results/resultats_benchmark_v0.5_meta.json`.

### 1. Benchmark

The results of Cauchy Point and Steihaug-CG come from an earlier run of the same code for these two solvers; the four
other solvers were run again with the current code, and the two parts were merged:

```bash
python scripts/run/run_benchmark_v0_5.py            # the other four solvers on the 180 slots (option --workers)
python scripts/run/fusion_v0_5.py                   # rerun/resultats_benchmark_v0.5.csv, to compare with results/
```

The whole benchmark (six solvers) can also be run directly with the benchmark engine:

```bash
python framework_united_final.py --csv my_results.csv            # options: --medium, --category A, --method Steihaug_CG
```

### 2. Multiple starting points and conditioning experiment

Multiple starts: 12 problems (4 per category) at n = 10 and n = 100, nominal starting point and 10 perturbed points at
each of the radii σ = 0.1 and σ = 1.0, i.e. 462 starts per solver and 2772 runs. Conditioning experiment: rotated
quadratics, 20 random starting points per cell, panel (a) n ∈ {10, …, 1000} at κ = 10⁴, panel (b) κ ∈ {10², …, 10⁵}
at n = 100, iteration cap K_max = 5000.

```bash
python scripts/run/run_multistart_v0_5.py           # rerun/results_multistart_v0.5.csv (the other four solvers)
python scripts/run/run_G3_v0_5.py                   # rerun/results_G3_v0.5.csv (the other four solvers)
python scripts/analysis/p0_fichiers_complets.py     # merges with the earlier rows of Cauchy Point and Steihaug-CG
```

### 3. Wall-clock timing

Times are measured in a dedicated run, separate from the runs that produce success and oracle counts, with diagnostic
instrumentation disabled (`instrument=False`). The reported times were measured on an Intel Core i7-1270P on mains
power, Windows 11, with OpenBLAS 0.3.31.188.0 and `OMP_NUM_THREADS`, `MKL_NUM_THREADS` and `OPENBLAS_NUM_THREADS` set to
1, in a single process pinned to one logical core; each task is repeated three times and timed by the median, tasks
below 1 ms are excluded from time comparisons, and the whole measurement was carried out twice (pass A, reported, and
pass B, replication).

```bash
python scripts/timing/a1_core_selection.py          # times a control workload on each logical core
python scripts/timing/timing_harness.py --core <k>  # passes A and B; outputs in rerun/timing/ (or TIMING_OUT)
```

`results/timing/regle_adaptee_prealable.md` records the robustness rule fixed before the robustness analysis of the
times, and its adaptation.

### 4. Statistics, tables and figures

```bash
cd scripts/analysis
python etape5_statistiques.py                       # permutation tests (Holm), BCa cluster-bootstrap intervals, costs
python p1_succes_156_et_comparaison_v04.py          # success table over the 156 distinct instances
python tl1_ntot_15paires.py
python etape1_validation_temps_reel.py
python b1_seuil_noeud2.py                           # wall-clock ratio and threshold dimension
python b2_arbre_decision_temps.py                   # decision tree
python b3_validation_LOO.py                         # leave-one-problem-out validation of the rules
python b4_pareto_temps.py                           # Pareto front
python b5_table_S1_temps.py
python p5_multistart_G3.py                          # multiple starts and conditioning experiment
python p6_sensibilite_BFGS.py                       # sensitivity to rounding (earlier algebra vs released algebra)
python s2_rosenbrock_n10.py
python etape1bis_robustesse.py                      # robustness of the conclusions drawn from times
python make_tables_v05.py
python make_figures_v05.py
python p10_profils.py                               # performance and data profiles
python p10_scalabilite.py
python p10_regle_decision.py
python p10_tables.py
```

The scripts read `results/` and write their outputs (CSV and JSON files, `tables/`, `figures/`, `robustesse/`) under
`ANALYSIS_OUT`. Random draws use recorded seeds: from one run to the next, the CSV, JSON, table and PNG outputs are
identical, and the PDF figures differ only in their creation date.

### 5. Sensitivity analysis

The sensitivity of the results to the duplicate Dixon–Price problem (A14 and B11) and to the four problems with
finite-difference derivatives (B3, B5, B10 and C04) is computed by the same analysis with problems excluded
(PowerShell):

```powershell
cd scripts/analysis/sensitivity
.\lancer_variante.ps1                                          # no exclusion
.\lancer_variante.ps1 -Exclure "B11_DixonPrice"
.\lancer_variante.ps1 -Exclure "A14"
.\lancer_variante.ps1 -Exclure "A14,B11_DixonPrice"
.\lancer_variante.ps1 -Exclure "B3_Ackley,B5_Griewank,B10_Schwefel,C04"
python comparer_variantes.py
```

### 6. Unit tests

```bash
python tests/test_derivatives.py                    # and the other tests/test_*.py, one by one
```

Four tests (three in `tests/test_memoize.py`, one in `tests/test_wolfe.py`) compare with result files of earlier
versions that are not distributed here (`results/reference/` does not contain them); they are reported as skipped.

---

## Oracle counts and options

The oracle cost N_tot = N_f + N_g + N_h counts the distinct points at which the function, gradient and Hessian oracles
are evaluated: a call at a point identical, bit for bit, to one of the 256 most recent distinct points already
evaluated by the same oracle (2 for the Hessian) returns the stored value and is not counted. On a sample of 461 runs,
these memory sizes reproduce the exact number of distinct points in every run.

`framework_united_final.run_one(pb, method, instrument=True, memoize=True)`:
- `memoize=True` (default) gives the count by distinct points; `memoize=False` restores the former count, in which
  repeated calls at the same point are counted;
- `instrument=False` disables the diagnostic instrumentation; it is used for wall-clock timing.

---

## Changes since v0.1

From Section 2.4 of the manuscript ("Code version and corrections") and the implementation changes it refers to:

- All results come from a single unified run in which the six solvers share the budget K_max = 1000.
- The gradient of the Beale function (A08) has been corrected.
- The analytical Hessians of four problems (A13, B6, B7 and C02) have been corrected by symbolic differentiation of the
  coded objectives, with a permanent test of the derivatives (`tests/test_derivatives.py`).
- Newton-LM: at every iteration, λ_k is reinitialized to 10⁻³‖g_k‖ and doubled, up to 30 attempts, until the Cholesky
  factorization of B_k + λ_k I succeeds and the direction is a descent direction.
- Dogleg: if the Cholesky factorization of B_k + 10⁻¹²I fails, the Newton step is replaced by the Cauchy point.
- BFGS-Wolfe: the bracketing line search of Nocedal and Wright (Alg. 3.5) with its zoom phase (Alg. 3.6) and cubic
  interpolation.
- The BFGS update is implemented in O(n²) operations; Newton-LM and Dogleg solve their systems by two triangular
  substitutions after the Cholesky factorization.
- Oracle counts by distinct points (see above).
- Four problems of Category C that were unbounded below (C01, C08, C09, C10) are replaced by coercive variants that keep
  the starting point and at least 50 % of the negative curvature present at it.

---

## Data files

| File | Content |
|---|---|
| `results/resultats_benchmark_v0.5.csv` | benchmark: 180 runs per solver for the six solvers (column `code_version`: run of origin of each row) |
| `results/resultats_benchmark_v0.5_meta.json` | commit and code fingerprint of the benchmark run |
| `results/results_multistart_v0.5_complet.csv` | multiple starts: 2772 runs |
| `results/results_G3_v0.5_complet.csv` | conditioning experiment on rotated quadratics |
| `results/timing/timing_catA_v0.5_A.csv`, `..._B.csv` | wall-clock times, passes A and B |
| `results/timing/control_measurement_v0.5_A.csv`, `..._B.csv` | control workload timed at regular intervals |
| `results/timing/plan_phase*_v0.5_*.json`, `a1_core_selection.csv` | execution plans (orders, seeds) and core selection |
| `results/reference/` | outputs of earlier versions of this code (benchmark and multiple starts of v0.4, conditioning experiment of v0.3, and the statistics of v0.4), read only for the comparisons reported in the supplementary material (sensitivity to rounding) and for internal checks |

---

## Citation

If you use this code or these data, please cite the manuscript:

```bibtex
@misc{SlimaniIbnBatouta2026,
  author = {Slimani, Jabrane and Ibn Batouta, Kawtar},
  title  = {Trust-Region vs Line Search for Unconstrained Optimization: A Fair-Cost Study on
            Ill-Conditioned, Non-Convex, and Indefinite-Hessian Problems},
  year   = {2026},
  note   = {Manuscript under review. Code and data: https://github.com/jslimani1989/trust-region-vs-line-search, tag v0.5.2}
}
```

See also `CITATION.cff`.

## License

MIT License — see [LICENSE](LICENSE).
