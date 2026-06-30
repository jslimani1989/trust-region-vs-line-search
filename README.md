# Trust-Region vs Line Search — Unconstrained Optimization Benchmark

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)](phase1/tests/)

Reproducible benchmark comparing **three trust-region subproblem solvers**
(Cauchy Point, Dogleg, Steihaug–Toint CG) against **three line-search methods**
(BFGS-Armijo, BFGS-Wolfe, Newton with backtracking) on unconstrained
optimization instances spanning ill-conditioning, non-convexity, and indefinite
Hessians.

> **Key finding:** Steihaug–Toint CG is the unique Pareto-efficient solver
> across the robustness–scalability plane: strong performance on Cat. A and
> Cat. B, functional on indefinite Hessians, and the only method scaling to
> n = 1000 at O(n²) per iteration.

---

## Repository Structure

```
trust-region-vs-line-search/
├── phase0/
│   ├── framework_united_final.py     # Benchmark engine (oracle instrumentation,
│   │                                  # unified Kmax = 1000, budget_ratio hook)
│   ├── trust_region.py                # Cauchy Point, Dogleg, Steihaug-CG
│   └── implementation_methode_lineaires.py  # BFGS-Armijo, BFGS-Wolfe, Newton-BT
├── phase1/
│   ├── benchmark_all_categories_no_levy_validation.py  # Problem library (A/B/C)
│   ├── tests/
│   │   ├── test_cauchy_point.py       # Unit test: Cauchy Point (2 sub-cases)
│   │   └── test_steihaug_newton.py    # Unit test: Steihaug → Newton (Δ→∞)
│   ├── analysis/
│   │   └── data_profiles.py           # Data profiles (Moré & Wild 2009)
│   └── results/
│       └── resultats_benchmark_v2.csv # Phase 1 benchmark results (frozen, tag v0.1)
└── phase2/
    ├── lbfgs_wrapper.py                # L-BFGS (m ∈ {5,10,20}), two-loop recursion
    ├── plot_lbfgs.py                   # L-BFGS vs Steihaug-CG/BFGS-Wolfe figure
    ├── wilcoxon_tests.py               # Wilcoxon signed-rank + Holm correction
    ├── pycutest_benchmark.py           # CUTEst validation suite (Tier 1 problems)
    ├── resultats_benchmark.csv         # Corrected benchmark results (Beale gradient fix)
    ├── results_lbfgs.csv               # L-BFGS large-scale experiment results
    ├── results_cutest.csv              # CUTEst raw results
    ├── audit_cutest.csv                # Benchmark vs CUTEst convergence comparison
    ├── wilcoxon_results.csv            # Pairwise statistical test results
    ├── wilcoxon_table.tex              # LaTeX table of statistical results
    ├── decision_tree_annotations.txt   # Empirical support per decision-tree node
    └── figure_lbfgs_S5.pdf / .png      # L-BFGS scalability figure
```

---

## Installation

```bash
git clone https://github.com/jslimani1989/trust-region-vs-line-search.git
cd trust-region-vs-line-search
pip install -r requirements.txt
```

**Requirements:** Python ≥ 3.10, NumPy, Matplotlib, SciPy, pandas.
No third-party optimization library is used for the six core solvers —
all are implemented from scratch.

---

## Phase 0 — Core Solvers and Benchmark Engine

### Reproducing the Results

**Run the full benchmark** (~5h on a modern laptop):

```bash
python phase0/framework_united_final.py --csv phase1/results/my_results.csv
```

**Quick validation** (n ∈ {10, 50, 100}, ~20 min):

```bash
python phase0/framework_united_final.py --medium --csv phase1/results/my_results_medium.csv
```

**Single category or method:**

```bash
python phase0/framework_united_final.py --category A --csv phase1/results/cat_a.csv
python phase0/framework_united_final.py --method Steihaug_CG --csv phase1/results/steihaug.csv
```

### Generate Data Profiles (Moré & Wild 2009)

```bash
python phase1/analysis/data_profiles.py phase1/results/resultats_benchmark_v2.csv phase1/analysis/
```

Produces `data_profiles_phase0.png` and `data_profiles_phase0.pdf`.

### Run Unit Tests

```bash
python phase1/tests/test_cauchy_point.py
python phase1/tests/test_steihaug_newton.py
```

Expected output:
```
✓  A — Newton step (Δ=10, ‖g‖<Δ)  [err = 0.00e+00]
✓  B — Boundary    (Δ= 2, ‖g‖>Δ)  [err = 0.00e+00]
  Result: 2/2 tests passed.

✓  Steihaug → Newton  (Δ=1e8, A=diag(2,3), x₀=[1,1])
  Result: 1/1 tests passed.
```

---

## Benchmark Design

### Problem Categories

| Category | Description | # Problems | Dimensions |
|---|---|---|---|
| A | Ill-conditioned (κ ∈ {10², 10³, 10⁴, 10⁵}) | 15 | {10, 50, 100, 500, 1000} |
| B | Non-convex (Rosenbrock, Rastrigin, CUTEst-like) | 11 | {10, 50, 100, 500, 1000} |
| C | Indefinite Hessian at x* | 10 | {10, 50, 100, 500, 1000} |
| **Total** | | **36** | **180 instances** |

### Fair-Cost Protocol

All six solvers share a **uniform iteration budget** `Kmax = 1000`.
The oracle cost is tracked separately:

```
N_tot = N_f + N_g + N_h
```

Line-search methods (BFGS variants) naturally incur `N_h = 0`, making
comparisons equitable. Results are reported via Dolan–Moré performance
profiles and Moré–Wild data profiles.

---

## Methods

### Trust-Region Solvers (`phase0/trust_region.py`)

- **Cauchy Point:** minimizes the quadratic model along the steepest-descent
  direction; O(n) per iteration.
- **Dogleg:** interpolates the Cauchy direction and the Newton step via
  Cholesky factorization; O(n²) per iteration.
- **Steihaug–Toint CG:** truncated conjugate gradient solving Bₖp = −gₖ;
  stops on negative curvature (dᵗBₖd ≤ 0), boundary, or
  Eisenstat–Walker tolerance; O(n²) per iteration; does not require Bₖ ≻ 0.

### Line-Search Methods (`phase0/implementation_methode_lineaires.py`)

- **BFGS-Armijo:** BFGS with backtracking Armijo line search.
- **BFGS-Wolfe:** BFGS with strong Wolfe conditions (zoom + cubic interpolation).
- **Newton-BT:** Exact Newton direction with Levenberg–Marquardt regularization
  and Armijo backtracking; the regularization guarantees a descent direction
  even when the Hessian is indefinite (Category C).

---

## Phase 2 — Large-Scale, Statistical, and External Validation

Phase 2 extends the Phase 1 benchmark along three independent axes.

### 1. L-BFGS Large-Scale Comparison

`lbfgs_wrapper.py` implements limited-memory BFGS (two-loop recursion,
Nocedal & Wright Alg. 7.4) with memory parameters m ∈ {5, 10, 20}, using the
same strong-Wolfe line search as `BFGS_Wolfe` for a fair comparison.

```bash
python phase2/lbfgs_wrapper.py
```

Produces `results_lbfgs.csv`: convergence and oracle counts on a rotated
quadratic family across two sweeps — dimension n ∈ {10, 50, 100, 500, 1000}
at fixed κ = 10⁴, and condition number κ ∈ {10², 10³, 10⁴, 10⁵} at fixed
n = 100.

```bash
python phase2/plot_lbfgs.py phase2/results_lbfgs.csv phase2/resultats_benchmark.csv
```

Produces `figure_lbfgs_S5.pdf/.png`, comparing L-BFGS against Steihaug-CG
and BFGS-Wolfe. **Finding:** L-BFGS fails to converge at κ ≥ 10⁴ regardless
of memory size, while Steihaug-CG remains flat in oracle cost across all
tested dimensions — confirming the scalability advantage of trust-region
methods on ill-conditioned problems independently of the line-search
memory budget.

### 2. Statistical Validation (Wilcoxon + Holm)

`wilcoxon_tests.py` performs paired Wilcoxon signed-rank tests with
Holm–Bonferroni correction (α = 0.05) on eight targeted method pairs,
separately for each problem category.

```bash
python phase2/wilcoxon_tests.py --input phase2/resultats_benchmark.csv
```

Produces:
- `wilcoxon_results.csv` — full statistical results (p-values, corrected
  p-values, significance annotations)
- `wilcoxon_table.tex` — ready-to-include LaTeX table
- `decision_tree_annotations.txt` — empirical support for each branch of
  the method-selection decision tree

Use `--pairs all` to test all 15 pairwise comparisons instead of the
8 targeted pairs.

### 3. External Validation via CUTEst

`pycutest_benchmark.py` re-runs all six solvers on a subset of problems
imported directly from the [CUTEst](https://github.com/ralna/CUTEst)
collection via [pycutest](https://github.com/jfowkes/pycutest), to validate
that the custom benchmark problem implementations match independently
maintained reference formulations.

Requires a working pycutest + CUTEst installation (see
[pycutest documentation](https://github.com/jfowkes/pycutest) for setup).

```bash
python phase2/pycutest_benchmark.py --bench phase2/resultats_benchmark.csv
```

Produces `results_cutest.csv` and `audit_cutest.csv` (convergence-rate
comparison, flagged OK / WARN / CRITICAL by deviation in percentage points).

**Validation outcome:** two problems (Beale, Freudenstein–Roth) show exact
agreement between the custom and CUTEst formulations. Four scalable problems
(Wood, Broyden tridiagonal, Broyden banded, Chained Woods) show formulation-level
differences in starting point and/or objective scaling between the custom
benchmark and the corresponding CUTEst SIF definitions — these are documented
as known formulation variants, not implementation errors.

### Notes on `resultats_benchmark.csv`

This file in `phase2/` reflects a corrected gradient implementation for the
Beale problem (a sign error in the second partial derivative was identified
and fixed; convergence on this instance moved from 0% to 100% across all six
methods). It supersedes `phase1/results/resultats_benchmark_v2.csv` for all
analyses performed in Phase 2. The Phase 1 file is kept unchanged as a
historical record matching tag `v0.1`.

---

## Reproducibility Notes

- All methods use identical random seeds (fixed via problem definitions).
- The benchmark engine wraps all oracles in transparent counters
  (`_make_counted_oracles`) so N_f, N_g, N_h are collected without
  modifying solver code.
- `phase1/results/resultats_benchmark_v2.csv` was produced with the
  codebase tagged `v0.1`.
- `phase2/resultats_benchmark.csv` was produced with the same codebase
  after the Beale gradient correction described above.

---

## Citation

If you use this code or results, please cite:

```bibtex
@misc{TODO_citation,
  author  = {TODO},
  title   = {Trust-Region vs Line Search: A Fair-Cost Benchmark},
  year    = {2026},
  url     = {https://github.com/jslimani1989/trust-region-vs-line-search},
  note    = {Version 0.1}
}
```

See also `CITATION.cff` for machine-readable citation metadata.

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

## References

- Conn, Gould & Toint (2000). *Trust-Region Methods*. SIAM.
- Nocedal & Wright (2006). *Numerical Optimization* (2nd ed.). Springer.
- Steihaug (1983). The conjugate gradient method and trust regions
  in large scale optimization. *SIAM J. Numer. Anal.*, 20(3), 626–637.
- Dolan & Moré (2002). Benchmarking optimization software with
  performance profiles. *Math. Programming*, 91(2), 201–213.
- Moré & Wild (2009). Benchmarking derivative-free optimization algorithms.
  *SIAM J. Optim.*, 20(1), 172–191.
- Gould, Orban & Toint (2015). CUTEst: a constrained and unconstrained
  testing environment with safe threads for mathematical optimization.
  *Computational Optimization and Applications*, 60(3), 545–557.
