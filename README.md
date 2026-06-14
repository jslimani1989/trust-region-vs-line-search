# Trust-Region vs Line Search — Unconstrained Optimization Benchmark

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)](tests/)

Reproducible benchmark comparing **three trust-region subproblem solvers**
(Cauchy Point, Dogleg, Steihaug–Toint CG) against **three line-search methods**
(BFGS-Armijo, BFGS-Wolfe, Newton with backtracking) on 180 unconstrained
optimization instances spanning ill-conditioning, non-convexity, and indefinite
Hessians.

> **Key finding:** Steihaug–Toint CG is the unique Pareto-efficient solver
> across the robustness–scalability plane: best Cat. A (82.7%), best Cat. B
> (89.1%), functional on indefinite Hessians (58.0%), and the only method
> scaling to n = 1000 at O(n²) per iteration.

---

## Repository Structure

```
trust-region-vs-line-search/
├── framework_united_final.py          # Benchmark engine (oracle instrumentation,
│                                      # unified Kmax = 1000, budget_ratio hook)
├── trust_region.py                    # Cauchy Point, Dogleg, Steihaug-CG
├── implementation methode lineaires.py# BFGS-Armijo, BFGS-Wolfe, Newton-BT
├── benchmark_all_categories_no_levy_validation.py  # Problem library (A/B/C)
├── tests/
│   ├── test_cauchy_point.py           # Unit test: Cauchy Point (2 sub-cases)
│   └── test_steihaug_newton.py        # Unit test: Steihaug → Newton (Δ→∞)
├── analysis/
│   └── data_profiles.py              # Data profiles (Moré & Wild 2009)
├── results/
│   └── resultats_benchmark_v2.csv    # Full benchmark results (180×6 runs)
└── paper/
    ├── article_conference.tex         # Conference article skeleton
    └── references_conf.bib           # BibTeX references
```

---

## Installation

```bash
git clone https://github.com/TODO/trust-region-vs-line-search.git
cd trust-region-vs-line-search
pip install -r requirements.txt
```

**Requirements:** Python ≥ 3.10, NumPy, Matplotlib.
No third-party optimization library is used — all six solvers are implemented
from scratch.

---

## Reproducing the Results

### Run the full benchmark (~5h on a modern laptop)

```bash
python framework_united_final.py --csv results/my_results.csv
```

**Quick validation** (n ∈ {10, 50, 100}, ~20 min):

```bash
python framework_united_final.py --medium --csv results/my_results_medium.csv
```

**Single category or method:**

```bash
python framework_united_final.py --category A --csv results/cat_a.csv
python framework_united_final.py --method Steihaug_CG --csv results/steihaug.csv
```

### Generate data profiles (Moré & Wild 2009)

```bash
python analysis/data_profiles.py results/resultats_benchmark_v2.csv analysis/
```

Produces `data_profiles_phase0.png` and `data_profiles_phase0.pdf`.

### Run unit tests

```bash
python tests/test_cauchy_point.py
python tests/test_steihaug_newton.py
```

Expected output:
```
✓  A — Pas Newton (Δ=10, ‖g‖<Δ)  [err = 0.00e+00]
✓  B — Frontière  (Δ= 2, ‖g‖>Δ)  [err = 0.00e+00]
  Résultat : 2/2 tests passés.

✓  Steihaug → Newton  (Δ=1e8, A=diag(2,3), x₀=[1,1])
  Résultat : 1/1 tests passés.
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

### Success Rates (v2, uniform Kmax = 1000)

| Method | Cat. A | Cat. B | Cat. C | Global |
|---|---|---|---|---|
| Cauchy Point | 10.7% | 61.8% | 48.0% | 36.7% |
| Dogleg | 82.7% | 81.8% | 50.0% | 73.3% |
| **Steihaug-CG** | **82.7%** | **89.1%** | **58.0%** | **77.8%** |
| BFGS-Armijo | 68.0% | 76.4% | 50.0% | 65.6% |
| BFGS-Wolfe | 69.3% | 67.3% | 28.0% | 57.2% |
| Newton-BT | 80.0% | 87.3% | 80.0% | 82.2% |

---

## Methods

### Trust-Region Solvers (`trust_region.py`)

- **Cauchy Point:** minimizes the quadratic model along the steepest-descent
  direction; O(n) per iteration.
- **Dogleg:** interpolates the Cauchy direction and the Newton step via
  Cholesky factorization; O(n²) per iteration.
- **Steihaug–Toint CG:** truncated conjugate gradient solving Bₖp = −gₖ;
  stops on negative curvature (dᵀBₖd ≤ 0), boundary, or
  Eisenstat–Walker tolerance; O(n²) per iteration; does not require Bₖ ≻ 0.

### Line-Search Methods (`implementation methode lineaires.py`)

- **BFGS-Armijo:** BFGS with backtracking Armijo line search.
- **BFGS-Wolfe:** BFGS with strong Wolfe conditions (zoom + cubic interpolation).
- **Newton-BT:** Exact Newton direction with Armijo backtracking;
  fallback to −gₖ when Hₖ is singular.

---

## Reproducibility Notes

- All methods use identical random seeds (fixed via problem definitions).
- The benchmark engine wraps all oracles in transparent counters
  (`_make_counted_oracles`) so N_f, N_g, N_h are collected without
  modifying solver code.
- Results in `results/resultats_benchmark_v2.csv` were produced with
  this exact codebase (tag `v0.1`).

---

## Citation

If you use this code or results, please cite:

```bibtex
@misc{TODO_citation,
  author  = {TODO},
  title   = {Trust-Region vs Line Search: A Fair-Cost Benchmark},
  year    = {2026},
  url     = {https://github.com/TODO/trust-region-vs-line-search},
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
