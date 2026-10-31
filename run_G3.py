#!/usr/bin/env python3
"""
run_G3.py — Experience kappa-controlee (figure pivot G3, peuple Section 3)
Adapte aux imports de v0.3 (consigne 008b, etape 1d) : le protocole
(constantes, graines, familles de problemes) est IDENTIQUE, terme a terme,
a la version d'origine retrouvee dans le dossier de travail de juin 2026
(Phase1/run_G3.py, non modifie) -- seule la resolution de
framework_united_final change (import du depot courant, pas d'une copie
co-localisee dans un dossier Phase1/ separe).

Sortie  : results_G3.csv (memes colonnes qu'en juin).

USAGE
    python run_G3.py                 # run complet 20 seeds, framework reel
    python run_G3.py --seeds 5       # passe rapide (medianes grossieres)
    python run_G3.py --smoke         # plomberie seule, sans framework
    python run_G3.py --out path.csv
"""

import argparse
import csv
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ============================================================================
# 1. CONSTANTES (identiques a juin : Phase1/run_G3.py, SPEC_G3.md S3/S5/S6)
# ============================================================================
KMAX          = 5000       # deviation G3 (D1) -- != KMAX du benchmark (1000)
TOL           = 1e-6
N_SEEDS       = 20

PANEL_A = {"n": 100, "kappas": [1e2, 1e3, 1e4, 1e5]}
PANEL_B = {"kappa": 1e4, "ns": [10, 50, 100, 500, 1000]}

METHODS = ["cauchy", "dogleg", "steihaug",
           "bfgs_armijo", "bfgs_wolfe", "newton_bt"]

CSV_FIELDS = ["panel", "method", "kappa", "n", "seed", "iter_outer",
              "cg_inner_total", "nfev", "ngev", "nhev",
              "converged", "gnorm_final", "cpu_s"]

def SEED_Q(n):  return 770000 + n
def SEED_X0(k): return 880000 + k

# ============================================================================
# 2. FAMILLE DE PROBLEMES (identique a juin)
# ============================================================================
def make_quadratic(kappa, n):
    """f(x)=1/2 x^T A x, A=Q^T D Q, spectre log-uniforme, cond(A)=kappa (exact)."""
    i   = np.arange(n)
    lam = kappa ** (i / max(n - 1, 1))
    rng = np.random.default_rng(SEED_Q(n))
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    A = Q.T @ (lam[:, None] * Q)
    A = 0.5 * (A + A.T)
    f = lambda x: 0.5 * float(x @ (A @ x))
    g = lambda x: A @ x
    h = lambda x: A
    return f, g, h

def make_x0(n, k):
    rng = np.random.default_rng(SEED_X0(k))
    v   = rng.standard_normal(n)
    return v / np.linalg.norm(v)

# ============================================================================
# 3. ADAPTATEUR FRAMEWORK (v0.3 : import du depot courant)
# ============================================================================
FW_NAME = {
    "cauchy":      "Cauchy_Point",
    "dogleg":      "Dogleg",
    "steihaug":    "Steihaug_CG",
    "bfgs_armijo": "BFGS_Armijo",
    "bfgs_wolfe":  "BFGS_Wolfe",
    "newton_bt":   "Newton_BT",
}

# NOTE (008b, 1d) : en juin, cg_inner_total etait renseigne pour Steihaug-CG
# grace a un patch de comptage dans trust_region.py, propre au dossier de
# travail Phase1/ (non present dans v0.2/v0.3 : trust_region.py de ce depot
# n'expose aucun compteur d'iterations CG internes). Aucune modification de
# trust_region.py n'est autorisee par la consigne 008b (hors perimetre de
# 1a-1g) : cg_inner_total reste donc VIDE pour Steihaug-CG dans les donnees
# v0.3, a la difference des donnees de juin -- non determinable ici, pas une
# omission. Signale dans le rapport 008b-suite.

def make_real_registry():
    import framework_united_final as fw

    def solve(method, f, g, h, x0, kmax):
        # Surcharge du budget module avant chaque appel (D1 : Kmax=5000)
        fw.KMAX = kmax

        pb = {
            "id":       "G3_quad",
            "name":     "G3 rotated quadratic",
            "category": "A",
            "n":        int(len(x0)),
            "f":        f,
            "grad_f":   g,
            "hess_f":   h,
            "x0":       x0.copy(),
            "f_opt":    0.0,
        }

        res = fw.run_one(pb, FW_NAME[method])

        cg = None
        if method == "steihaug":
            cg_raw = res.get("cg_inner_total")
            if cg_raw is not None:
                try:
                    cg = int(cg_raw)
                except (TypeError, ValueError):
                    cg = None

        return {
            "iter_outer":     int(res["n_iter"]),
            "nfev":           int(res["n_f_evals"]),
            "ngev":           int(res["n_g_evals"]),
            "nhev":           int(res["n_h_evals"]),
            "converged":      bool(res["converged"]),
            "gnorm_final":    float(res["grad_norm"]) if res["grad_norm"] is not None else float("nan"),
            "cg_inner_total": cg,
        }

    return solve

# ============================================================================
# 3bis. REGISTRE FACTICE (--smoke) -- identique a juin
# ============================================================================
def make_mock_registry():
    class _Cnt:
        def __init__(self, fn): self.fn, self.n = fn, 0
        def __call__(self, x): self.n += 1; return self.fn(x)

    def solve(method, f, g, h, x0, kmax):
        cf, cg, ch = _Cnt(f), _Cnt(g), _Cnt(h)
        x = x0.copy()
        A = ch(x)
        if method in ("newton_bt", "dogleg", "steihaug"):
            x = x - np.linalg.solve(A, cg(x)); cf(x); it = 1
        else:
            it = 0
            for _ in range(min(kmax, 200)):
                gr = cg(x)
                if np.linalg.norm(gr) < TOL: break
                alpha = float(gr @ gr) / float(gr @ (A @ gr))
                x = x - alpha * gr; cf(x); it += 1
        gn = float(np.linalg.norm(g(x)))
        return {
            "iter_outer": it, "nfev": cf.n, "ngev": cg.n, "nhev": ch.n,
            "converged": gn < TOL, "gnorm_final": gn,
            "cg_inner_total": (it if method == "steihaug" else None),
        }
    return solve

# ============================================================================
# 4. BOUCLE EXPERIENCE (pairing identique a juin)
# ============================================================================
def run_cell(panel, kappa, n, n_seeds, solve, kmax):
    rows = []
    f, g, h = make_quadratic(kappa, n)             # A construite une fois / cellule
    for k in range(n_seeds):
        x0 = make_x0(n, k)                         # meme x0 pour les 6 methodes
        for method in METHODS:
            t0 = time.perf_counter()
            r  = solve(method, f, g, h, x0, kmax)
            r["cpu_s"] = time.perf_counter() - t0
            r.update(panel=panel, method=method, kappa=kappa, n=n, seed=k)
            rows.append(r)
    return rows

def run_all(n_seeds, solve, kmax):
    rows = []
    for kappa in PANEL_A["kappas"]:
        rows += run_cell("A", kappa, PANEL_A["n"], n_seeds, solve, kmax)
        print(f"  [A] kappa={kappa:.0e}  n={PANEL_A['n']:4d}  cumul={len(rows)}")
    for n in PANEL_B["ns"]:
        rows += run_cell("B", PANEL_B["kappa"], n, n_seeds, solve, kmax)
        print(f"  [B] kappa={PANEL_B['kappa']:.0e}  n={n:4d}  cumul={len(rows)}")
    return rows

# ============================================================================
# 5. ECRITURE CSV (schema identique a juin)
# ============================================================================
def write_csv(rows, path):
    with open(path, "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=CSV_FIELDS)
        w.writeheader()
        for r in rows:
            out = {k: r[k] for k in CSV_FIELDS}
            if out["cg_inner_total"] is None:
                out["cg_inner_total"] = ""
            w.writerow(out)

# ============================================================================
# 6. MAIN
# ============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out",   default="results_G3.csv")
    ap.add_argument("--seeds", type=int, default=N_SEEDS)
    ap.add_argument("--kmax",  type=int, default=KMAX,
                    help=f"budget iterations (defaut G3 : {KMAX})")
    ap.add_argument("--smoke", action="store_true",
                    help="solveur factice -- valide la plomberie sans framework")
    args = ap.parse_args()

    mode  = "SMOKE" if args.smoke else "REEL"
    solve = make_mock_registry() if args.smoke else make_real_registry()

    print(f"=== run_G3 [{mode}] | seeds={args.seeds} | Kmax={args.kmax} ===")
    t0   = time.perf_counter()
    rows = run_all(args.seeds, solve, args.kmax)
    write_csv(rows, args.out)
    n_total_attendu = (len(PANEL_A["kappas"]) + len(PANEL_B["ns"])) * args.seeds * len(METHODS)
    print(f"--- {len(rows)} lignes ecrites dans '{args.out}' "
          f"(attendu {n_total_attendu}) en {time.perf_counter()-t0:.1f}s ---")

if __name__ == "__main__":
    main()
