# -*- coding: utf-8 -*-
"""
test_algebre_v05.py
------------------------------------------------------------
TEST UNITAIRE -- corrections d'algebre lineaire de v0.5 (consigne 009d, Etape 3)

1) BFGS : la nouvelle mise a jour en O(n^2),
       H' = H - rho (s (Hy)^T + (Hy) s^T) + (rho^2 y^T H y + rho) s s^T,
   coincide avec l'ancienne formule (I - rho s y^T) H (I - rho y s^T) + rho s s^T (ecart relatif en norme de
   Frobenius < 1e-10) et la nouvelle matrice est EXACTEMENT symetrique (H est symetrique exactement : H = (M + M^T)/2,
   comme dans les solveurs ou H0 = I et la mise a jour preserve la symetrie). 200 tirages par dimension
   n in {5, 50, 200} ; H = Q diag(lambda) Q^T symetrisee, lambda dans [0,1 ; 10] ; s, y avec y^T s >= 0,1 ||s|| ||y||.
2) Resolutions : sur des matrices symetriques definies positives aleatoires (conditionnement <= 1e6),
   n in {5, 50, 500}, l'ecart relatif entre l'ancienne resolution np.linalg.solve(L.T, np.linalg.solve(L, b)) et la
   nouvelle (deux scipy.linalg.solve_triangular) est < 1e-10 ; 20 tirages par dimension.
3) Les solveurs modifies (bfgs, bfgs_wolfe, newton_backtracking, trust_region_dogleg) convergent sur une
   quadratique SPD de petite dimension (garde-fou d'integration).
Graine : 20261011 (notee).
"""
import os
import sys

import numpy as np
from scipy.linalg import solve_triangular

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import implementation_methode_lineaires as ls
import trust_region as tr

SEED = 20261011
TOL_REL = 1e-10


def _old_bfgs_update(H, s, y):
    rho = 1.0 / (s @ y)
    I = np.eye(len(s))
    A = I - rho * np.outer(s, y)
    B = I - rho * np.outer(y, s)
    return A @ H @ B + rho * np.outer(s, s)


def _new_bfgs_update(H, s, y):
    rho = 1.0 / (s @ y)
    Hy = H @ y
    yHy = float(y @ Hy)
    return H - rho * (np.outer(s, Hy) + np.outer(Hy, s)) + (rho**2 * yHy + rho) * np.outer(s, s)


def test_bfgs_update_matches_old_formula():
    rng = np.random.default_rng(SEED)
    worst, sym_ok, n_draws = 0.0, True, 0
    for n in (5, 50, 200):
        for _ in range(200):
            Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
            lam = rng.uniform(0.1, 10.0, size=n)
            M = (Q * lam) @ Q.T
            H = 0.5 * (M + M.T)
            s = rng.standard_normal(n)
            while True:  # y^T s >= 0,1 ||s|| ||y||
                y = s + rng.uniform(0.0, 3.0) * rng.standard_normal(n)
                if y @ s >= 0.1 * np.linalg.norm(s) * np.linalg.norm(y):
                    break
            Ho, Hn = _old_bfgs_update(H, s, y), _new_bfgs_update(H, s, y)
            worst = max(worst, np.linalg.norm(Hn - Ho, "fro") / np.linalg.norm(Ho, "fro"))
            sym_ok = sym_ok and np.array_equal(Hn, Hn.T)
            n_draws += 1
    assert sym_ok, "la nouvelle matrice n'est pas exactement symetrique"
    assert worst < TOL_REL, f"ecart relatif maximal {worst:.3e} >= {TOL_REL:g}"
    return f"{n_draws} tirages, ecart relatif maximal {worst:.2e}, symetrie exacte"


def test_triangular_solves_match_old():
    rng = np.random.default_rng(SEED + 1)
    worst, n_draws = 0.0, 0
    for n in (5, 50, 500):
        for _ in range(20):
            Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
            lam = 10.0 ** rng.uniform(0.0, 6.0, size=n)
            lam[0], lam[-1] = 1.0, 1e6  # conditionnement exactement 1e6
            M = (Q * lam) @ Q.T
            A = 0.5 * (M + M.T)
            b = rng.standard_normal(n)
            L = np.linalg.cholesky(A)
            old = np.linalg.solve(L.T, np.linalg.solve(L, b))
            new = solve_triangular(L.T, solve_triangular(L, b, lower=True, check_finite=False),
                                   lower=False, check_finite=False)
            worst = max(worst, np.linalg.norm(new - old) / np.linalg.norm(old))
            n_draws += 1
    assert worst < TOL_REL, f"ecart relatif maximal {worst:.3e} >= {TOL_REL:g}"
    return f"{n_draws} tirages (n = 5, 50, 500 ; cond = 1e6), ecart relatif maximal {worst:.2e}"


def test_modified_solvers_converge():
    n = 8
    rng = np.random.default_rng(SEED + 2)
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    A = (Q * np.linspace(1.0, 50.0, n)) @ Q.T
    A = 0.5 * (A + A.T)
    f = lambda x: 0.5 * float(x @ A @ x)
    g = lambda x: A @ x
    h = lambda x: A
    x0 = np.ones(n)
    out = []
    for name, fn in (("bfgs", lambda: ls.bfgs(f, g, x0, tol=1e-8, max_iter=500)),
                     ("bfgs_wolfe", lambda: ls.bfgs_wolfe(f, g, x0, tol=1e-8, max_iter=500)),
                     ("newton_backtracking", lambda: ls.newton_backtracking(f, g, h, x0, tol=1e-8, max_iter=100)),
                     ("trust_region_dogleg", lambda: tr.trust_region_dogleg(f, g, h, x0, tol=1e-8, max_iter=200))):
        res = fn()
        if isinstance(res, dict):
            ok = res["status"] == "converged" and np.linalg.norm(g(res["x_opt"])) < 1e-7
            it = res["iterations"]
        else:
            ok = res.converged and res.grad_norm < 1e-7
            it = res.iterations
        assert ok, f"{name} n'a pas converge"
        out.append(f"{name}:{it}")
    return "convergence en iterations : " + ", ".join(out)


TESTS = [
    ("(1) BFGS : nouvelle mise a jour = ancienne formule (< 1e-10), symetrie exacte", test_bfgs_update_matches_old_formula),
    ("(2) Resolutions triangulaires = anciennes resolutions (< 1e-10)", test_triangular_solves_match_old),
    ("(3) Les quatre solveurs modifies convergent sur une quadratique SPD", test_modified_solvers_converge),
]


def run():
    print("=" * 70)
    print("TEST UNITAIRE - corrections d'algebre lineaire de v0.5 (consigne 009d)")
    print("=" * 70)
    passed = 0
    for label, fn in TESTS:
        try:
            info = fn()
            print(f"  OK    {label}\n        {info}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL  {label}  [{e}]")
    print("-" * 70)
    print(f"  Resultat : {passed}/{len(TESTS)} tests passes.")
    print("=" * 70)
    return passed == len(TESTS)


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
