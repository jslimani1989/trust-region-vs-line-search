# -*- coding: utf-8 -*-
"""
test_journalisation.py
------------------------------------------------------------
TEST UNITAIRE — Journalisation B1/B2/B3 et classification Q9 (consigne 003 S1e/S1f)

Verifie, sur des problemes-jouets construits pour declencher chaque
branche, sans jamais toucher a la sequence d'iteres produite par les
solveurs (voir commit 1e pour la preuve de non-regression bit-a-bit) :
  1) B3 : un essai non fini dans backtracking_armijo/backtracking_armijo_newton
     est rejete (ls_failed ne se declenche pas prematurement, n_nonfinite>0).
  2) B1 : une "direction" qui n'est jamais de descente force ls_failed=True
     (plancher alpha<1e-14 atteint) dans backtracking_armijo et
     backtracking_armijo_newton.
  3) B2 : newton_backtracking sur un point de depart avec Hessienne
     indefinie produit un lm_trace non vide dont les entrees ont les cles
     attendues et au moins un lambda_final > 0.
  4) Q9 : le classificateur d'ordre 2 de framework_united_final.run_one
     accepte un minimiseur (Hessienne definie positive) comme
     stationnaire_ordre2, et rejette un col (Hessienne indefinie, meme
     gradient nul) -- verifie sur C01' (x=0 est un col, cf. rapport 002b/A8).
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import implementation_methode_lineaires as ls
import framework_united_final as fw
import benchmark_all_categories_no_levy_validation as bm


def test_b3_nonfinite_rejected():
    # f(x) = x^2 pour |x|<1e3, +inf au-dela : un pas initial trop grand
    # produit un essai non fini que le backtracking doit rejeter puis
    # corriger par halvings successifs jusqu'a retomber dans le domaine fini.
    def f(x):
        x = np.asarray(x, dtype=float)
        if np.any(np.abs(x) > 1e3):
            return np.inf
        return float(np.sum(x ** 2))

    def g(x):
        return 2.0 * np.asarray(x, dtype=float)

    xk = np.array([1.0])
    dk = np.array([-1e6])  # alpha0=1 -> xk+dk = -1e6+1, hors domaine fini
    alpha, n_eval, ls_failed, n_nonfinite = ls.backtracking_armijo(f, g, xk, dk, alpha0=1.0)
    assert n_nonfinite > 0, "aucun essai non fini detecte alors que le premier pas sort du domaine fini"
    assert np.isfinite(f(xk + alpha * dk)), "alpha retenu ne ramene pas dans le domaine fini"
    assert not ls_failed, "le backtracking aurait du finir par trouver un pas satisfaisant Armijo"


def test_b1_line_search_failure():
    # f(x)=x^2, direction dk = +1 (ascension pure) : aucune valeur de alpha>0
    # ne peut satisfaire Armijo -> le plancher alpha<1e-14 doit etre atteint.
    def f(x):
        return float(np.sum(np.asarray(x, dtype=float) ** 2))

    def g(x):
        return 2.0 * np.asarray(x, dtype=float)

    xk = np.array([1.0])
    dk = np.array([1.0])  # direction de montee, pas de descente
    alpha, n_eval, ls_failed, _ = ls.backtracking_armijo(f, g, xk, dk, alpha0=1.0)
    assert ls_failed, "ls_failed attendu (direction de montee, Armijo jamais satisfaite)"
    assert alpha < 1e-13

    alpha2, ls_failed2, _ = ls.backtracking_armijo_newton(f, g, xk, dk, alpha0=1.0)
    assert ls_failed2, "ls_failed attendu (backtracking_armijo_newton, direction de montee)"


def test_b2_lm_trace_structure():
    # f(x) = x1^2 - x2^2 (selle pure) : Hessienne diag(2,-2) indefinie en
    # tout point -> la regularisation LM doit s'activer (lambda_final>0).
    def f(x):
        x1, x2 = x
        return float(x1 ** 2 - x2 ** 2)

    def g(x):
        x1, x2 = x
        return np.array([2.0 * x1, -2.0 * x2])

    def h(x):
        return np.diag([2.0, -2.0])

    hist = ls.newton_backtracking(f, g, h, np.array([1.0, 1.0]), max_iter=5, tol=1e-6)
    assert "lm_trace" in hist
    assert len(hist["lm_trace"]) >= 1, "lm_trace vide alors que la Hessienne est indefinie partout"
    entry = hist["lm_trace"][0]
    for key in ("iter", "lambda_final", "n_doublings", "reasons", "lambda_min_H"):
        assert key in entry, f"cle manquante dans lm_trace : {key}"
    assert entry["lambda_final"] > 0.0, "lambda_final devrait etre > 0 (Hessienne indefinie)"
    assert entry["lambda_min_H"] < 0.0, "lambda_min_H devrait etre negatif (diag(2,-2))"


def test_q9_classifier_minimizer_vs_saddle():
    n = 10
    pb = bm.get_problem("C01", n)  # C01' coercive, cf. rapport 001b/003 S1d
    d = bm._c01_build(n)

    # (a) point de selle connu : x=0 -- grad nul, Hessienne = diag(2d) indefinie
    #     car d alterne de signe par construction (_c01_build).
    pb_saddle = dict(pb)
    pb_saddle["x0"] = np.zeros(n)
    row_saddle = fw.run_one(pb_saddle, "BFGS_Armijo")
    assert row_saddle["stationnaire_ordre1"] is True, "grad(0) devrait etre nul pour C01'"
    assert row_saddle["stationnaire_ordre2"] is False, "x=0 est un col (Hessienne indefinie) : ne doit pas etre classe ordre2"

    # (b) minimiseur global connu : xi = +/-1 si d_i<0, 0 sinon (Hessienne DP)
    x_star = np.where(d < 0, 1.0, 0.0)
    pb_min = dict(pb)
    pb_min["x0"] = x_star
    row_min = fw.run_one(pb_min, "BFGS_Armijo")
    assert row_min["stationnaire_ordre1"] is True
    assert row_min["stationnaire_ordre2"] is True, "minimiseur global attendu : Hessienne definie positive"


TESTS = [
    ("B3 : essai non fini rejete par backtracking_armijo", test_b3_nonfinite_rejected),
    ("B1 : echec de recherche lineaire (direction de montee)", test_b1_line_search_failure),
    ("B2 : structure de lm_trace (Newton-LM, Hessienne indefinie)", test_b2_lm_trace_structure),
    ("Q9 : classificateur ordre2 (minimiseur vs col, C01')", test_q9_classifier_minimizer_vs_saddle),
]


def run():
    print("=" * 70)
    print("TEST UNITAIRE — Journalisation B1/B2/B3 et classification Q9 (consigne 003)")
    print("=" * 70)
    passed = 0
    for name, fn in TESTS:
        try:
            fn()
            print(f"  OK  {name}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL  {name}  [{e}]")
        except Exception as e:
            print(f"  ERROR  {name}  [{type(e).__name__}: {e}]")
    print("-" * 70)
    print(f"  Resultat : {passed}/{len(TESTS)} tests passes.")
    print("=" * 70)
    return passed == len(TESTS)


if __name__ == "__main__":
    ok = run()
    sys.exit(0 if ok else 1)
