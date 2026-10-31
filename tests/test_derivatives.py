# -*- coding: utf-8 -*-
"""
test_derivatives.py
------------------------------------------------------------
TEST UNITAIRE PERMANENT — Derivees des 36 problemes du benchmark (consigne 003b, Etape 2)

Reprend la logique du controle A2 (consigne 003) : verifie, par
differences finies centrees en x0 et en 2 points aleatoires (graine
fixee), que le gradient ET la Hessienne analytiques de chacun des 36
problemes concordent avec l'approximation numerique a moins de 1e-6
d'erreur relative.

Ce test a ete ecrit pour reproduire l'echec constate au controle A2 du
rapport 003 (commit 623b85b : A13, B6_Broydn3D, B7_BryBnd et C02 ont une
Hessienne fausse) et pour verifier que le commit 1g (correction
symbolique SymPy) le fait passer. Il doit rester dans le depot pour
detecter toute regression future sur les 36 fonctions de problemes.

Le gradient ET la Hessienne sont verifies a la plus petite dimension
testee du benchmark uniquement (bm.adjust_n(pid, 10) : n=10, ou sa valeur
ajustee par probleme -- ex. n=8 pour A09, ou la dimension fixe pour les
problemes n=2/n=3), pas a chaque dimension du benchmark. C'est deja la
dimension a laquelle la consigne demande de deriver symboliquement les
corrections de dimension variable (B6/B7 a n=10) ; se limiter a cette
seule dimension garde le test rapide (Hessienne en O(n^2) par differences
finies), au prix de ne pas couvrir explicitement les dimensions
superieures (500, 1000) -- non requis par la consigne, les formules
analytiques etant independantes de n pour tous les problemes concernes.
(Precision 008b/1e : la mention anterieure d'une verification du
gradient "a chaque dimension" etait inexacte -- corrigee ici.)
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import benchmark_all_categories_no_levy_validation as bm

SEED = 20260924
TOL = 1e-6


def _fd_grad(f, x, h=1e-6):
    x = np.asarray(x, dtype=float)
    n = len(x)
    g = np.zeros(n)
    for i in range(n):
        e = np.zeros(n); e[i] = h
        g[i] = (f(x + e) - f(x - e)) / (2 * h)
    return g


def _fd_hess(grad_f, x, h=1e-5):
    x = np.asarray(x, dtype=float)
    n = len(x)
    H = np.zeros((n, n))
    for i in range(n):
        e = np.zeros(n); e[i] = h
        H[:, i] = (grad_f(x + e) - grad_f(x - e)) / (2 * h)
    return 0.5 * (H + H.T)


def _check_problem(pid):
    rng = np.random.default_rng(SEED)
    n_use = bm.adjust_n(pid, 10)
    pb = bm.get_problem(pid, n_use)
    n_use = pb["n"]
    x0 = pb["x0"]
    pts = [x0, x0 + 0.05 * rng.standard_normal(n_use), x0 + 0.05 * rng.standard_normal(n_use)]

    worst_g, worst_h = 0.0, 0.0
    for x in pts:
        g_ana = pb["grad_f"](x)
        g_fd = _fd_grad(pb["f"], x)
        denom_g = max(1.0, np.max(np.abs(g_fd)))
        worst_g = max(worst_g, np.max(np.abs(g_ana - g_fd)) / denom_g)

        H_ana = pb["hess_f"](x)
        H_fd = _fd_hess(pb["grad_f"], x)
        denom_h = max(1.0, np.max(np.abs(H_fd)))
        worst_h = max(worst_h, np.max(np.abs(H_ana - H_fd)) / denom_h)

    return worst_g, worst_h


def test_all_36_problems():
    failures = []
    worst_overall_g, worst_overall_h = 0.0, 0.0
    for pid in bm.PROBLEMS_ALL_IDS:
        worst_g, worst_h = _check_problem(pid)
        worst_overall_g = max(worst_overall_g, worst_g)
        worst_overall_h = max(worst_overall_h, worst_h)
        if worst_g >= TOL or worst_h >= TOL:
            failures.append((pid, worst_g, worst_h))
    assert len(bm.PROBLEMS_ALL_IDS) == 36
    if failures:
        detail = ", ".join(f"{pid}(g={g:.2e},h={h:.2e})" for pid, g, h in failures)
        raise AssertionError(f"{len(failures)}/36 problemes en echec : {detail}")
    return worst_overall_g, worst_overall_h


TESTS = [
    ("Gradient et Hessienne des 36 problemes vs differences finies", test_all_36_problems),
]


def run():
    print("=" * 70)
    print("TEST UNITAIRE PERMANENT — Derivees des 36 problemes (consigne 003b)")
    print("=" * 70)
    passed = 0
    for name, fn in TESTS:
        try:
            wg, wh = fn()
            print(f"  OK  {name}  [max err grad={wg:.2e}, max err hess={wh:.2e}]")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL  {name}")
            print(f"        {e}")
        except Exception as e:
            print(f"  ERROR  {name}  [{type(e).__name__}: {e}]")
    print("-" * 70)
    print(f"  Resultat : {passed}/{len(TESTS)} tests passes.")
    print("=" * 70)
    return passed == len(TESTS)


if __name__ == "__main__":
    ok = run()
    sys.exit(0 if ok else 1)
