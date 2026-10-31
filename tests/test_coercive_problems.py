# -*- coding: utf-8 -*-
"""
test_coercive_problems.py
------------------------------------------------------------
TEST UNITAIRE — Variantes coercives C01'/C08'/C09'/C10' (consigne 003 S1d/S1f)

Verifie :
  1) grad_f et hess_f des 4 variantes coercives concordent avec des
     differences finies centrees (controle allege ; le controle bloquant
     A2 complet, sur 36 problemes x 2 points aleatoires, est fait separement).
  2) x0 des variantes coercives est identique (bit a bit) a x0 des
     problemes d'origine (*_orig) -- "les x0 sont inchanges" (consigne).
  3) Les fonctions/dicts d'origine (*_orig) existent toujours mais sont
     hors de PROBLEMS_C / PROBLEMS_ALL_IDS.
  4) f croit bien au-dela de f(x0) tres loin de l'origine (signal de
     coercivite informel ; le controle bloquant A4 rigoureux est separe).
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import benchmark_all_categories_no_levy_validation as bm

DIMS = [10, 50]
IDS = ["C01", "C08", "C09", "C10"]


def _fd_grad(f, x, h=1e-6):
    x = np.asarray(x, dtype=float)
    n = len(x)
    g = np.zeros(n)
    for i in range(n):
        e = np.zeros(n); e[i] = h
        g[i] = (f(x + e) - f(x - e)) / (2 * h)
    return g


def _fd_hess_diag_check(grad_f, x, h=1e-5):
    """Verifie seulement la diagonale de la Hessienne par differences finies
    sur le gradient (suffisant pour un controle allege ; les termes hors
    diagonale de C09'/C10' sont verifies analytiquement dans le rapport)."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    Hd = np.zeros(n)
    for i in range(n):
        e = np.zeros(n); e[i] = h
        Hd[i] = (grad_f(x + e)[i] - grad_f(x - e)[i]) / (2 * h)
    return Hd


def test_gradients_fd():
    rng = np.random.default_rng(42)
    max_err = 0.0
    for pid in IDS:
        for n in DIMS:
            pb = bm.get_problem(pid, n)
            for _ in range(2):
                x = pb["x0"] + 0.05 * rng.standard_normal(n)
                g_analytic = pb["grad_f"](x)
                g_fd = _fd_grad(pb["f"], x)
                err = np.max(np.abs(g_analytic - g_fd)) / max(1.0, np.max(np.abs(g_fd)))
                max_err = max(max_err, err)
    assert max_err < 1e-4, f"Erreur relative gradient (FD) trop grande : {max_err:.2e}"
    return max_err


def test_hessian_diag_fd():
    rng = np.random.default_rng(7)
    max_err = 0.0
    for pid in IDS:
        for n in DIMS:
            pb = bm.get_problem(pid, n)
            x = pb["x0"] + 0.05 * rng.standard_normal(n)
            H_analytic = np.diag(pb["hess_f"](x))
            H_fd = _fd_hess_diag_check(pb["grad_f"], x)
            err = np.max(np.abs(H_analytic - H_fd)) / max(1.0, np.max(np.abs(H_fd)))
            max_err = max(max_err, err)
    assert max_err < 1e-3, f"Erreur relative diagonale Hessienne (FD) trop grande : {max_err:.2e}"
    return max_err


def test_x0_unchanged():
    for pid in IDS:
        for n in DIMS:
            x0_new = bm.PROBLEMS_C[pid]["x0"](n)
            x0_orig = bm.PROBLEMS_C[pid + "_orig"]["x0"](n) if False else None
    # x0 sont des lambdas partagees textuellement identiques ; on verifie
    # directement contre les dicts *_orig conserves dans le module.
    pairs = [("C01", bm.C01_orig), ("C08", bm.C08_orig), ("C09", bm.C09_orig), ("C10", bm.C10_orig)]
    for pid, orig in pairs:
        for n in DIMS:
            x0_new = bm.PROBLEMS_C[pid]["x0"](n)
            x0_orig = orig["x0"](n)
            assert np.array_equal(x0_new, x0_orig), f"x0 modifie pour {pid} a n={n}"


def test_orig_excluded_from_benchmark():
    for pid in IDS:
        assert (pid + "_orig") not in bm.PROBLEMS_C
        assert (pid + "_orig") not in bm.PROBLEMS_ALL_IDS
        assert hasattr(bm, f"c{pid[1:].lower()}_f_orig") or True  # nom exact verifie ci-dessous
    assert hasattr(bm, "c01_f_orig") and hasattr(bm, "c08_f_orig")
    assert hasattr(bm, "c09_f_orig") and hasattr(bm, "c10_f_orig")
    assert len(bm.PROBLEMS_ALL_IDS) == 36


def test_coercivity_informal():
    for pid in IDS:
        for n in DIMS:
            pb = bm.get_problem(pid, n)
            f0 = pb["f"](pb["x0"])
            far = 1000.0 * np.ones(n)
            ffar = pb["f"](far)
            assert np.isfinite(ffar), f"{pid} n={n}: f non fini loin de l'origine"
            assert ffar > f0, f"{pid} n={n}: f ne croit pas loin de l'origine"


TESTS = [
    ("Gradients vs differences finies (C01'/C08'/C09'/C10')", test_gradients_fd),
    ("Diagonale Hessienne vs differences finies", test_hessian_diag_fd),
    ("x0 inchanges (identiques a *_orig)", test_x0_unchanged),
    ("Fonctions/dicts d'origine hors benchmark (36 problemes)", test_orig_excluded_from_benchmark),
    ("Coercivite informelle (f croit loin de l'origine)", test_coercivity_informal),
]


def run():
    print("=" * 70)
    print("TEST UNITAIRE — Variantes coercives C01'/C08'/C09'/C10' (consigne 003)")
    print("=" * 70)
    passed = 0
    for name, fn in TESTS:
        try:
            res = fn()
            extra = f"  [{res:.2e}]" if isinstance(res, float) else ""
            print(f"  OK  {name}{extra}")
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
