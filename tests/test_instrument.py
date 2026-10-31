# -*- coding: utf-8 -*-
"""
test_instrument.py
------------------------------------------------------------
TEST UNITAIRE -- Option instrument=False (consigne 008e, Etape 1.3)

Verifie que desactiver l'instrumentation (decomposition spectrale par
iteration dans newton_backtracking ; classification Q9 en fin d'execution
dans run_one) ne change AUCUN des iteres : sur 12 instances (2 par solveur,
graine notee, dont au moins 4 a n>=500), instrument=False et instrument=True
donnent le meme statut, n_iter, N_tot, f_final et grad_norm, au bit pres.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import framework_united_final as fw

SEED = 20260929  # graine notee (20260928 ne donne que 3/12 instances a n>=500, essai suivant)
METHODS = ["Cauchy_Point", "Dogleg", "Steihaug_CG", "BFGS_Armijo", "BFGS_Wolfe", "Newton_BT"]
N_PER_METHOD = 2
MIN_LARGE_N = 4  # au moins 4 des 12 instances a n>=500


def _select_instances():
    specs = fw.build_problem_specs(dims=[10, 50, 100, 500, 1000], categories=["A", "B", "C"])
    distinct = sorted(set(specs))
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(distinct))
    chosen = {}
    pos = 0
    for m in METHODS:
        picks = []
        while len(picks) < N_PER_METHOD:
            pid, n = distinct[idx[pos]]
            pos += 1
            picks.append((pid, n))
        chosen[m] = picks
    n_large = sum(1 for m in METHODS for (_, n) in chosen[m] if n >= 500)
    assert n_large >= MIN_LARGE_N, (
        f"seulement {n_large}/12 instances a n>=500 avec la graine {SEED} (minimum {MIN_LARGE_N} requis) -- "
        "choisir une autre graine et noter le changement dans le rapport"
    )
    return chosen


def test_instrument_false_matches_instrument_true():
    chosen = _select_instances()
    n_compared = 0
    for method in METHODS:
        for (pid, n) in chosen[method]:
            pb = fw._get_problem(pid, n)
            row_true = fw.run_one(pb, method, instrument=True)
            pb2 = fw._get_problem(pid, n)  # instance fraiche (aucun etat partage entre les deux appels)
            row_false = fw.run_one(pb2, method, instrument=False)
            n_compared += 1

            ctx = f"{method} {pid} n={n}"
            assert row_true["status"] == row_false["status"], f"{ctx} : statut differe"
            assert row_true["n_iter"] == row_false["n_iter"], f"{ctx} : n_iter differe"
            assert row_true["n_total"] == row_false["n_total"], f"{ctx} : N_tot differe"
            assert row_true["f_final"] == row_false["f_final"] or (
                np.isnan(row_true["f_final"]) and np.isnan(row_false["f_final"])
            ), f"{ctx} : f_final differe ({row_true['f_final']} vs {row_false['f_final']})"
            assert row_true["grad_norm"] == row_false["grad_norm"] or (
                np.isnan(row_true["grad_norm"]) and np.isnan(row_false["grad_norm"])
            ), f"{ctx} : grad_norm differe ({row_true['grad_norm']} vs {row_false['grad_norm']})"

            # instrument=False : colonnes Q9 laissees a NaN (desactivees), jamais utilisees pour le statut
            assert np.isnan(row_false["lambda_min_hess_final"])
            assert np.isnan(row_false["hess_spectral_norm_final"])
    assert n_compared == len(METHODS) * N_PER_METHOD


TESTS = [
    ("instrument=False == instrument=True (statut/n_iter/N_tot/f_final/grad_norm, 12 instances)",
     test_instrument_false_matches_instrument_true),
]


def run():
    print("=" * 70)
    print("TEST UNITAIRE -- Option instrument=False (consigne 008e)")
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
