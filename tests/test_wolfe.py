# -*- coding: utf-8 -*-
"""
test_wolfe.py
------------------------------------------------------------
TEST UNITAIRE — Recherche lineaire de Wolfe, correction de l'intervalle
d'encadrement degenere a alpha_max (consigne 008b, etape 1a/1c).

Contexte (rapport 008a) : les 12 exceptions ZeroDivisionError de BFGS-Wolfe
dans v0.2 provenaient toutes de _cubic_interp appelee avec un intervalle
[a,b] deja reduit a un point (a=b=alpha_max), transmis par _zoom lorsque la
phase d'expansion de strong_wolfe reevaluait deux fois de suite au meme
point plafonne a alpha_max sans jamais former d'encadrement. Corrige dans
cette version (008b/1a) selon Nocedal & Wright, algorithme 3.5 : le pas
n'est plus jamais reevalue au meme point plafonne ; un garde-fou
supplementaire protege aussi _cubic_interp elle-meme.

  1) Les 3 instances de 008a (C10 n=10, B5_Griewank n=10, B5_Griewank n=50)
     s'executent desormais sans exception, avec alpha_max_reached>0.
  2) Un cas synthetique (f(x)=-x, descente illimitee, courbure jamais
     satisfaite) atteint alpha_max deux fois de suite et ne produit plus
     d'intervalle degenere (aucune exception, alpha_max_reached>=1).
  3) Un cas non degenere (A01, n=10, BFGS-Wolfe) donne le meme resultat que
     celui deja enregistre dans resultats_benchmark_v0.2.csv -- preuve que la
     correction n'affecte aucun cas non degenere. Propriete d'origine validee
     par le rapport 008b : egalite au bit pres (n_iter, f_final, grad_norm, status).

     009d (v0.5) : BFGS-Wolfe est un solveur MODIFIE (mise a jour BFGS en O(n^2),
     ordre des operations change : les iteres ne sont plus identiques au bit
     pres a ceux de v0.2). Regle de comparaison a un resultat d'une version
     anterieure pour un solveur modifie : statut et n_iter identiques,
     |ecart de f_final| <= 1e-10 * max(1, |f_ref|), grad_norm <= 1e-6 des deux
     cotes en cas de convergence. Les assertions sur alpha_max_reached et
     interp_guard_used (aucune activation du mecanisme degenere) sont inchangees.
     Voir rapports/009d_rapport_corrections_algebre_v0.5.md.
"""

import sys
import os
import csv

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import implementation_methode_lineaires as ls
import framework_united_final as fw

# Fichier de reference d'une version anterieure, non distribue dans ce depot : le test (iii) est ignore (SKIP) s'il
# est absent. Chemin relatif a la racine du depot.
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V02_REF_CSV = os.path.join(_REPO, "results", "reference", "resultats_benchmark_v0.2.csv")


class SkipTest(Exception):
    """Fichier de reference absent de l'arbre publie : le test est ignore (SKIP), pas compte comme echec."""

INSTANCES_008A = [("C10", 10), ("B5_Griewank", 10), ("B5_Griewank", 50)]


def test_008a_instances_no_exception():
    for pid, n in INSTANCES_008A:
        pb = fw._get_problem(pid, n)
        f_counted, g_counted, h_counted, counts = fw._make_counted_oracles(pb)
        hist = fw._call_solver(fw.LS_BFGS_WOLFE, f_counted, g_counted, pb["x0"].copy())
        assert hist["alpha_max_reached"] > 0, (
            f"{pid} n={n} : attendu alpha_max_reached>0 (le mecanisme degenere "
            f"doit etre exerce sur ces 3 instances, cf. rapport 008a)")
        assert np.isfinite(hist["grad_norm"][-1]), f"{pid} n={n} : grad_norm final non fini"


def test_synthetic_alpha_max_twice_no_degenerate_interval():
    # f(x) = -x : descente illimitee, gradient constant -1, direction dk=+1
    # (Hk=I initial). Armijo toujours satisfaite (phi decroit lineairement,
    # plus vite que la borne c1*alpha*dphi0 puisque c1<1) ; courbure jamais
    # satisfaite (|dphi|=1 > -c2*dphi0=0.9 pour tout alpha). L'expansion
    # double donc alpha jusqu'a alpha_max=10.0 puis, SANS la correction,
    # replanterait au 2e passage au plafond.
    f = lambda x: -x[0]
    grad_f = lambda x: np.array([-1.0])
    xk = np.array([0.0])
    dk = np.array([1.0])

    counters = {"alpha_max_reached": 0, "interp_guard_used": 0}
    alpha, wolfe_ok, hit_nf = ls.strong_wolfe(f, grad_f, xk, dk, alpha_max=10.0, counters=counters)

    assert np.isfinite(alpha), "alpha doit rester fini (plus d'exception attendue)"
    assert alpha == 10.0, f"attendu alpha=alpha_max=10.0 (Armijo satisfaite au plafond), obtenu {alpha}"
    assert counters["alpha_max_reached"] >= 1, "le mecanisme degenere doit avoir ete detecte au moins une fois"
    assert wolfe_ok is False, "la courbure n'est pas verifiee au plafond : wolfe_ok doit rester False"


def test_non_degenerate_case_bit_identical_to_v0_2():
    # Appel direct (bfgs_wolfe), pas via framework_united_final.run_one : les
    # champs alpha_max_reached/interp_guard_used (008b/1a.3) sont dans le
    # history renvoye par le solveur, pas (encore) dans les FIELDS de
    # run_one -- run_one lui-meme est inchange (aucun autre solveur ni le
    # framework partage n'est touche par cette consigne, cf. rapport 008b S3).
    if not os.path.exists(V02_REF_CSV):
        raise SkipTest(f"fichier de reference absent : {os.path.relpath(V02_REF_CSV, _REPO)}")
    with open(V02_REF_CSV, encoding="utf-8") as fh:
        ref = {(r["problem"], r["n"], r["method"]): r for r in csv.DictReader(fh)}
    ref_row = ref[("A01", "10", "BFGS_Wolfe")]
    assert ref_row["status"] != "exception", "cas de reference attendu non degenere"

    pb = fw._get_problem("A01", 10)
    f_counted, g_counted, h_counted, counts = fw._make_counted_oracles(pb)
    hist = fw._call_solver(fw.LS_BFGS_WOLFE, f_counted, g_counted, pb["x0"].copy())
    x_final = hist["x_opt"]
    f_final = float(pb["f"](x_final))
    grad_norm = float(np.linalg.norm(pb["grad_f"](x_final)))
    status = "success" if hist["status"] in ("converged", "success") else "max_iter_or_not_converged"

    # 009d : regle des solveurs modifies (voir docstring) ; ancienne version : egalite au bit pres (rapport 008b).
    assert status == ref_row["status"]
    assert hist["iterations"] == int(ref_row["n_iter"])
    f_ref = float(ref_row["f_final"])
    assert abs(f_final - f_ref) <= 1e-10 * max(1.0, abs(f_ref)), (f_final, ref_row["f_final"])
    if status == "success":
        assert grad_norm <= 1e-6 and float(ref_row["grad_norm"]) <= 1e-6, (grad_norm, ref_row["grad_norm"])
    assert hist["alpha_max_reached"] == 0, "cas non degenere : le mecanisme ne doit jamais s'activer"
    assert hist["interp_guard_used"] == 0, "cas non degenere : le garde-fou de _cubic_interp ne doit jamais s'activer"


TESTS = [
    ("(i) Les 3 instances de 008a s'executent sans exception", test_008a_instances_no_exception),
    ("(ii) Cas synthetique alpha_max x2 : plus d'intervalle degenere", test_synthetic_alpha_max_twice_no_degenerate_interval),
    ("(iii) Cas non degenere (A01,n=10) : meme statut/n_iter que v0.2, f_final a 1e-10 pres (009d)", test_non_degenerate_case_bit_identical_to_v0_2),
]


def run():
    print("=" * 70)
    print("TEST UNITAIRE — Recherche lineaire de Wolfe, correction alpha_max (consigne 008b)")
    print("=" * 70)
    passed = 0
    skipped = 0
    for name, fn in TESTS:
        try:
            fn()
            print(f"  OK  {name}")
            passed += 1
        except SkipTest as e:
            print(f"  SKIP  {name}  [{e}]")
            skipped += 1
        except AssertionError as e:
            print(f"  FAIL  {name}  [{e}]")
        except Exception as e:
            print(f"  ERROR  {name}  [{type(e).__name__}: {e}]")
    print("-" * 70)
    print(f"  Resultat : {passed}/{len(TESTS)} tests passes, {skipped} ignores (fichier de reference absent).")
    print("=" * 70)
    return passed + skipped == len(TESTS)


if __name__ == "__main__":
    ok = run()
    sys.exit(0 if ok else 1)
