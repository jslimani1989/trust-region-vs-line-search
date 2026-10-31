# -*- coding: utf-8 -*-
"""
test_memoize.py
------------------------------------------------------------
TEST UNITAIRE -- Option memoize (consigne 009, Etape 3)

1) memoize=False restitue EXACTEMENT le comptage de v0.3 (N_f,N_g,N_h,N_tot),
   sur 30 instances tirees au hasard (graine notee, les six solveurs
   representes).
2) memoize=True contre memoize=False, memes instances + C06|10|BFGS_Wolfe
   (cas extreme de 008f) : statut, n_iter, f_final, grad_norm identiques au
   bit pres.
3) Accord avec la mesure de 008f : reexecute avec memoize=True toutes les
   executions de l'echantillon 008f a n<=100 ; N_f/N_g/N_h doivent egaler les
   decomptes "points distincts" calcules depuis les enregistrements bruts.
4) Securite de la memoire : mutation en place d'une valeur renvoyee par un
   appel memorise n'affecte pas un appel ulterieur au meme point.

IMPORTANT : ce test rejoue des instances a l'echelle du benchmark (jusqu'a
n=1000, 1000 iterations), contrairement aux autres tests unitaires (problemes
jouets de petite taille) -- BLAS/OpenBLAS multi-thread n'est PAS reproductible
au bit pres d'une execution a l'autre (ordre de reduction variable selon
l'ordonnancement des threads) ; a cette echelle, cela peut faire diverger la
trajectoire exacte d'une methode non convergente (budget K_max) d'une
execution a l'autre. D'ou l'epinglage a un seul thread AVANT l'import de
NumPy, comme dans tous les harnais de chronometrage/observation du projet
(008c/008e/008f) -- necessaire ici pour la meme raison (reproductibilite au
bit pres), meme si ce fichier ne mesure aucun temps.
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import csv
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import framework_united_final as fw

# Fichiers de reference d'une version anterieure, non distribues dans ce depot : les tests qui les lisent sont
# ignores (SKIP) s'ils sont absents. Chemins relatifs a la racine du depot.
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF_CSV = os.path.join(_REPO, "results", "reference", "resultats_benchmark_v0.3.csv")
DIAG_DIR = os.path.join(_REPO, "results", "reference", "diagnostics_008f")


class SkipTest(Exception):
    """Fichier de reference absent de l'arbre publie : le test est ignore (SKIP), pas compte comme echec."""
SEED = 20260929  # graine notee (meme convention que 008c/008e/008f)
N_SAMPLE = 30

# ── 009d (v0.5) : comparaison a des resultats d'une version anterieure ───────────────────────────────────────
# Cauchy_Point et Steihaug_CG (code inchange) : egalite au bit pres maintenue (propriete d'origine, consigne 009).
# BFGS_Armijo, BFGS_Wolfe, Newton_BT, Dogleg (code modifie en v0.5, ordre des operations change) : statut et n_iter
# identiques, |ecart de f_final| <= 1e-10 * max(1, |f_ref|), grad_norm <= 1e-6 des deux cotes en cas de convergence.
# Si un statut ou un n_iter differe sur une instance : la valeur de reference est remplacee par celle de v0.5 dans
# tests/refs_v05_overrides.csv et consignee dans rapports/009d_rapport_corrections_algebre_v0.5.md (instance,
# ancienne et nouvelle valeur), sans assouplir davantage. Les tests internes a une meme version (memoize True/False,
# securite de la memoire) restent a egalite exacte.
MODIFIED_V05 = {"BFGS_Armijo", "BFGS_Wolfe", "Newton_BT", "Dogleg"}
OVERRIDES_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "refs_v05_overrides.csv")


def _load_overrides():
    out = {}
    if os.path.isfile(OVERRIDES_CSV):
        with open(OVERRIDES_CSV, encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                out[(r["problem"], r["n"], r["method"])] = r
    return out


def _assert_modified_vs_ref(ctx, key, row, ref):
    """Regle des solveurs modifies (009d). ref : ligne de reference (statut, n_iter, f_final, grad_norm)."""
    ov = _load_overrides().get(key)
    src = ov if ov else ref
    assert row["status"] == src["status"], (
        f"{ctx} : statut {src['status']} -> {row['status']} (a declarer dans tests/refs_v05_overrides.csv et au rapport 009d)")
    assert str(row["n_iter"]) == str(src["n_iter"]), (
        f"{ctx} : n_iter {src['n_iter']} -> {row['n_iter']} (a declarer dans tests/refs_v05_overrides.csv et au rapport 009d)")
    f_ref, f_new = float(src["f_final"]), float(row["f_final"])
    assert (np.isnan(f_ref) and np.isnan(f_new)) or abs(f_new - f_ref) <= 1e-10 * max(1.0, abs(f_ref)), (
        f"{ctx} : f_final {f_ref} -> {f_new}")
    if row["status"] == "success":
        assert float(row["grad_norm"]) <= 1e-6 and float(src["grad_norm"]) <= 1e-6, (
            f"{ctx} : grad_norm {src['grad_norm']} -> {row['grad_norm']}")


def _load_ref():
    if not os.path.exists(REF_CSV):
        raise SkipTest(f"fichier de reference absent : {os.path.relpath(REF_CSV, _REPO)}")
    with open(REF_CSV, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    idx = {}
    for r in rows:
        idx.setdefault((r["problem"], r["n"], r["method"]), r)
    return idx


def _sample_30_all_methods(ref_idx, seed=SEED, n=N_SAMPLE):
    by_method = {}
    for (pid, ndim, method), row in ref_idx.items():
        by_method.setdefault(method, []).append((pid, ndim, method))
    rng = random.Random(seed)
    methods = sorted(by_method.keys())
    assert len(methods) == 6, methods
    sample = []
    for m in methods:  # au moins une instance par solveur
        sample.append(rng.choice(by_method[m]))
    all_keys = list(ref_idx.keys())
    rng.shuffle(all_keys)
    for k in all_keys:
        if len(sample) >= n:
            break
        if k not in sample:
            sample.append(k)
    return sample[:n]


def test_memoize_false_matches_v03():
    ref_idx = _load_ref()
    sample = _sample_30_all_methods(ref_idx)
    devs = []
    for (pid, ndim, method) in sample:
        ref = ref_idx[(pid, ndim, method)]
        pb = fw._get_problem(pid, int(ndim))
        row = fw.run_one(pb, method, instrument=True, memoize=False)
        ctx = f"{pid} n={ndim} {method}"
        if method in MODIFIED_V05:  # 009d : solveur modifie -> regle de tolerance (voir en-tete du fichier)
            try:
                _assert_modified_vs_ref(ctx, (pid, ndim, method), row, ref)
            except AssertionError as e:  # on collecte TOUS les ecarts (un seul passage du test long)
                devs.append(str(e))
            continue
        assert str(row["n_f_evals"]) == ref["n_f_evals"], f"{ctx} : n_f_evals {row['n_f_evals']} vs {ref['n_f_evals']}"
        assert str(row["n_g_evals"]) == ref["n_g_evals"], f"{ctx} : n_g_evals"
        assert str(row["n_h_evals"]) == ref["n_h_evals"], f"{ctx} : n_h_evals"
        assert str(row["n_total"]) == ref["n_total"], f"{ctx} : n_total"
        assert row["status"] == ref["status"], f"{ctx} : status"
        assert str(row["n_iter"]) == ref["n_iter"], f"{ctx} : n_iter"
    assert not devs, " || ".join(devs)
    return sample


def test_iterates_unchanged_memoize_true_vs_false():
    ref_idx = _load_ref()
    sample = _sample_30_all_methods(ref_idx)
    cases = list(sample) + [("C06", "10", "BFGS_Wolfe")]
    for (pid, ndim, method) in cases:
        pb1 = fw._get_problem(pid, int(ndim))
        row_t = fw.run_one(pb1, method, instrument=True, memoize=True)
        pb2 = fw._get_problem(pid, int(ndim))
        row_f = fw.run_one(pb2, method, instrument=True, memoize=False)
        ctx = f"{pid} n={ndim} {method}"
        assert row_t["status"] == row_f["status"], f"{ctx} : statut differe"
        assert row_t["n_iter"] == row_f["n_iter"], f"{ctx} : n_iter differe"
        ft, ff = row_t["f_final"], row_f["f_final"]
        assert ft == ff or (np.isnan(ft) and np.isnan(ff)), f"{ctx} : f_final differe ({ft} vs {ff})"
        gt, gf = row_t["grad_norm"], row_f["grad_norm"]
        assert gt == gf or (np.isnan(gt) and np.isnan(gf)), f"{ctx} : grad_norm differe ({gt} vs {gf})"
        # N_tot(memoize) ne doit jamais depasser N_tot(sans memoire)
        assert row_t["n_total"] <= row_f["n_total"], f"{ctx} : N_tot(memoize) > N_tot(sans memoire)"


def _distinct_counts_current_version(pid, ndim, method):
    """Nombre de points DISTINCTS evalues (f, grad_f, hess_f) par une execution du solveur dans la version courante,
    sans memoire, enregistres derriere la couche de comptage (009d)."""
    pb = fw._get_problem(pid, ndim)
    seen = {"f": set(), "g": set(), "h": set()}

    def rec(kind, fn):
        def wrapped(x):
            seen[kind].add(fw._memo_fingerprint(x))
            return fn(x)
        return wrapped

    pb_rec = dict(pb, f=rec("f", pb["f"]), grad_f=rec("g", pb["grad_f"]), hess_f=rec("h", pb["hess_f"]))
    f_c, g_c, h_c, _ = fw._make_counted_oracles(pb_rec, memoize=False)
    kind, fn = fw.METHODS[method]
    if kind == "LS1":
        fw._run_ls_first_order(fn, f_c, g_c, pb["x0"], instrument=True)
    elif kind == "LS2":
        fw._run_ls_second_order(fn, f_c, g_c, h_c, pb["x0"], instrument=True)
    else:
        fw._run_tr(fn, f_c, g_c, h_c, pb["x0"], instrument=True)
    return {k: len(v) for k, v in seen.items()}


def test_agreement_with_008f_sample_n_le_100():
    if not os.path.exists(DIAG_DIR):
        raise SkipTest(f"fichier de reference absent : {os.path.relpath(DIAG_DIR, _REPO)}")
    import pickle
    exec_csv = os.path.join(DIAG_DIR, "observe_oracles_executions.csv")
    with open(exec_csv, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    rows = [r for r in rows if r["status"] not in ("NON_MESUREE_TIMEOUT", "ERROR")
            and r.get("records_file") and int(r["n"]) <= 100]
    assert len(rows) > 0
    n_checked = 0
    devs_008f = []
    for r in rows:
        path = os.path.join(DIAG_DIR, "records", r["records_file"])
        with open(path, "rb") as fh:
            records = pickle.load(fh)
        distinct = {"f": set(), "g": set(), "h": set()}
        for kind, fp, dt in records:
            distinct[kind].add(fp)

        pb = fw._get_problem(r["problem"], int(r["n"]))
        row = fw.run_one(pb, r["method"], instrument=True, memoize=True)
        ctx = f"{r['problem']} n={r['n']} {r['method']}"
        if r["method"] in MODIFIED_V05:
            # 009d : les enregistrements de 008f viennent de trajectoires de l'ancien code ; pour un solveur modifie :
            # (a) statut et n_iter compares a ceux de 008f (regle de tolerance, surcharges declarees), (b) la propriete
            # d'origine (N_f/N_g/N_h = points distincts) verifiee DANS la version courante, sur la meme execution.
            ov = _load_overrides().get((r["problem"], r["n"], r["method"]))
            src = ov if ov else r
            if row["status"] != src["status"] or str(row["n_iter"]) != str(src["n_iter"]):
                devs_008f.append(f"{ctx} : statut {src['status']} -> {row['status']}, n_iter {src['n_iter']} -> {row['n_iter']} "
                                 f"(a declarer dans tests/refs_v05_overrides.csv et au rapport 009d)")
            cur = _distinct_counts_current_version(r["problem"], int(r["n"]), r["method"])
            assert row["n_f_evals"] == cur["f"], f"{ctx} : n_f_evals={row['n_f_evals']} vs distinct (version courante)={cur['f']}"
            assert row["n_g_evals"] == cur["g"], f"{ctx} : n_g_evals vs distinct (version courante)"
            assert row["n_h_evals"] == cur["h"], f"{ctx} : n_h_evals vs distinct (version courante)"
            n_checked += 1
            continue
        assert row["n_f_evals"] == len(distinct["f"]), f"{ctx} : n_f_evals={row['n_f_evals']} vs distinct={len(distinct['f'])}"
        assert row["n_g_evals"] == len(distinct["g"]), f"{ctx} : n_g_evals"
        assert row["n_h_evals"] == len(distinct["h"]), f"{ctx} : n_h_evals"
        n_checked += 1
    print(f"    ({n_checked} executions 008f a n<=100 verifiees)")
    assert not devs_008f, " || ".join(devs_008f)


def test_memory_safety_mutation():
    pb = fw._get_problem("A12", 100)
    f_c, g_c, h_c, counts = fw._make_counted_oracles(pb, memoize=True, k_fg=4, k_h=2)
    x = np.zeros(100)

    v1 = g_c(x)
    v1_expected = v1.copy()
    v1 += 12345.0  # mutation en place du tableau RENVOYE par l'appel non memorise
    v2 = g_c(x)    # appel memorise au meme point
    assert np.array_equal(v2, v1_expected), "la mutation de la valeur renvoyee a corrompu la memoire"
    assert counts["g"] == 1, "le second appel (memorise) n'aurait pas du etre compte"

    v3 = g_c(x)
    v2 += 999.0  # mutation du resultat d'un appel memorise
    v4 = g_c(x)
    assert np.array_equal(v4, v1_expected), "la mutation d'une valeur memorisee renvoyee a corrompu la memoire"

    H1 = h_c(x)
    H1_expected = H1.copy()
    H1 += 777.0
    H2 = h_c(x)
    assert np.array_equal(H2, H1_expected), "mutation d'une Hessienne renvoyee a corrompu la memoire"


TESTS = [
    ("memoize=False restitue le comptage v0.3 (30 instances, 6 solveurs)", test_memoize_false_matches_v03),
    ("iteres inchanges memoize=True vs False (30 instances + C06|10|BFGS_Wolfe)", test_iterates_unchanged_memoize_true_vs_false),
    ("accord avec 008f (n<=100, N_f/N_g/N_h = points distincts)", test_agreement_with_008f_sample_n_le_100),
    ("securite de la memoire (mutation en place)", test_memory_safety_mutation),
]


def run():
    print("=" * 70)
    print("TEST UNITAIRE -- Option memoize (consigne 009)")
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
