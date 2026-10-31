#!/usr/bin/env python3
"""
run_multistart.py -- Departs multiples (consigne 008b, etape 1f / Execution C, R2.7).

Sous-ensemble : 4 problemes par categorie (A, B, C), tires au hasard (graine
notee) parmi les 15/11/10 problemes de chaque categorie ; un probleme a
dimension fixe (A06, A08, A11, A13, C02, C04) est utilise a sa seule
dimension.

Dimensions : n=10 et n=100 pour les problemes non a dimension fixe (sinon
les deux plus petites dimensions disponibles, cf. adjust_n) ; une seule
dimension pour les problemes a dimension fixe.

Departs : pour chaque (probleme, n), le depart NOMINAL x0 (1 execution),
puis pour sigma in {0.1, 1} : 10 tirages x0^(k) = x0 + sigma * max(1,
||x0||_inf) * xi_k, xi_k ~ N(0, I_n).

6 solveurs, Kmax=1000, tolerance 1e-6 (identiques au benchmark principal,
AUCUNE mesure de temps utilisee/enregistree, cf. contraintes 008b). Pour la
Categorie C, lambda_min(hess_f(x0^(k))) est enregistre au depart (avant
toute execution de solveur).

Graine unique (MASTER_SEED, notee ci-dessous) pilotant, dans cet ordre fixe
et deterministe, TOUTE l'alea du script : (1) le tirage des 4x3=12
problemes, (2) les tirages de perturbation xi_k, cellule par cellule, dans
l'ordre categorie -> probleme -> dimension -> sigma -> k. Reproductible
tel quel (aucun seed derive/calcule a la volee).

Sortie : results_multistart_v0.3.csv.
"""
import argparse
import csv
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

MASTER_SEED = 20260927  # graine unique, notee (date d'execution de la consigne 008b-suite)
N_PER_CATEGORY = 4
SIGMAS = [0.1, 1.0]
N_DRAWS = 10
KMAX = 1000
TOL = 1e-6
FIXED_DIM_PROBLEMS = {"A06", "A08", "A11", "A13", "C02", "C04"}
TARGET_DIMS = [10, 100]

METHODS = ["Cauchy_Point", "Dogleg", "Steihaug_CG", "BFGS_Armijo", "BFGS_Wolfe", "Newton_BT"]

CSV_FIELDS = ["category", "problem", "n", "radius", "draw", "method",
              "status", "converged", "n_iter", "n_f_evals", "n_g_evals", "n_h_evals",
              "n_total", "f_final", "grad_norm", "lambda_min_x0"]


def select_problems(fw, rng):
    """4 problemes par categorie (A, B, C), tires sans remise."""
    cats = {
        "A": list(fw.PROBLEMS_A_IDS),
        "B": list(fw.PROBLEMS_B_IDS),
        "C": [p for p in fw.PROBLEMS_ALL_IDS if p not in fw.PROBLEMS_A_IDS and p not in fw.PROBLEMS_B_IDS],
    }
    selected = {}
    for cat, ids in cats.items():
        idx = rng.choice(len(ids), size=N_PER_CATEGORY, replace=False)
        selected[cat] = sorted(ids[i] for i in idx)
    return selected


def run_instance(fw, pb, method, x0):
    f_counted, g_counted, h_counted, counts = fw._make_counted_oracles(pb)
    pb_local = dict(pb)
    pb_local["x0"] = x0
    # run_one recalcule x0 depuis pb -- on l'appelle directement avec le x0 perturbe
    row = fw.run_one(pb_local, method)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results_multistart_v0.3.csv")
    args = ap.parse_args()

    import framework_united_final as fw
    fw.KMAX = KMAX

    rng = np.random.default_rng(MASTER_SEED)
    selected = select_problems(fw, rng)
    print(f"=== run_multistart (graine maitresse={MASTER_SEED}) ===")
    for cat in ("A", "B", "C"):
        print(f"  Categorie {cat} : {selected[cat]}")

    out_rows = []
    n_total_runs = 0
    for cat in ("A", "B", "C"):
        for pid in selected[cat]:
            # Essaie les deux dimensions cibles (10 et 100) et deduplique par n
            # REEL obtenu : pour un probleme a dimension fixe (adjust_n interne
            # a _mod_all.get_problem), les deux requetes resolvent a la meme
            # unique dimension et ne comptent qu'une fois -- pas de cas special.
            pbs_by_n = {}
            for n_req in TARGET_DIMS:
                pb_try = fw._get_problem(pid, n_req)
                pbs_by_n[pb_try["n"]] = pb_try
            is_fixed = pid in FIXED_DIM_PROBLEMS
            assert is_fixed == (len(pbs_by_n) == 1), (pid, is_fixed, sorted(pbs_by_n))

            for n_actual, pb0 in sorted(pbs_by_n.items()):
                x0_nominal = np.asarray(pb0["x0"], dtype=float)
                scale = max(1.0, float(np.max(np.abs(x0_nominal))))

                # Point nominal (radius="nominal", 1 seul tirage = pas de perturbation)
                starts = [("nominal", 0, x0_nominal.copy())]
                for sigma in SIGMAS:
                    for k in range(N_DRAWS):
                        xi = rng.standard_normal(n_actual)
                        x0_k = x0_nominal + sigma * scale * xi
                        starts.append((f"sigma={sigma}", k, x0_k))

                for radius_label, draw_idx, x0_pert in starts:
                    lam_min_x0 = None
                    if cat == "C":
                        H0 = pb0["hess_f"](x0_pert)
                        lam_min_x0 = float(np.linalg.eigvalsh(0.5 * (H0 + H0.T)).min())
                    for method in METHODS:
                        row = run_instance(fw, pb0, method, x0_pert)
                        out_rows.append({
                            "category": cat, "problem": pid, "n": n_actual,
                            "radius": radius_label, "draw": draw_idx, "method": method,
                            "status": row["status"], "converged": row["converged"],
                            "n_iter": row["n_iter"], "n_f_evals": row["n_f_evals"],
                            "n_g_evals": row["n_g_evals"], "n_h_evals": row["n_h_evals"],
                            "n_total": row["n_total"], "f_final": row["f_final"],
                            "grad_norm": row["grad_norm"], "lambda_min_x0": lam_min_x0,
                        })
                        n_total_runs += 1
                print(f"  {pid} n={n_actual} : {len(starts)} departs x 6 solveurs "
                      f"= {len(starts) * 6} lignes (cumul={n_total_runs})")

    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(out_rows)
    print(f"--- {len(out_rows)} lignes ecrites dans '{args.out}' ---")


if __name__ == "__main__":
    main()
