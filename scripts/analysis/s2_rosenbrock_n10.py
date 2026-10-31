# -*- coding: utf-8 -*-
"""
Consigne 009e, priorite 2 : table S2 (Rosenbrock B1, n = 10) depuis resultats_benchmark_v0.5.csv, avec les valeurs v0.3 et
v0.4 (lecture seule de analyse_v0.3/etape4_S2_rosenbrock_n10_v0.3.csv et run_v0.4/etape7_S2_rosenbrock_n10_v0.4.csv).
Les six solveurs sont ecrits dans le CSV ; le tableau du manuscrit en retient trois (Steihaug-CG, BFGS-Wolfe, Newton-LM).
"""
import csv
import os

from paths import BENCH_CSV, BENCH_V04_CSV, OUT_DIR, REFERENCE_DIR

S2_V03 = os.path.join(REFERENCE_DIR, "etape4_S2_rosenbrock_n10_v0.3.csv")
S2_V04 = os.path.join(REFERENCE_DIR, "etape7_S2_rosenbrock_n10_v0.4.csv")
METHODS = ["Cauchy_Point", "Dogleg", "Steihaug_CG", "BFGS_Armijo", "BFGS_Wolfe", "Newton_BT"]


def read(path):
    with open(path, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def pick(rows, method):
    return next(r for r in rows if r["problem"] == "B1_Rosenbrock" and r["n"] == "10" and r["method"] == method)


def main():
    b5, b4 = read(BENCH_CSV), read(BENCH_V04_CSV)
    s3 = {r["method"]: r for r in read(S2_V03)}
    s4 = {r["method"]: r for r in read(S2_V04)}
    out = []
    for m in METHODS:
        r5, r4 = pick(b5, m), pick(b4, m)
        row = {"method": m, "status_v0.5": r5["status"], "n_iter_v0.5": r5["n_iter"], "N_tot_v0.5": r5["n_total"],
               "grad_norm_final_v0.5": r5["grad_norm"], "n_f_evals_v0.5": r5["n_f_evals"], "n_g_evals_v0.5": r5["n_g_evals"],
               "n_h_evals_v0.5": r5["n_h_evals"],
               "code_version_ligne": r5["code_version"], "status_v0.4": r4["status"], "n_iter_v0.4": r4["n_iter"],
               "N_tot_v0.4": r4["n_total"],
               "n_iter_v0.3": s3[m]["n_iter"] if m in s3 else "", "N_tot_v0.3": s3[m]["n_total_Ntot"] if m in s3 else "",
               "N_tot_v0.4_reference_etape7": s4[m]["n_total_Ntot_v04"] if m in s4 else "",
               "n_iter_change_v04_v05": r5["n_iter"] != r4["n_iter"], "N_tot_change_v04_v05": r5["n_total"] != r4["n_total"]}
        out.append(row)
        # controle : la ligne v0.4 recoupe la reference de 009 pour les trois solveurs de S2
        if m in s4:
            assert s4[m]["n_iter"] == r4["n_iter"] and s4[m]["n_total_Ntot_v04"] == r4["n_total"], m
    with open(os.path.join(OUT_DIR, "s2_rosenbrock_n10_v0.5.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)
    for r in out:
        print(r)


if __name__ == "__main__":
    main()
