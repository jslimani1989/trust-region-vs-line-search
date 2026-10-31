# -*- coding: utf-8 -*-
"""
Consigne 009e, T-L1 : rapport de cout N_tot des 15 paires, recalcule par la procedure de 009
(revision/run_v0.4/scripts/etape7_couts_15paires.py : instances resolues par les deux solveurs, moyenne
geometrique par probleme puis sur les problemes, IC BCa par grappes (probleme), 10 000 reechantillons,
graine 20260927 consommee sequentiellement sur les 15 paires). Aucune p-valeur sur les couts.

Controle de methode : la meme procedure appliquee a resultats_benchmark_v0.4.csv doit reproduire les colonnes *_v04 de
revision/run_v0.4/etape7_15_paires_v03_v04.csv (ecart maximal < 1e-4) ; arret sinon.
Sorties : tl1_ntot_15paires_recalcule.csv (v0.5, colonnes *_v05 ; colonnes v0.4 recalculees a titre de comparaison),
          tl1_controle_methode_v04.json.
"""
import csv
import itertools
import json
import math
import os
import sys

import numpy as np
from scipy import stats as sps

import common_data
from etape5_statistiques import bca_ci, SEED_BOOT
from paths import OUT_DIR, BENCH_CSV, BENCH_V04_CSV, REFERENCE_DIR

PAIRES_015_CSV = os.path.join(REFERENCE_DIR, "etape7_15_paires_v03_v04.csv")
PAIRS = list(itertools.combinations(common_data.METHODS, 2))


def load_distinct(path):
    import csv as _csv
    with open(path, encoding="utf-8") as fh:
        rows = list(_csv.DictReader(fh))
    idx = {}
    for r in rows:
        idx.setdefault((r["problem"], r["n"], r["method"]), r)
    return list(idx.values())


def fifteen_pairs(rows):
    all_probs = sorted({r["problem"] for r in rows})
    assert len(all_probs) == 36
    ntot, succ = common_data.ntot_by_instance(rows)
    rng = np.random.default_rng(SEED_BOOT)
    out = []
    for a, b in PAIRS:
        gms, n_inst = [], 0
        for p in all_probs:
            ratios = [ntot[p][nk][a] / ntot[p][nk][b] for nk in ntot.get(p, {})
                      if a in succ[p][nk] and b in succ[p][nk] and succ[p][nk][a] and succ[p][nk][b]]
            if ratios:
                gms.append(sps.gmean(ratios))
                n_inst += len(ratios)
        theta_log, lo_log, hi_log = bca_ci(np.log(np.array(gms)), rng)
        out.append({"solveur_a": a, "solveur_b": b, "n_problemes_utilises": len(gms), "n_instances_utilisees": n_inst,
                    "ratio_geo_moyen": round(math.exp(theta_log), 4),
                    "IC95_bas": round(math.exp(lo_log), 4), "IC95_haut": round(math.exp(hi_log), 4)})
    return out


def main():
    r04 = fifteen_pairs(load_distinct(BENCH_V04_CSV))
    with open(PAIRES_015_CSV, encoding="utf-8") as fh:
        ref = {(r["solveur_a"], r["solveur_b"]): r for r in csv.DictReader(fh)}
    worst = 0.0
    for r in r04:
        x = ref[(r["solveur_a"], r["solveur_b"])]
        for k in ("ratio_geo_moyen", "IC95_bas", "IC95_haut"):
            worst = max(worst, abs(float(r[k]) - float(x[k + "_v04"])))
        assert int(x["n_problemes_utilises"]) == r["n_problemes_utilises"]
        assert int(x["n_instances_utilisees"]) == r["n_instances_utilisees"]
    with open(os.path.join(OUT_DIR, "tl1_controle_methode_v04.json"), "w", encoding="utf-8") as fh:
        json.dump({"ecart_max_vs_etape7_15_paires_v03_v04_colonnes_v04": worst, "tolerance": 1e-4,
                   "note": "procedure appliquee aux donnees v0.4 : doit reproduire la reference de 009"}, fh, indent=2)
    print(f"controle de methode (v0.4) : ecart maximal {worst:.2e}")
    if worst > 1e-4:
        print("*** ECART : la procedure ne reproduit pas la reference v0.4 ***")
        sys.exit(2)
    r05 = fifteen_pairs(load_distinct(BENCH_CSV))
    out = []
    for a, b in zip(r05, r04):
        out.append({"solveur_a": a["solveur_a"], "solveur_b": a["solveur_b"],
                    "n_problemes_utilises": a["n_problemes_utilises"], "n_instances_utilisees": a["n_instances_utilisees"],
                    "ratio_geo_moyen_v05": a["ratio_geo_moyen"], "IC95_bas_v05": a["IC95_bas"], "IC95_haut_v05": a["IC95_haut"],
                    "ratio_geo_moyen_v04": b["ratio_geo_moyen"], "IC95_bas_v04": b["IC95_bas"], "IC95_haut_v04": b["IC95_haut"],
                    "IC_v05_exclut_1": (a["IC95_haut"] < 1) or (a["IC95_bas"] > 1)})
    with open(os.path.join(OUT_DIR, "tl1_ntot_15paires_recalcule.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)
    for r in out:
        if {r["solveur_a"], r["solveur_b"]} == {"BFGS_Armijo", "BFGS_Wolfe"}:
            print(r)


if __name__ == "__main__":
    main()
