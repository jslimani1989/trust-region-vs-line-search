# -*- coding: utf-8 -*-
"""
Consigne 009e (reprise de 008d-bis), etape 1 (front de Pareto) -- DECISION 4 : robustesse x pente log-log du TEMPS REEL PAR
ITERATION (mediane des repetitions du temps reel / n_iter) en fonction de n (n = 100, 500, 1000), sur les
problemes de Categorie A disponibles aux trois dimensions (11 problemes), TOUTES executions (reussies ou non :
le cout d'une iteration ne depend pas de la reussite). Valeur du solveur = mediane des pentes sur les
problemes, IC bootstrap par grappes (probleme), percentile, 10000 reechantillons, graine 20260927 ;
< 5 problemes -> non estimable. Analyse principale : problemes dont les trois temps sont >= 1 ms ;
sensibilite : tous les problemes (y compris "faible precision").
La pente du NOMBRE D'ITERATIONS (executions reussies aux trois dimensions) est rapportee a cote, a titre
DESCRIPTIF ; elle n'est pas utilisee pour le front.
Robustesse, regle de dominance certaine : comme en 008c/008d (taux de succes moyen par probleme, IC BCa,
tests de permutation Holm globaux : etape5_*).
"""
import csv
import json
import os

import numpy as np

from common_data import load_distinct_rows, METHODS
from common_timing import get_timing, LOW_PREC_S
from paths import OUT_DIR

SEED_BOOT_B4 = 20260927
B_BOOT = 10000
SCAL_DIMS = [100, 500, 1000]
MIN_PROBLEMS_FOR_SLOPE = 5


def slope(xs, ys):
    return float(np.polyfit(np.log10(xs), np.log10(ys), 1)[0])


def time_per_iter_slopes(rows, wall, exclude_low_precision):
    catA = [r for r in rows if r["category"] == "A"]
    idx = {(r["problem"], r["n"], r["method"]): r for r in catA}
    problems = sorted({r["problem"] for r in catA})
    out = {m: {} for m in METHODS}
    for m in METHODS:
        for p in problems:
            pts = []
            for n in SCAL_DIMS:
                r = idx.get((p, str(n), m))
                t = wall.get((p, str(n), m))
                if r is None or t is None or int(r["n_iter"]) < 1 or t <= 0:
                    break
                if exclude_low_precision and t < LOW_PREC_S:
                    break
                pts.append((n, t / int(r["n_iter"])))
            if len(pts) == len(SCAL_DIMS):
                out[m][p] = slope([a for a, _ in pts], [b for _, b in pts])
    return out


def iter_count_slopes(rows):
    catA = [r for r in rows if r["category"] == "A"]
    idx = {(r["problem"], r["n"], r["method"]): r for r in catA}
    problems = sorted({r["problem"] for r in catA})
    out = {m: {} for m in METHODS}
    for m in METHODS:
        for p in problems:
            pts = []
            for n in SCAL_DIMS:
                r = idx.get((p, str(n), m))
                if r is None or r["status"] != "success" or int(r["n_iter"]) < 1:
                    break
                pts.append((n, int(r["n_iter"])))
            if len(pts) == len(SCAL_DIMS):
                out[m][p] = slope([a for a, _ in pts], [b for _, b in pts])
    return out


def bootstrap_median_ci(values, rng, n_boot=B_BOOT):
    vals = np.array(list(values), dtype=float)
    if len(vals) < 2:
        return None, None
    idx = rng.integers(0, len(vals), size=(n_boot, len(vals)))
    lo, hi = np.percentile(np.median(vals[idx], axis=1), [2.5, 97.5])
    return float(lo), float(hi)


def summarize(slopes, rng, min_problems=MIN_PROBLEMS_FOR_SLOPE):
    res = {}
    for m in METHODS:
        k = len(slopes[m])
        if k < min_problems:
            res[m] = {"pente_mediane": None, "IC95_bas": None, "IC95_haut": None, "n_problemes": k, "estimable": False}
        else:
            med = float(np.median(list(slopes[m].values())))
            lo, hi = bootstrap_median_ci(slopes[m].values(), rng)
            res[m] = {"pente_mediane": med, "IC95_bas": lo, "IC95_haut": hi, "n_problemes": k, "estimable": True}
    return res


def load_robustness():
    with open(os.path.join(OUT_DIR, "etape5_2_IC_taux_succes.csv"), encoding="utf-8") as fh:
        rob = {r["method"]: {"taux": float(r["taux_succes_moyen_par_probleme"]), "IC_bas": float(r["IC95_bas"]),
                             "IC_haut": float(r["IC95_haut"])} for r in csv.DictReader(fh) if r["scope"] == "global"}
    with open(os.path.join(OUT_DIR, "etape5_1_tests_permutation.csv"), encoding="utf-8") as fh:
        perm_global = [r for r in csv.DictReader(fh) if r["famille"] == "global"]
    return rob, perm_global


def pareto_front(scal, rob, perm_global):
    def rob_sig(a, b):
        row = next(r for r in perm_global if {r["solveur_a"], r["solveur_b"]} == {a, b})
        return float(row["p_holm"]) < 0.05

    pareto = []
    for m in METHODS:
        if not scal[m]["estimable"]:
            continue
        dominated_by = []
        for m2 in METHODS:
            if m2 == m or not scal[m2]["estimable"]:
                continue
            ge_rob = rob[m2]["taux"] >= rob[m]["taux"]
            le_sc = scal[m2]["pente_mediane"] <= scal[m]["pente_mediane"]
            strict = rob[m2]["taux"] > rob[m]["taux"] or scal[m2]["pente_mediane"] < scal[m]["pente_mediane"]
            if ge_rob and le_sc and strict:
                disjoint = (scal[m2]["IC95_haut"] < scal[m]["IC95_bas"] or scal[m]["IC95_haut"] < scal[m2]["IC95_bas"])
                certain = rob_sig(m, m2) and disjoint
                dominated_by.append(f"{m2}({'certaine' if certain else 'provisoire'})")
        pareto.append({"method": m, "taux_succes": rob[m]["taux"], "IC95_bas_succes": rob[m]["IC_bas"],
                       "IC95_haut_succes": rob[m]["IC_haut"], "pente_temps_par_iteration": scal[m]["pente_mediane"],
                       "IC95_bas_pente": scal[m]["IC95_bas"], "IC95_haut_pente": scal[m]["IC95_haut"],
                       "n_problemes_pente": scal[m]["n_problemes"], "sur_le_front": len(dominated_by) == 0,
                       "domine_par": ";".join(dominated_by)})
    return pareto


def front_for_variant(rows, wall, rob, perm_global):
    """Front de Pareto (principale : problemes aux trois temps >= 1 ms) pour une variante de mesure."""
    s_main = time_per_iter_slopes(rows, wall, True)
    scal = summarize(s_main, np.random.default_rng(SEED_BOOT_B4))
    return scal, pareto_front(scal, rob, perm_global)


def _r(v, nd=4):
    return None if v is None else round(v, nd)


def main():
    rows = load_distinct_rows()
    wall = get_timing("principale").wall
    rob, perm_global = load_robustness()

    s_main = time_per_iter_slopes(rows, wall, True)
    s_all = time_per_iter_slopes(rows, wall, False)
    s_iter = iter_count_slopes(rows)
    scal = summarize(s_main, np.random.default_rng(SEED_BOOT_B4))
    scal_all = summarize(s_all, np.random.default_rng(SEED_BOOT_B4 + 1))
    scal_iter = summarize(s_iter, np.random.default_rng(SEED_BOOT_B4 + 2), min_problems=MIN_PROBLEMS_FOR_SLOPE)

    detail = []
    for tag, sl in (("temps_par_iteration_principale", s_main), ("temps_par_iteration_avec_faible_precision", s_all),
                    ("nombre_iterations_succes", s_iter)):
        for m in METHODS:
            for p, v in sl[m].items():
                detail.append({"axe": tag, "method": m, "problem": p, "pente_log_log": round(v, 4)})
    with open(os.path.join(OUT_DIR, "b4_scalabilite_detail.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["axe", "method", "problem", "pente_log_log"])
        w.writeheader()
        w.writerows(detail)

    pareto = pareto_front(scal, rob, perm_global)
    for r in pareto:  # pente du nombre d'iterations, a cote, a titre descriptif
        si = scal_iter[r["method"]]
        r["pente_nombre_iterations_descriptif"] = si["pente_mediane"]
        r["IC95_bas_pente_iter"] = si["IC95_bas"]
        r["IC95_haut_pente_iter"] = si["IC95_haut"]
        r["n_problemes_pente_iter"] = si["n_problemes"]
    with open(os.path.join(OUT_DIR, "b4_pareto_front.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(pareto[0].keys()))
        w.writeheader()
        for r in pareto:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else ("" if v is None else v)) for k, v in r.items()})

    def clean(d):
        return {m: {k: (_r(v) if isinstance(v, float) else v) for k, v in x.items()} for m, x in d.items()}
    meta = {"pente_temps_par_iteration_principale": clean(scal),
            "pente_temps_par_iteration_avec_faible_precision": clean(scal_all),
            "pente_nombre_iterations_succes_descriptif_min5": clean(scal_iter),
            "solveurs_exclus_pente_non_estimable": [m for m in METHODS if not scal[m]["estimable"]],
            "problemes_retenus_principale": sorted(set.intersection(*[set(s_main[m]) for m in METHODS]) if all(s_main[m] for m in METHODS) else []),
            "seuil_min_problemes": MIN_PROBLEMS_FOR_SLOPE, "seed_bootstrap": SEED_BOOT_B4, "B_bootstrap": B_BOOT}
    with open(os.path.join(OUT_DIR, "b4_meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, ensure_ascii=False)

    print("Pente du temps reel par iteration (principale ; avec faible precision ; nb d'iterations succes) :")
    for m in METHODS:
        a, b, c = scal[m], scal_all[m], scal_iter[m]
        f = lambda d: (f"{d['pente_mediane']:+.3f} [{d['IC95_bas']:+.3f},{d['IC95_haut']:+.3f}] k={d['n_problemes']}"
                       if d["estimable"] else f"non estimable k={d['n_problemes']}")
        print(f"  {m:13s} {f(a)} | {f(b)} | {f(c)}")
    for r in pareto:
        tag = "SUR LE FRONT" if r["sur_le_front"] else f"domine par {r['domine_par']}"
        print(f"  {r['method']:13s} succes={r['taux_succes']:.4f} pente={r['pente_temps_par_iteration']:+.3f} -> {tag}")


if __name__ == "__main__":
    main()
