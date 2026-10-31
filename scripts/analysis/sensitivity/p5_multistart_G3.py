# -*- coding: utf-8 -*-
"""
Consigne 009e, priorite 5.

5.1 Departs multiples (results_multistart_v0.5_complet.csv, 2772 lignes) :
    - proportions de succes par solveur, categorie et rayon ; N_tot median (executions reussies) ;
    - tau de Kendall (SANS p-valeur) : rang par taux de succes au depart nominal contre chaque rayon de perturbation,
      global (moyenne non ponderee des taux par categorie, comme en 008b) et par categorie ;
    - part des departs de Categorie C a Hessienne indefinie (lambda_min_x0 < 0), taux sur tous les departs et sur les
      departs indefinis ;
    - comparaison avec v0.4 (results_multistart_v0.4.csv) : taux, N_tot median, tau, part indefinie.
5.2 G3 (results_G3_v0.5_complet.csv) : chiffres cites dans le manuscrit (l. 570-629) recalcules, compares a v0.3
    (results_G3_v0.3.csv), et nombre de tirages qui atteignent le plafond K_max = 5000 sans converger.
La figure G3 est regeneree par plot_G3.py (code du tag v0.5, inchange) depuis le fichier complet : voir p5_figure_G3.ps1/log.
Aucune execution de solveur.
"""
import csv
import json
import os
import statistics as st
from collections import defaultdict

from scipy.stats import kendalltau

from paths import MS_COMPLET_CSV, MS_V04_CSV, G3_COMPLET_CSV, G3_V03_CSV, OUT_DIR, keep, EXCLUDED

METHODS = ["Cauchy_Point", "Dogleg", "Steihaug_CG", "BFGS_Armijo", "BFGS_Wolfe", "Newton_BT"]
RADII = ["nominal", "sigma=0.1", "sigma=1.0"]


def read(path):
    with open(path, encoding="utf-8") as fh:
        return [r for r in csv.DictReader(fh) if keep(r)]  # 012


def to_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def proportions(rows):
    out = {}
    for cat in ("A", "B", "C"):
        for radius in RADII:
            for m in METHODS:
                sub = [r for r in rows if r["category"] == cat and r["radius"] == radius and r["method"] == m]
                if not sub:
                    continue
                succ = [r for r in sub if r["status"] == "success"]
                ntots = [int(r["n_total"]) for r in succ]
                out[(cat, radius, m)] = {"n": len(sub), "n_succes": len(succ), "taux": len(succ) / len(sub),
                                         "ntot_med": st.median(ntots) if ntots else None}
    return out


def kendall(props):
    out = []
    for scope, cat in [("global", None), ("Cat.A", "A"), ("Cat.B", "B"), ("Cat.C", "C")]:
        rates = defaultdict(dict)
        for (c, radius, m), p in props.items():
            if cat is None or c == cat:
                rates[radius].setdefault(m, []).append(p["taux"])
        rates = {rad: {m: st.mean(v) for m, v in d.items()} for rad, d in rates.items()}
        for radius in ("sigma=0.1", "sigma=1.0"):
            x = [rates["nominal"][m] for m in METHODS]
            y = [rates[radius][m] for m in METHODS]
            tau = kendalltau(x, y)[0]
            out.append({"scope": scope, "radius": radius, "kendall_tau": tau, "n_methodes": len(METHODS)})
    return out


def indefinite(rows):
    catC = [r for r in rows if r["category"] == "C"]
    starts = {}
    for r in catC:
        starts[(r["problem"], r["n"], r["radius"], r["draw"])] = to_float(r["lambda_min_x0"])
    n_ind = sum(1 for v in starts.values() if v < 0)
    res = {"departs_distincts": len(starts), "n_indefinis": n_ind, "proportion": n_ind / len(starts)}
    rates = {}
    for m in METHODS:
        for scope, filt in (("tous_departs", lambda k: True), ("departs_indefinis", lambda k: starts[k] < 0)):
            sub = [r for r in catC if r["method"] == m and filt((r["problem"], r["n"], r["radius"], r["draw"]))]
            ns = sum(r["status"] == "success" for r in sub)
            rates[(m, scope)] = {"n": len(sub), "n_succes": ns, "taux": ns / len(sub)}
    return res, rates


def g3_figures(rows):
    cells = defaultdict(list)
    for r in rows:
        cells[(r["panel"], float(r["kappa"]), int(r["n"]), r["method"])].append(r)
    med = {k: st.median(int(r["iter_outer"]) for r in v) for k, v in cells.items()}
    nonconv = {k: sum(1 for r in v if r["converged"] not in ("True", "true", "1")) for k, v in cells.items()}
    cap = {k: sum(1 for r in v if int(r["iter_outer"]) >= 5000 and r["converged"] not in ("True", "true", "1")) for k, v in cells.items()}
    d = {}
    for m in ("cauchy", "dogleg", "steihaug", "bfgs_armijo", "bfgs_wolfe", "newton_bt"):
        a, b = med.get(("B", 1e4, 10, m)), med.get(("B", 1e4, 1000, m))
        d[f"B_{m}_n10"], d[f"B_{m}_n1000"] = a, b
        d[f"B_{m}_facteur_n10_a_n1000"] = (b / a) if a and b else None
        lo, hi = med.get(("A", 1e2, 100, m)), med.get(("A", 1e5, 100, m))
        d[f"A_{m}_kappa1e2"], d[f"A_{m}_kappa1e5"] = lo, hi
        d[f"A_{m}_facteur_kappa1e2_a_1e5"] = (hi / lo) if lo and hi else None
    d["A_rapport_BFGSWolfe_sur_Steihaug_kappa1e5"] = d["A_bfgs_wolfe_kappa1e5"] / d["A_steihaug_kappa1e5"]
    return d, med, nonconv, cap


def main():
    ms5, ms4 = read(MS_COMPLET_CSV), read(MS_V04_CSV)
    assert EXCLUDED or (len(ms5) == 2772 and len(ms4) == 2772)  # 012
    p5, p4 = proportions(ms5), proportions(ms4)
    # 5.1 proportions
    out = []
    for k in sorted(p5, key=lambda k: (k[0], RADII.index(k[1]), METHODS.index(k[2]))):
        a, b = p4[k], p5[k]
        out.append({"category": k[0], "radius": k[1], "method": k[2], "n": b["n"], "n_succes_v0.5": b["n_succes"],
                    "taux_v0.5": round(b["taux"], 4), "n_succes_v0.4": a["n_succes"], "taux_v0.4": round(a["taux"], 4),
                    "ecart_n_succes": b["n_succes"] - a["n_succes"],
                    "Ntot_median_succes_v0.5": b["ntot_med"], "Ntot_median_succes_v0.4": a["ntot_med"],
                    "ratio_Ntot_v0.4_sur_v0.5": (round(a["ntot_med"] / b["ntot_med"], 4) if a["ntot_med"] and b["ntot_med"] else "")})
    with open(os.path.join(OUT_DIR, "p5_departs_multiples_proportions.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)
    # globale par solveur (tous departs, toutes categories)
    glob = {}
    for m in METHODS:
        for tag, rows in (("v0.5", ms5), ("v0.4", ms4)):
            sub = [r for r in rows if r["method"] == m]
            glob[(m, tag)] = (sum(r["status"] == "success" for r in sub), len(sub))
    with open(os.path.join(OUT_DIR, "p5_departs_multiples_global.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["method", "n_departs", "n_succes_v0.5", "taux_v0.5", "n_succes_v0.4", "taux_v0.4"])
        for m in METHODS:
            s5, n5 = glob[(m, "v0.5")]
            s4, n4 = glob[(m, "v0.4")]
            w.writerow([m, n5, s5, round(s5 / n5, 4), s4, round(s4 / n4, 4)])
    # Kendall
    k5, k4 = kendall(p5), kendall(p4)
    with open(os.path.join(OUT_DIR, "p5_departs_multiples_kendall.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["scope", "radius", "kendall_tau_v0.5", "kendall_tau_v0.4", "ecart"])
        for a, b in zip(k5, k4):
            w.writerow([a["scope"], a["radius"], round(a["kendall_tau"], 4), round(b["kendall_tau"], 4),
                        round(a["kendall_tau"] - b["kendall_tau"], 4)])
    # Categorie C indefinie
    i5, r5 = indefinite(ms5)
    i4, r4 = indefinite(ms4)
    with open(os.path.join(OUT_DIR, "p5_departs_multiples_categorieC_indefinie.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["method", "sous_ensemble", "n", "n_succes_v0.5", "taux_v0.5", "n_succes_v0.4", "taux_v0.4"])
        for m in METHODS:
            for sc in ("tous_departs", "departs_indefinis"):
                a, b = r5[(m, sc)], r4[(m, sc)]
                w.writerow([m, sc, a["n"], a["n_succes"], round(a["taux"], 4), b["n_succes"], round(b["taux"], 4)])
    # 5.2 G3
    g5, g3v03 = read(G3_COMPLET_CSV), read(G3_V03_CSV)
    assert len(g5) == 1080
    d5, med5, nc5, cap5 = g3_figures(g5)
    d3, med3, nc3, cap3 = g3_figures(g3v03)
    rows_g3 = []
    for k in d5:
        a, b = d3[k], d5[k]
        rows_g3.append({"grandeur": k, "v0.3": "" if a is None else round(a, 4), "v0.5_complet": "" if b is None else round(b, 4),
                        "inchange": (a == b) or (a is not None and b is not None and abs(a - b) < 1e-12)})
    with open(os.path.join(OUT_DIR, "p5_G3_chiffres.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["grandeur", "v0.3", "v0.5_complet", "inchange"])
        w.writeheader()
        w.writerows(rows_g3)
    cells_changed = [k for k in med5 if med5[k] != med3[k]]
    with open(os.path.join(OUT_DIR, "p5_G3_cellules.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["panel", "kappa", "n", "method", "n_graines", "mediane_iter_v0.3", "mediane_iter_v0.5", "non_converges_v0.3",
                    "non_converges_v0.5", "plafond_atteint_v0.5"])
        for k in sorted(med5):
            n_seeds = sum(1 for r in g5 if (r["panel"], float(r["kappa"]), int(r["n"]), r["method"]) == k)
            w.writerow([k[0], f"{k[1]:g}", k[2], k[3], n_seeds, med3[k], med5[k], nc3[k], nc5[k], cap5[k]])
    # iter_outer et statut converged identiques par ligne (v0.3 vs v0.5 complet) ?
    idx3 = {(r["panel"], r["method"], r["kappa"], r["n"], r["seed"]): r for r in g3v03}
    diff_iter = sum(1 for r in g5 if idx3[(r["panel"], r["method"], r["kappa"], r["n"], r["seed"])]["iter_outer"] != r["iter_outer"])
    diff_conv = sum(1 for r in g5 if idx3[(r["panel"], r["method"], r["kappa"], r["n"], r["seed"])]["converged"] != r["converged"])
    summary = {
        "departs_multiples": {
            "lignes": len(ms5), "statuts_changes_vs_v0.4": sum(1 for a, b in zip(ms4, ms5) if a["status"] != b["status"]),
            "kendall_v0.5": [(r["scope"], r["radius"], round(r["kendall_tau"], 4)) for r in k5],
            "categorieC_indefinie_v0.5": i5, "categorieC_indefinie_v0.4": i4,
            "taux_globaux": {m: {"v0.5": round(glob[(m, 'v0.5')][0] / glob[(m, 'v0.5')][1], 4),
                                 "v0.4": round(glob[(m, 'v0.4')][0] / glob[(m, 'v0.4')][1], 4)} for m in METHODS}},
        "G3": {"lignes": len(g5), "lignes_iter_outer_differentes_vs_v0.3": diff_iter, "lignes_converged_differentes_vs_v0.3": diff_conv,
               "cellules_dont_la_mediane_change": [list(map(str, k)) for k in cells_changed],
               "chiffres_cites": {k: d5[k] for k in ("B_bfgs_wolfe_n10", "B_bfgs_wolfe_n1000", "B_bfgs_wolfe_facteur_n10_a_n1000",
                                                   "B_steihaug_n10", "B_steihaug_n1000", "B_steihaug_facteur_n10_a_n1000",
                                                   "A_steihaug_facteur_kappa1e2_a_1e5", "A_bfgs_wolfe_facteur_kappa1e2_a_1e5",
                                                   "A_rapport_BFGSWolfe_sur_Steihaug_kappa1e5")},
               "chiffres_cites_inchanges_vs_v0.3": all(r["inchange"] for r in rows_g3),
               "tirages_au_plafond_non_converges_v0.5": sum(cap5.values()), "tirages_au_plafond_non_converges_v0.3": sum(cap3.values())}}
    with open(os.path.join(OUT_DIR, "p5_synthese.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
