# -*- coding: utf-8 -*-
"""
Consigne 009e, priorite 1 (suite de etape5_statistiques.py) :

 (a) table des taux de succes en PROPORTION DES 156 INSTANCES DISTINCTES (decision 6), par solveur et par categorie
     (ordre 1 : stationnaire_ordre1 = grad_norm <= 1e-6 ; il coincide avec status == "success" sur les 1080 lignes, et
     avec stationnaire_ordre2 : verifie ici) ; avec les valeurs v0.4 et les ecarts ;
 (b) comparaison avec v0.4 (analyse_v0.4/008d_bis, lecture seule) : taux qui changent (moyenne par probleme et proportion
     des 156 instances), paires dont la conclusion (significative ou non, seuil 0,05 sur le p ajuste par Holm) change :
     tests de permutation (familles globale, A, B, C), McNemar exact, et rapports de cout N_tot (l'IC exclut-il 1 ?).

Les tests, intervalles et le front restent en moyenne par probleme (36 grappes) ; seule la table (a) est en proportion
des instances. Aucune execution de solveur.
"""
import csv
import json
import os

from paths import BENCH_CSV, BENCH_V04_CSV, OUT_DIR, DIR_008D_BIS

METHODS = ["Cauchy_Point", "Dogleg", "Steihaug_CG", "BFGS_Armijo", "BFGS_Wolfe", "Newton_BT"]
CATS = ["A", "B", "C"]
ALPHA = 0.05


def read(path):
    with open(path, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def distinct(rows):
    idx = {}
    for r in rows:
        idx.setdefault((r["problem"], r["n"], r["method"]), r)
    return idx


def proportions(rows):
    """{(methode, portee): (n_succes, n_instances)} sur les instances distinctes ; portee in A, B, C, global."""
    idx = distinct(rows)
    out = {}
    for (p, n, m), r in idx.items():
        for scope in (r["category"], "global"):
            s, t = out.get((m, scope), (0, 0))
            out[(m, scope)] = (s + (r["status"] == "success"), t + 1)
    return out


def main():
    rows5, rows4 = read(BENCH_CSV), read(BENCH_V04_CSV)
    # ordre 1 == status success == ordre 2 (controle)
    ctl = {"lignes_v0.5": len(rows5),
           "ordre1_egal_status_success": all((r["stationnaire_ordre1"] == "True") == (r["status"] == "success") for r in rows5),
           "ordre1_egal_ordre2": all(r["stationnaire_ordre1"] == r["stationnaire_ordre2"] for r in rows5),
           "instances_distinctes": len({(r["problem"], r["n"]) for r in rows5})}
    assert ctl["instances_distinctes"] == 156 and ctl["ordre1_egal_status_success"] and ctl["ordre1_egal_ordre2"], ctl
    p5, p4 = proportions(rows5), proportions(rows4)
    per_cat_n = {sc: p5[("Cauchy_Point", sc)][1] for sc in CATS + ["global"]}
    ctl["instances_par_portee"] = per_cat_n

    # (a) table 156 instances
    table = []
    for m in METHODS:
        for sc in CATS + ["global"]:
            s5, t5 = p5[(m, sc)]
            s4, t4 = p4[(m, sc)]
            assert t5 == t4
            table.append({"solveur": m, "portee": sc, "n_instances": t5, "n_succes_v0.5": s5,
                          "proportion_v0.5": round(s5 / t5, 4), "n_succes_v0.4": s4, "proportion_v0.4": round(s4 / t4, 4),
                          "ecart_n_succes": s5 - s4, "ecart_proportion": round((s5 - s4) / t5, 4)})
    with open(os.path.join(OUT_DIR, "p1_table_succes_156_instances.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(table[0].keys()))
        w.writeheader()
        w.writerows(table)

    # (b) comparaison avec v0.4
    comp = []

    def load_new(f):
        return read(os.path.join(OUT_DIR, f))

    def load_old(f):
        return read(os.path.join(DIR_008D_BIS, f))

    # taux moyens par probleme (36 grappes)
    new_r = {(r["method"], r["scope"]): r for r in load_new("etape5_2_IC_taux_succes.csv")}
    old_r = {(r["method"], r["scope"]): r for r in load_old("etape5_2_IC_taux_succes.csv")}
    for k in sorted(new_r, key=lambda k: (METHODS.index(k[0]), ["global", "A", "B", "C"].index(k[1]))):
        a, b = float(old_r[k]["taux_succes_moyen_par_probleme"]), float(new_r[k]["taux_succes_moyen_par_probleme"])
        comp.append({"famille": "taux moyen par probleme", "objet": f"{k[0]} | {k[1]}", "v0.4": a, "v0.5": b,
                     "ecart": round(b - a, 4), "conclusion_v0.4": "", "conclusion_v0.5": "", "conclusion_change": ""})
    # permutation
    new_p = {(r["famille"], r["solveur_a"], r["solveur_b"]): r for r in load_new("etape5_1_tests_permutation.csv")}
    old_p = {(r["famille"], r["solveur_a"], r["solveur_b"]): r for r in load_old("etape5_1_tests_permutation.csv")}
    changed = []
    for k in new_p:
        po, pn = float(old_p[k]["p_holm"]), float(new_p[k]["p_holm"])
        co, cn = ("significatif" if po < ALPHA else "non significatif"), ("significatif" if pn < ALPHA else "non significatif")
        comp.append({"famille": "permutation (p Holm)", "objet": f"{k[0]} | {k[1]} vs {k[2]}", "v0.4": round(po, 5), "v0.5": round(pn, 5),
                     "ecart": round(pn - po, 5), "conclusion_v0.4": co, "conclusion_v0.5": cn, "conclusion_change": co != cn})
        if co != cn:
            changed.append(("permutation", k, po, pn, co, cn))
    # McNemar
    new_m = {(r["solveur_a"], r["solveur_b"]): r for r in load_new("etape5_4_mcnemar.csv")}
    old_m = {(r["solveur_a"], r["solveur_b"]): r for r in load_old("etape5_4_mcnemar.csv")}
    for k in new_m:
        po, pn = float(old_m[k]["p_holm"]), float(new_m[k]["p_holm"])
        co, cn = ("significatif" if po < ALPHA else "non significatif"), ("significatif" if pn < ALPHA else "non significatif")
        comp.append({"famille": "McNemar (p Holm, 156 instances)", "objet": f"{k[0]} vs {k[1]}", "v0.4": round(po, 5), "v0.5": round(pn, 5),
                     "ecart": round(pn - po, 5), "conclusion_v0.4": co, "conclusion_v0.5": cn, "conclusion_change": co != cn})
        if co != cn:
            changed.append(("McNemar", k, po, pn, co, cn))
    # couts N_tot : l'IC exclut-il 1 ?
    new_c = {(r["scope"], r["solveur_a"], r["solveur_b"]): r for r in load_new("etape5_3_couts.csv")}
    old_c = {(r["scope"], r["solveur_a"], r["solveur_b"]): r for r in load_old("etape5_3_couts.csv")}

    def excl(r):
        if r["ratio_geo_moyen"] in ("", None):
            return "non determinable"
        lo, hi = float(r["IC95_bas"]), float(r["IC95_haut"])
        return "IC exclut 1" if (hi < 1 or lo > 1) else "IC contient 1"
    for k in new_c:
        co, cn = excl(old_c[k]), excl(new_c[k])
        ro, rn = old_c[k]["ratio_geo_moyen"], new_c[k]["ratio_geo_moyen"]
        comp.append({"famille": "cout N_tot (rapport geometrique)", "objet": f"{k[0]} | {k[1]} / {k[2]}", "v0.4": ro, "v0.5": rn,
                     "ecart": "" if ro in ("", None) or rn in ("", None) else round(float(rn) - float(ro), 4),
                     "conclusion_v0.4": co, "conclusion_v0.5": cn, "conclusion_change": co != cn})
        if co != cn:
            changed.append(("cout", k, ro, rn, co, cn))
    with open(os.path.join(OUT_DIR, "p1_comparaison_v04_v05.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(comp[0].keys()))
        w.writeheader()
        w.writerows(comp)

    summary = {"controles": ctl,
               "taux_156_instances_qui_changent": [{"solveur": t["solveur"], "portee": t["portee"], "v0.4": t["proportion_v0.4"],
                                                    "v0.5": t["proportion_v0.5"]} for t in table if t["ecart_n_succes"] != 0],
               "taux_moyens_par_probleme_qui_changent": [(c["objet"], c["v0.4"], c["v0.5"]) for c in comp
                                                         if c["famille"] == "taux moyen par probleme" and c["ecart"] != 0],
               "conclusions_qui_changent": [{"famille": f, "objet": "|".join(k) if isinstance(k, tuple) else k,
                                             "v0.4": a, "v0.5": b, "conclusion_v0.4": c, "conclusion_v0.5": d}
                                            for f, k, a, b, c, d in changed],
               "n_comparaisons_conclusion": {f: sum(1 for c in comp if c["famille"] == f)
                                             for f in ("permutation (p Holm)", "McNemar (p Holm, 156 instances)", "cout N_tot (rapport geometrique)")}}
    with open(os.path.join(OUT_DIR, "p1_synthese.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print()
    for sc in CATS + ["global"]:
        print(sc, f"({per_cat_n[sc]} instances)", " ".join(f"{m}={p5[(m, sc)][0]}/{p5[(m, sc)][1]}" for m in METHODS))


if __name__ == "__main__":
    main()
