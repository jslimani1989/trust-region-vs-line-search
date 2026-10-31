# -*- coding: utf-8 -*-
"""
Consigne 009e, priorite 6 : sensibilite aux arrondis (formules mathematiquement equivalentes en v0.5).

Pour BFGS-Armijo et BFGS-Wolfe (mise a jour en O(n^2)), et pour Newton-LM et Dogleg (substitutions triangulaires) :
nombre d'instances (benchmark : 156 instances distinctes ; 180 creneaux) et de departs (departs multiples : 462) dont le
STATUT ou n_iter a change entre v0.4 et v0.5, sens des changements de statut, effet sur le taux de succes.
Cauchy Point et Steihaug-CG (code inchange) servent de controle : aucune difference attendue (lignes reprises de v0.4).
Entrees : resultats_benchmark_v0.4.csv / _v0.5.csv, results_multistart_v0.4.csv / results_multistart_v0.5_complet.csv.
Aucune execution de solveur.
"""
import csv
import json
import os
import statistics as st
from collections import defaultdict

from paths import BENCH_CSV, BENCH_V04_CSV, MS_V04_CSV, MS_COMPLET_CSV, OUT_DIR

SOLVERS = ["BFGS_Armijo", "BFGS_Wolfe", "Newton_BT", "Dogleg", "Cauchy_Point", "Steihaug_CG"]
NAME = {"BFGS_Armijo": "BFGS-Armijo", "BFGS_Wolfe": "BFGS-Wolfe", "Newton_BT": "Newton-LM", "Dogleg": "Dogleg",
        "Cauchy_Point": "Cauchy Point", "Steihaug_CG": "Steihaug-CG"}


def read(path):
    with open(path, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def key_bench(r):
    return (r["problem"], r["n"], r["method"])


def key_ms(r):
    return (r["category"], r["problem"], r["n"], r["radius"], r["draw"], r["method"])


def compare(old, new, keyf, label):
    """old, new : listes de lignes (memes creneaux). Retourne {solveur: stats}, details."""
    by_old, by_new = defaultdict(list), defaultdict(list)
    for r in old:
        by_old[keyf(r)].append(r)
    for r in new:
        by_new[keyf(r)].append(r)
    assert set(by_old) == set(by_new) and all(len(by_old[k]) == len(by_new[k]) for k in by_old)
    stats, details = {}, []
    for m in SOLVERS:
        keys = sorted(k for k in by_old if k[-1] == m)
        s = {"unites": len(keys), "statut_change": 0, "succes_vers_echec": 0, "echec_vers_succes": 0, "n_iter_change": 0,
             "n_iter_change_statut_inchange": 0, "statut_ou_n_iter_change": 0, "N_tot_change": 0,
             "succes_v0.4": 0, "succes_v0.5": 0, "rel_n_iter": []}
        for k in keys:
            a, b = by_old[k][0], by_new[k][0]  # creneaux repetes identiques entre eux (controle plus bas)
            sa, sb = a["status"] == "success", b["status"] == "success"
            s["succes_v0.4"] += sa
            s["succes_v0.5"] += sb
            ch_status, ch_iter, ch_ntot = sa != sb, a["n_iter"] != b["n_iter"], a["n_total"] != b["n_total"]
            s["statut_change"] += ch_status
            s["succes_vers_echec"] += (sa and not sb)
            s["echec_vers_succes"] += (sb and not sa)
            s["n_iter_change"] += ch_iter
            s["n_iter_change_statut_inchange"] += (ch_iter and not ch_status)
            s["statut_ou_n_iter_change"] += (ch_status or ch_iter)
            s["N_tot_change"] += ch_ntot
            if ch_iter and int(a["n_iter"]) > 0:
                s["rel_n_iter"].append((int(b["n_iter"]) - int(a["n_iter"])) / int(a["n_iter"]))
            if ch_status or ch_iter:
                details.append({"ensemble": label, "solveur": m, "cle": "|".join(k[:-1]), "statut_v0.4": a["status"],
                                "n_iter_v0.4": a["n_iter"], "statut_v0.5": b["status"], "n_iter_v0.5": b["n_iter"],
                                "N_tot_v0.4": a["n_total"], "N_tot_v0.5": b["n_total"],
                                "sens_statut": ("succes -> echec" if (sa and not sb) else ("echec -> succes" if (sb and not sa) else "inchange"))})
        stats[m] = s
    # controle : creneaux repetes identiques (benchmark)
    inconsistent = sum(1 for k, v in by_new.items() if len(v) > 1 and any(x["status"] != v[0]["status"] or x["n_iter"] != v[0]["n_iter"] for x in v))
    return stats, details, inconsistent


def main():
    b4, b5 = read(BENCH_V04_CSV), read(BENCH_CSV)
    m4, m5 = read(MS_V04_CSV), read(MS_COMPLET_CSV)
    assert len(b4) == len(b5) == 1080 and len(m4) == len(m5) == 2772
    sb, db, inc_b = compare(b4, b5, key_bench, "benchmark (156 instances distinctes)")
    sm, dm, inc_m = compare(m4, m5, key_ms, "departs multiples (462 departs)")
    # creneaux repetes du benchmark : nombre de creneaux (180) dont le statut ou n_iter change
    slots = {}
    for m in SOLVERS:
        c = 0
        i4 = [r for r in b4 if r["method"] == m]
        i5 = [r for r in b5 if r["method"] == m]
        assert len(i4) == len(i5) == 180
        for a, b in zip(i4, i5):  # memes creneaux dans le meme ordre (fusion_v0_5.py)
            assert (a["problem"], a["n"]) == (b["problem"], b["n"])
            c += (a["status"] != b["status"]) or (a["n_iter"] != b["n_iter"])
        slots[m] = c
    rows = []
    for ens, S, n_total in (("benchmark", sb, 156), ("departs multiples", sm, 462)):
        for m in SOLVERS:
            s = S[m]
            rel = s["rel_n_iter"]
            rows.append({"ensemble": ens, "solveur": NAME[m], "unites": s["unites"], "statut_change": s["statut_change"],
                         "succes_vers_echec": s["succes_vers_echec"], "echec_vers_succes": s["echec_vers_succes"],
                         "n_iter_change": s["n_iter_change"], "n_iter_change_statut_inchange": s["n_iter_change_statut_inchange"],
                         "statut_ou_n_iter_change": s["statut_ou_n_iter_change"],
                         "creneaux_180_statut_ou_n_iter_change": slots[m] if ens == "benchmark" else "",
                         "N_tot_change": s["N_tot_change"],
                         "mediane_ecart_relatif_n_iter": round(st.median(rel), 4) if rel else "",
                         "min_ecart_relatif_n_iter": round(min(rel), 4) if rel else "",
                         "max_ecart_relatif_n_iter": round(max(rel), 4) if rel else "",
                         "succes_v0.4": s["succes_v0.4"], "succes_v0.5": s["succes_v0.5"],
                         "taux_v0.4": round(s["succes_v0.4"] / s["unites"], 4), "taux_v0.5": round(s["succes_v0.5"] / s["unites"], 4),
                         "ecart_taux_points": round(100 * (s["succes_v0.5"] - s["succes_v0.4"]) / s["unites"], 2)})
    with open(os.path.join(OUT_DIR, "p6_sensibilite_BFGS.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(OUT_DIR, "p6_instances_modifiees_detail.csv"), "w", newline="", encoding="utf-8") as fh:
        det = db + dm
        w = csv.DictWriter(fh, fieldnames=list(det[0].keys()))
        w.writeheader()
        w.writerows(det)
    # liste des changements de statut (benchmark)
    flips = [d for d in db if d["sens_statut"] != "inchange"]
    summary = {"controle_creneaux_repetes_incoherents_v0.5": {"benchmark": inc_b, "departs_multiples": inc_m},
               "statuts_changes_benchmark": {NAME[m]: {"total": sb[m]["statut_change"], "succes_vers_echec": sb[m]["succes_vers_echec"],
                                                       "echec_vers_succes": sb[m]["echec_vers_succes"]} for m in SOLVERS},
               "statuts_changes_departs": {NAME[m]: {"total": sm[m]["statut_change"], "succes_vers_echec": sm[m]["succes_vers_echec"],
                                                     "echec_vers_succes": sm[m]["echec_vers_succes"]} for m in SOLVERS},
               "liste_statuts_changes_benchmark": [(d["solveur"], d["cle"], d["sens_statut"], d["n_iter_v0.4"], d["n_iter_v0.5"]) for d in flips]}
    with open(os.path.join(OUT_DIR, "p6_synthese.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    for r in rows:
        print(r)
    print(json.dumps(summary["liste_statuts_changes_benchmark"], indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
