# -*- coding: utf-8 -*-
"""
Consigne 009e (reprise de 008d-bis), etape 2 : validation hors echantillon, 36 plis (un probleme exclu a la fois ;
n* CONTINU reappris si le probleme exclu est de Categorie A -- decision 2, aucun arrondi ;
feuilles reapprises ; modele de temps t_iter recalcule SANS le probleme exclu).
Portee : branche Hessienne (candidats Cauchy_Point, Dogleg, Steihaug_CG, Newton_BT), comme en 008c/008d ;
verification complementaire de la feuille L1 (BFGS) conservee.

Quatre regles (decision 3) :
  - robustesse d'abord ("naive") : Newton_BT des que la Hessienne est disponible ;
  - temps d'abord ("un_noeud") : Newton_BT si n < n* (n* reappris dans chaque pli), Steihaug_CG sinon ;
  - arbre appris ("arbre") : regle d'apprentissage delta = 0,05 reapprise sans le probleme exclu ;
  - article : regle publiee (n >= 500 -> Steihaug_CG ; Cat. C originales a selle -> Steihaug_CG).
Mesures par probleme (moyenne / moyenne geometrique sur ses instances) :
  - perte de succes = (meilleur observe parmi les 4 candidats) - (succes de la regle) ;
  - taux d'accord ;
  - regret de temps = temps(regle) / temps(plus rapide parmi les candidats ayant reussi), instances ou la
    regle a reussi, toutes categories. Le regret n'est calculable que si le temps de la regle ET de TOUS les
    candidats ayant reussi est utilisable :
      * "mesure seul"   : temps mesures non "faible precision" uniquement ;
      * "mesure + estime" : temps mesure s'il existe, temps estime seulement si AUCUNE mesure n'existe (decision 5).
    (Une tache mesuree mais "faible precision" n'est jamais remplacee par une estimation : instance non retenue.)
Les intervalles (bootstrap par probleme, percentile, 10000 reechantillons, graine 20260927) sont des
informations complementaires pour la figure et les tableaux.
Supplement : regret sur le sous-ensemble COMMUN d'instances (regret defini pour les quatre regles), pour
comparer les regles sur les memes instances ; sensibilite "faible precision incluse".
"""
import csv
import json
import os

import numpy as np

from common_data import load_distinct_rows
from common_timing import get_timing
from tree_model_b2 import LEAVES, HESS, NO_HESS, leaf_for, build_leaf_instances
import math
from b1_seuil_noeud2 import n_star_point_estimate, n_star_effective
from b2_arbre_decision_temps import learn_leaf, index_rows, DELTA_MAIN
from time_model import load_timebook, timebook_excluding
from paths import OUT_DIR

ARTICLE_THRESHOLD_N = 500
SADDLE_ORIGINAL = {"C01", "C08", "C09", "C10"}
RULES = ("arbre", "naive", "article", "un_noeud")
SEED_BOOT = 20260927
B_BOOT = 10000


def article_rule(category, n, problem):
    if int(n) >= ARTICLE_THRESHOLD_N:
        return "Steihaug_CG"
    if category in ("A", "B"):
        return "Newton_BT"
    return "Steihaug_CG" if problem in SADDLE_ORIGINAL else "Newton_BT"


def regret(rows_idx, tb, p, n, candidates, recommended, measured_only, allow_lp=False):
    d = rows_idx.get((p, n), {})
    if recommended not in d or not d[recommended]["success"]:
        return None
    succ = [m for m in candidates if m in d and d[m]["success"]]
    times = {}
    for m in succ:
        t, src = tb.get(p, n, m, allow_estimate=not measured_only, allow_low_precision=allow_lp)
        if t is None or t <= 0:
            return None  # un candidat ayant reussi (ou la regle) n'a pas de temps utilisable : regret non calculable
        times[m] = t
    if recommended not in times:
        return None
    return times[recommended] / min(times.values())


def gmean(v):
    return float(np.exp(np.mean(np.log(v)))) if len(v) else None


def run_loo(timing, rows=None, rows_idx=None, allow_lp=False, with_ci=False, verbose=False):
    """Validation hors echantillon pour une variante de mesure. Retourne dict(inst_rows, fold_rows, per_problem,
    synth, skipped, n_star_full)."""
    rows = rows if rows is not None else load_distinct_rows()
    rows_idx = rows_idx if rows_idx is not None else index_rows(rows)
    wall = timing.wall
    problems = sorted({r["problem"] for r in rows})
    cat_of = {r["problem"]: r["category"] for r in rows}
    dims_of = {}
    for r in rows:
        dims_of.setdefault(r["problem"], set()).add(r["n"])

    n_star_full = n_star_effective(wall, rows=rows)
    tb_full, recs, va, vl, use = load_timebook(timing)

    inst_rows, fold_rows, skipped = [], [], []
    for p in problems:
        cat = cat_of[p]
        if cat == "A":
            n_star_loo = n_star_effective(wall, exclude_problem=p, rows=rows)
            if n_star_loo is None:
                skipped.append(p)
                fold_rows.append({"problem": p, "category": cat, "n_star_pli": "non determinable", "pli_ecarte": True})
                continue
            tb = timebook_excluding(p, use, timing)
        else:
            n_star_loo, tb = n_star_full, tb_full
        fold_rows.append({"problem": p, "category": cat,
                          "n_star_pli": round(n_star_loo, 4) if math.isfinite(n_star_loo) else "non determinable (seuil infini)",
                          "pli_ecarte": False})

        rows_wo_p = [r for r in rows if r["problem"] != p]
        leaf_data = build_leaf_instances(rows_wo_p, n_star_loo)
        learned = {leaf: learn_leaf(rows_idx, leaf_data[leaf]["instances"], leaf_data[leaf]["candidats"], tb,
                                    delta=DELTA_MAIN, rule="delta", compute_tests=False, allow_lp=allow_lp)
                   for leaf in LEAVES[1:]}
        for n in sorted(dims_of[p], key=int):
            _, leaf_hess = leaf_for(cat, n, n_star_loo)
            recs_rule = {"arbre": learned[leaf_hess]["recommande"], "naive": "Newton_BT",
                         "article": article_rule(cat, n, p),
                         "un_noeud": "Newton_BT" if int(n) < n_star_loo else "Steihaug_CG"}
            d = rows_idx[(p, n)]
            succ = {m: d[m]["success"] for m in HESS if m in d}
            best = 1 if any(succ.values()) else 0
            for rule, rec in recs_rule.items():
                s = int(d.get(rec, {}).get("success", False))
                inst_rows.append({
                    "problem": p, "category": cat, "n": n, "leaf": leaf_hess, "regle": rule, "solveur_recommande": rec,
                    "n_star_pli": round(n_star_loo, 4) if math.isfinite(n_star_loo) else "inf", "perte": best - s, "accord": int(s == best),
                    "regret_mesure_seul": regret(rows_idx, tb, p, n, HESS, rec, True, allow_lp),
                    "regret_mesure_estime": regret(rows_idx, tb, p, n, HESS, rec, False, allow_lp)})

    # sous-ensemble commun d'instances (regret defini pour les quatre regles), mesure seul
    by_inst = {}
    for r in inst_rows:
        by_inst.setdefault((r["problem"], r["n"]), {})[r["regle"]] = r
    common_keys = {k for k, v in by_inst.items() if len(v) == len(RULES)
                   and all(x["regret_mesure_seul"] is not None for x in v.values())}
    common_keys_est = {k for k, v in by_inst.items() if len(v) == len(RULES)
                       and all(x["regret_mesure_estime"] is not None for x in v.values())}

    per_problem = []
    for p in problems:
        if p in skipped:
            continue
        for rule in RULES:
            sub = [r for r in inst_rows if r["problem"] == p and r["regle"] == rule]
            rm = [r["regret_mesure_seul"] for r in sub if r["regret_mesure_seul"] is not None]
            re_ = [r["regret_mesure_estime"] for r in sub if r["regret_mesure_estime"] is not None]
            rc = [r["regret_mesure_seul"] for r in sub if (r["problem"], r["n"]) in common_keys]
            rce = [r["regret_mesure_estime"] for r in sub if (r["problem"], r["n"]) in common_keys_est]
            per_problem.append({
                "problem": p, "category": cat_of[p], "regle": rule, "n_instances": len(sub),
                "perte_moyenne": float(np.mean([r["perte"] for r in sub])),
                "taux_accord": float(np.mean([r["accord"] for r in sub])),
                "regret_mesure_seul": gmean(rm), "n_inst_regret_mesure_seul": len(rm),
                "regret_mesure_estime": gmean(re_), "n_inst_regret_mesure_estime": len(re_),
                "regret_mesure_seul_commun": gmean(rc), "n_inst_commun": len(rc),
                "regret_mesure_estime_commun": gmean(rce), "n_inst_commun_estime": len(rce)})

    rng = np.random.default_rng(SEED_BOOT)

    def boot_ci(vals, geometric):
        vals = np.array(vals, dtype=float)
        if len(vals) < 2:
            return None, None
        if geometric:
            vals = np.log(vals)
        idx = rng.integers(0, len(vals), size=(B_BOOT, len(vals)))
        bm = vals[idx].mean(axis=1)
        lo, hi = np.percentile(bm, [2.5, 97.5])
        return (float(np.exp(lo)), float(np.exp(hi))) if geometric else (float(lo), float(hi))

    synth = []
    for rule in RULES:
        for scope in ("toutes", "A", "B", "C"):
            sub = [r for r in per_problem if r["regle"] == rule and (scope == "toutes" or r["category"] == scope)]
            if not sub:
                continue
            pertes = [r["perte_moyenne"] for r in sub]
            acc = [r["taux_accord"] for r in sub]
            rm = [r["regret_mesure_seul"] for r in sub if r["regret_mesure_seul"] is not None]
            re_ = [r["regret_mesure_estime"] for r in sub if r["regret_mesure_estime"] is not None]
            rc = [r["regret_mesure_seul_commun"] for r in sub if r["regret_mesure_seul_commun"] is not None]
            rce = [r["regret_mesure_estime_commun"] for r in sub if r["regret_mesure_estime_commun"] is not None]
            row = {"regle": rule, "portee": scope, "n_problemes": len(sub),
                   "perte_moyenne": float(np.mean(pertes)), "taux_accord": float(np.mean(acc)),
                   "regret_mesure_seul_geo": gmean(rm), "n_pb_regret_mesure_seul": len(rm),
                   "n_inst_regret_mesure_seul": sum(r["n_inst_regret_mesure_seul"] for r in sub),
                   "regret_mesure_estime_geo": gmean(re_), "n_pb_regret_mesure_estime": len(re_),
                   "n_inst_regret_mesure_estime": sum(r["n_inst_regret_mesure_estime"] for r in sub),
                   "regret_mesure_seul_commun_geo": gmean(rc), "n_pb_commun": len(rc),
                   "n_inst_commun": sum(r["n_inst_commun"] for r in sub),
                   "regret_mesure_estime_commun_geo": gmean(rce), "n_pb_commun_estime": len(rce),
                   "n_inst_commun_estime": sum(r["n_inst_commun_estime"] for r in sub)}
            if scope == "toutes" and with_ci:
                lo, hi = boot_ci(pertes, False)
                row["perte_IC95"] = [lo, hi] if lo is not None else None
                for key, vals in (("regret_mesure_seul_IC95", rm), ("regret_mesure_estime_IC95", re_),
                                  ("regret_mesure_seul_commun_IC95", rc), ("regret_mesure_estime_commun_IC95", rce)):
                    lo, hi = boot_ci(vals, True) if vals else (None, None)
                    row[key] = [lo, hi] if lo is not None else None
            synth.append(row)
    return {"inst_rows": inst_rows, "fold_rows": fold_rows, "per_problem": per_problem, "synth": synth,
            "skipped": skipped, "n_star_full": n_star_full, "problems": problems, "cat_of": cat_of,
            "dims_of": dims_of, "tb_full": tb_full, "use": use, "n_common_inst": len(common_keys),
            "n_common_inst_estime": len(common_keys_est)}


def write_csv(path, rows_):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows_[0].keys()))
        w.writeheader()
        w.writerows(rows_)


def _round_row(r, nd=5):
    return {k: (round(v, nd) if isinstance(v, float) else (json.dumps([round(x, 4) for x in v]) if isinstance(v, list) else v))
            for k, v in r.items()}


def l1_check(rows, rows_idx, timing, tb_full, use, n_star_full, problems, cat_of, dims_of):
    l1 = []
    for p in problems:
        rows_wo_p = [r for r in rows if r["problem"] != p]
        tb = timebook_excluding(p, use, timing) if cat_of[p] == "A" else tb_full
        info = learn_leaf(rows_idx, build_leaf_instances(rows_wo_p, n_star_full)["L1_pas_hessienne"]["instances"],
                          NO_HESS, tb, delta=DELTA_MAIN, rule="delta", compute_tests=False)
        rec = info["recommande"]
        for n in sorted(dims_of[p], key=int):
            d = rows_idx[(p, n)]
            best = 1 if any(d[m]["success"] for m in NO_HESS if m in d) else 0
            s = int(d.get(rec, {}).get("success", False))
            l1.append({"problem": p, "category": cat_of[p], "n": n, "solveur_recommande": rec,
                       "perte": best - s, "accord": int(s == best)})
    return l1


def main():
    rows = load_distinct_rows()
    rows_idx = index_rows(rows)
    timing = get_timing("principale")
    res = run_loo(timing, rows, rows_idx, with_ci=True)
    write_csv(os.path.join(OUT_DIR, "b3_validation_LOO_instances.csv"),
              [_round_row(r) for r in res["inst_rows"]])
    write_csv(os.path.join(OUT_DIR, "b3_plis_nstar.csv"), res["fold_rows"])
    write_csv(os.path.join(OUT_DIR, "b3_validation_LOO_par_probleme.csv"), [_round_row(r) for r in res["per_problem"]])
    write_csv(os.path.join(OUT_DIR, "b3_synthese.csv"), [_round_row(r, 4) for r in res["synth"]])

    # sensibilite : taches "faible precision" utilisees comme mesures
    res_lp = run_loo(timing, rows, rows_idx, allow_lp=True, with_ci=False)
    sens = []
    for s in res_lp["synth"]:
        if s["portee"] == "toutes":
            sens.append(_round_row({k: v for k, v in s.items() if k != "portee"}, 4))
    write_csv(os.path.join(OUT_DIR, "b3_sensibilite_faible_precision_incluse.csv"), sens)

    l1 = l1_check(rows, rows_idx, timing, res["tb_full"], res["use"], res["n_star_full"], res["problems"],
                  res["cat_of"], res["dims_of"])
    write_csv(os.path.join(OUT_DIR, "b3_L1_seule.csv"), l1)

    for r in res["synth"]:
        if r["portee"] == "toutes":
            print(f"{r['regle']:9s} perte={r['perte_moyenne']:.4f} accord={r['taux_accord']:.4f} "
                  f"regret(mesure seul)={r['regret_mesure_seul_geo']:.4f} [{r['n_pb_regret_mesure_seul']} pb] "
                  f"regret(mesure+estime)={r['regret_mesure_estime_geo']:.4f} [{r['n_pb_regret_mesure_estime']} pb] "
                  f"commun={r['regret_mesure_seul_commun_geo']} [{r['n_pb_commun']} pb, {r['n_inst_commun']} inst]")
    print("sensibilite faible precision incluse :")
    for r in res_lp["synth"]:
        if r["portee"] == "toutes":
            print(f"  {r['regle']:9s} regret(mesure seul)={r['regret_mesure_seul_geo']:.4f} [{r['n_pb_regret_mesure_seul']} pb] "
                  f"regret(mesure+estime)={r['regret_mesure_estime_geo']:.4f} [{r['n_pb_regret_mesure_estime']} pb]")
    print(f"plis ecartes (n* non determinable) : {res['skipped']}")
    print(f"n* plein echantillon = {res['n_star_full']} ; n* des plis : "
          f"{sorted({str(r['n_star_pli']) for r in res['fold_rows']})}")
    print(f"L1 seule : accord={np.mean([r['accord'] for r in l1]):.4f} perte={np.mean([r['perte'] for r in l1]):.4f}")
    with open(os.path.join(OUT_DIR, "b3_meta.json"), "w", encoding="utf-8") as fh:
        json.dump({"plis_ecartes": res["skipped"], "n_plis": len(res["problems"]) - len(res["skipped"]),
                   "delta": DELTA_MAIN, "seed_bootstrap": SEED_BOOT, "B_bootstrap": B_BOOT,
                   "n_star_plein_echantillon": res["n_star_full"] if math.isfinite(res["n_star_full"]) else None,
                   "n_star_plein_echantillon_note": "None = non determinable (seuil +infini)",
                   "n_star_pli_valeurs": sorted({str(r["n_star_pli"]) for r in res["fold_rows"]}),
                   "n_instances_sous_ensemble_commun_mesure_seul": res["n_common_inst"],
                   "n_instances_sous_ensemble_commun_mesure_estime": res["n_common_inst_estime"],
                   "estimations_utilisees_par_solveur": res["use"]}, fh, indent=2)


if __name__ == "__main__":
    main()
