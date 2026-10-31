# -*- coding: utf-8 -*-
"""
Consigne 009e (reprise de 008d-bis), etape 1 (n*) : B1 refait sur les mesures v0.5 (passage A), avec la DECISION 2 :
  - croisement MONTANT seulement (rapport Newton_BT / Steihaug_CG qui passe au-dessus de 1) ;
  - SEULES les cellules (dimensions) d'au moins 3 problemes sont retenues ;
  - PAS d'arrondi : n* est utilise tel quel (valeur continue), avec son intervalle.
Methode inchangee par ailleurs (008c/008d) : rapport Newton_BT / Steihaug_CG sur les instances de
Categorie A resolues par les deux et a temps mesures >= 1 ms (aucun temps estime), moyenne
geometrique par dimension, interpolation log-log entre deux dimensions consecutives retenues,
intervalle bootstrap par grappes (probleme), percentile, 10 000 reechantillons.

Bootstrap : les cellules retenues sont celles de l'echantillon d'origine (>= 3 problemes) ;
dans un reechantillon, une cellule retenue sans aucun probleme tire est ignoree.

Sensibilites (supplements, signales) : (a) instances "faible precision" incluses ; (c) toutes les
cellules (sans la regle des 3 problemes) ; (d) critere litteral de 008c (deux sens) ; ainsi que
n* sans chaque probleme (un a la fois : valeurs des plis de la validation hors echantillon).
"""
import csv
import json
import math
import os

import numpy as np
from scipy import stats as sps

from common_data import load_distinct_rows
from common_timing import get_timing, LOW_PREC_S
from etape5_statistiques import bca_ci
from paths import OUT_DIR

SEED_BOOT_B1 = 20260927
B_BOOT = 10000
MIN_PROBLEMS_CELL = 3
RATIO_KEY = "ratio_geo_moyen_NewtonLM_sur_SteihaugCG"


def build_ratio_table(times, min_time=LOW_PREC_S, exclude_problem=None, rows=None):
    """Ratios Newton_BT / Steihaug_CG par instance Cat. A (resolue par les deux). Les instances dont l'un des
    deux temps est absent ou < min_time sont exclues (comptees)."""
    rows = rows if rows is not None else load_distinct_rows()
    catA = [r for r in rows if r["category"] == "A" and r["problem"] != exclude_problem]
    status = {(r["problem"], r["n"], r["method"]): r["status"] for r in catA}
    by_n, by_pn = {}, {}
    n_both = n_low = 0
    instances = sorted({(r["problem"], r["n"]) for r in catA}, key=lambda t: (t[0], int(t[1])))
    for p, n in instances:
        if status.get((p, n, "Newton_BT")) != "success" or status.get((p, n, "Steihaug_CG")) != "success":
            continue
        n_both += 1
        t_nb, t_cg = times.get((p, n, "Newton_BT")), times.get((p, n, "Steihaug_CG"))
        if t_nb is None or t_cg is None or t_nb < min_time or t_cg < min_time or t_nb <= 0 or t_cg <= 0:
            n_low += 1
            continue
        ratio = t_nb / t_cg
        by_n.setdefault(int(n), []).append((p, ratio))
        by_pn[(p, n)] = ratio
    meta = {"n_instances_A": len(instances), "n_resolues_par_les_deux": n_both, "n_exclues_faible_precision": n_low}
    return by_n, by_pn, instances, meta


def geomean_by_dim(by_n, rng):
    out = {}
    for n, pairs in sorted(by_n.items()):
        ratios = np.array([r for _, r in pairs])
        log_r = np.log(ratios)
        gm = float(np.exp(np.mean(log_r)))
        if len(ratios) >= 2:
            _, lo_log, hi_log = bca_ci(log_r, rng, B=B_BOOT)
            lo, hi = math.exp(lo_log), math.exp(hi_log)
        else:
            lo = hi = None
        out[n] = {"n": n, "n_problemes": len(pairs), RATIO_KEY: gm,
                  "IC95_bas": lo, "IC95_haut": hi, "problemes": ";".join(p for p, _ in pairs)}
    return out


def interpolate_crossing(dim_stats, upward_only=True):
    """n* : interpolation log-log entre deux dimensions consecutives (de dim_stats) ou R(n) PASSE AU-DESSUS de 1
    (r_lo < 1 <= r_hi), premiere paire trouvee. Decision 2 : croisement montant seulement (upward_only=True) ;
    upward_only=False restitue le critere litteral de 008c (deux sens), rapporte a titre de transparence."""
    ns = sorted(dim_stats.keys())
    rs = [dim_stats[n][RATIO_KEY] for n in ns]
    for i in range(len(ns) - 1):
        n_lo, n_hi = ns[i], ns[i + 1]
        r_lo, r_hi = rs[i], rs[i + 1]
        crosses = (r_lo < 1.0 <= r_hi) if upward_only else ((r_lo < 1.0 <= r_hi) or (r_hi < 1.0 <= r_lo))
        if crosses:
            log_r_lo, log_r_hi = math.log10(r_lo), math.log10(r_hi)
            if log_r_hi == log_r_lo:
                continue
            frac = (0.0 - log_r_lo) / (log_r_hi - log_r_lo)
            log_n_star = math.log10(n_lo) + frac * (math.log10(n_hi) - math.log10(n_lo))
            return 10 ** log_n_star, n_lo, n_hi
    return None, None, None


def retained_cells(by_n, min_problems=MIN_PROBLEMS_CELL):
    return {n: v for n, v in by_n.items() if len({p for p, _ in v}) >= min_problems}


def point_stats(by_n):
    return {n: {RATIO_KEY: float(sps.gmean([r for _, r in pairs]))} for n, pairs in by_n.items()}


def n_star_point_estimate(times, exclude_problem=None, min_time=LOW_PREC_S, min_problems=MIN_PROBLEMS_CELL, rows=None):
    """Estimation ponctuelle continue de n* (aucun arrondi), croisement montant, cellules >= min_problems.
    Retourne None si aucun croisement montant n'est determinable."""
    by_n, _, _, _ = build_ratio_table(times, min_time=min_time, exclude_problem=exclude_problem, rows=rows)
    ret = retained_cells(by_n, min_problems)
    n_star, _, _ = interpolate_crossing(point_stats(ret), upward_only=True)
    return n_star


def n_star_effective(times, exclude_problem=None, min_time=LOW_PREC_S, min_problems=MIN_PROBLEMS_CELL, rows=None):
    """Seuil utilise par les regles de decision. CONVENTION 009e : si aucun croisement montant n'est determinable
    (le rapport Newton-LM / Steihaug-CG ne passe jamais au-dessus de 1 entre dimensions retenues consecutives), le noeud 2 est
    candidat a la suppression (consigne 008c) et le seuil est pris egal a +infini : aucune instance n'est "non abordable"."""
    v = n_star_point_estimate(times, exclude_problem=exclude_problem, min_time=min_time, min_problems=min_problems, rows=rows)
    return math.inf if v is None else v


def bootstrap_crossing(by_n_retained, rng, n_boot=B_BOOT):
    problems = sorted({p for pairs in by_n_retained.values() for p, _ in pairs})
    samples, n_degenerate = [], 0
    for _ in range(n_boot):
        sampled = rng.choice(problems, size=len(problems), replace=True)
        counts = {}
        for p in sampled:
            counts[p] = counts.get(p, 0) + 1
        dim_boot = {}
        for n, pairs in by_n_retained.items():
            weighted = []
            for p, r in pairs:
                weighted.extend([r] * counts.get(p, 0))
            if weighted:
                dim_boot[n] = {RATIO_KEY: float(sps.gmean(weighted))}
        n_star, _, _ = interpolate_crossing(dim_boot, upward_only=True)
        if n_star is None:
            n_degenerate += 1
        else:
            samples.append(n_star)
    return samples, n_degenerate


def analyse(times, min_time, rng, label, write_prefix=None, min_problems=MIN_PROBLEMS_CELL, rows=None, boot=True):
    """Analyse n* complete (ratios par dimension, n*, IC bootstrap). Retourne un dict serialisable ; le detail par
    dimension (cellules) est sous 'par_dimension' avec le drapeau 'retenue'."""
    rows = rows if rows is not None else load_distinct_rows()
    by_n, by_pn, instances, meta = build_ratio_table(times, min_time=min_time, rows=rows)
    meta["n_exploitables"] = len(by_pn)
    dim_stats = geomean_by_dim(by_n, rng)
    ret = retained_cells(by_n, min_problems)
    res = {"variante": label, **meta,
           "regle_cellules": f"cellules d'au moins {min_problems} problemes",
           "dimensions_retenues": sorted(ret),
           "par_dimension": {str(n): {**{k: (round(v, 4) if isinstance(v, float) else v) for k, v in d.items()
                                         if k != "problemes"}, "retenue": n in ret}
                             for n, d in dim_stats.items()}}
    lit, lit_lo, lit_hi = interpolate_crossing(dim_stats, upward_only=False)
    res["n_star_sensibilite_toutes_cellules_deux_sens_critere_litteral_008c"] = {
        "n_star": round(lit, 2) if lit else None, "entre": [lit_lo, lit_hi]}
    allc, allc_lo, allc_hi = interpolate_crossing(dim_stats, upward_only=True)
    res["n_star_sensibilite_toutes_cellules_montant"] = {"n_star": round(allc, 2) if allc else None,
                                                         "entre": [allc_lo, allc_hi]}
    ret_stats = {n: dim_stats[n] for n in ret}
    n_star, n_lo, n_hi = interpolate_crossing(ret_stats, upward_only=True)
    if n_star is None:
        res.update({"n_star": None, "n_star_exact": None, "noeud2_candidat_suppression": True,
                    "n_star_note": "non determinable : le rapport ne franchit jamais 1 vers le haut entre dimensions retenues "
                                   "consecutives (cellules >= 3 problemes) ; noeud 2 candidat a la suppression (consigne 008c)",
                    "ratios_par_dimension_retenue": {str(n): round(ret_stats[n][RATIO_KEY], 4) for n in sorted(ret_stats)}})
        if boot:  # information complementaire : le croisement existe-t-il dans des reechantillons bootstrap ?
            by_n_ret = {n: by_n[n] for n in ret}
            samples, n_degenerate = bootstrap_crossing(by_n_ret, rng)
            q = np.percentile(samples, [2.5, 50, 97.5]) if samples else [None] * 3
            res.update({"bootstrap_part_replicats_avec_croisement_montant": len(samples) / (len(samples) + n_degenerate),
                        "bootstrap_n_replicats_avec_croisement": len(samples),
                        "bootstrap_n_replicats_sans_croisement": n_degenerate,
                        "bootstrap_n_star_quantiles_2.5_50_97.5_conditionnels": [None if x is None else round(float(x), 2) for x in q],
                        "bootstrap_note": "information descriptive, conditionnelle a l'existence d'un croisement ; ne definit pas un IC de n*"})
    else:
        res.update({"n_star": round(n_star, 2), "n_star_exact": n_star, "n_encadrant_bas": n_lo, "n_encadrant_haut": n_hi,
                    "noeud2_candidat_suppression": False})
        if boot:
            by_n_ret = {n: by_n[n] for n in ret}
            samples, n_degenerate = bootstrap_crossing(by_n_ret, rng)
            lo_b, hi_b = np.percentile(samples, [2.5, 97.5]) if samples else (None, None)
            res.update({"n_star_IC95_bas": round(float(lo_b), 2) if lo_b is not None else None,
                        "n_star_IC95_haut": round(float(hi_b), 2) if hi_b is not None else None,
                        "n_star_IC95_bas_exact": float(lo_b) if lo_b is not None else None,
                        "n_star_IC95_haut_exact": float(hi_b) if hi_b is not None else None,
                        "n_bootstrap_replicats_determines": len(samples),
                        "n_bootstrap_replicats_degeneres": n_degenerate})
    if write_prefix:
        with open(os.path.join(OUT_DIR, f"{write_prefix}_ratio_par_dimension.csv"), "w", newline="",
                  encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=["n", "n_problemes", "retenue", RATIO_KEY, "IC95_bas", "IC95_haut", "problemes"])
            w.writeheader()
            for n in sorted(dim_stats):
                d = dim_stats[n]
                w.writerow({"n": n, "n_problemes": d["n_problemes"], "retenue": int(n in ret),
                            RATIO_KEY: round(d[RATIO_KEY], 4),
                            "IC95_bas": "" if d["IC95_bas"] is None else round(d["IC95_bas"], 4),
                            "IC95_haut": "" if d["IC95_haut"] is None else round(d["IC95_haut"], 4),
                            "problemes": d["problemes"]})
    return res


def leave_one_out(times, min_time=LOW_PREC_S, rows=None):
    rows = rows if rows is not None else load_distinct_rows()
    probs = sorted({r["problem"] for r in rows if r["category"] == "A"})
    out = {}
    for p in probs:
        v = n_star_point_estimate(times, exclude_problem=p, min_time=min_time, rows=rows)
        out[p] = None if v is None else round(v, 2)
    return out


def main():
    rows = load_distinct_rows()
    t = get_timing("principale")
    out = {}
    out["principale"] = analyse(t.wall, LOW_PREC_S, np.random.default_rng(SEED_BOOT_B1),
                                "principale : passage A v0.5, mediane du temps reel, instances >= 1 ms, temps mesures, "
                                "cellules >= 3 problemes, croisement montant, sans arrondi", "b1", rows=rows)
    out["sensibilite_a_avec_faible_precision"] = analyse(
        t.wall, 1e-12, np.random.default_rng(SEED_BOOT_B1 + 1),
        "sensibilite (a) : mediane du temps reel, instances < 1 ms incluses, cellules >= 3 problemes",
        "b1_sensib_a", rows=rows)
    out["n_star_sans_chaque_probleme_CatA"] = leave_one_out(t.wall, rows=rows)
    vals = [v for v in out["n_star_sans_chaque_probleme_CatA"].values() if v is not None]
    out["n_star_sans_chaque_probleme_resume"] = {"n_plis_avec_croisement": len(vals), "min": min(vals) if vals else None,
                                                  "max": max(vals) if vals else None,
                                                  "problemes_non_determinables": [p for p, v in out["n_star_sans_chaque_probleme_CatA"].items() if v is None]}
    with open(os.path.join(OUT_DIR, "b1_seuil_noeud2.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
