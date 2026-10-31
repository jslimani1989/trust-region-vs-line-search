# -*- coding: utf-8 -*-
"""
Consigne 009e (reprise de 008d-bis), etape 1 (arbre) : regle d'apprentissage des feuilles avec marge de
non-inferiorite (reprise de 008d, etape 4), sur les mesures v0.5 (passage A), avec :
  - DECISION 2 : seuil du noeud 2 = n* CONTINU (aucun arrondi), croisement montant, cellules >= 3 problemes ;
  - DECISION 5 : temps estimes seulement la ou aucune mesure n'existe ;
  - DECISION 3 : le noeud 2 est formule comme "factorisation dense abordable ?" (n < n*), n* etant un repere
    mesure sur la machine de reference ; l'argument est uniquement dense (cout O(n^3) par iteration pour la
    factorisation, produits matrice-vecteur denses).
Structure de l'arbre inchangee (3 noeuds, 5 feuilles, cf. tree_model_b2.py).

Regle (dans chaque feuille) :
 1. Plus robuste = meilleur taux de succes (moyenne par probleme, restreinte a la feuille).
 2. Survivants = le plus robuste et les candidats dont le taux est au plus delta en dessous
    (delta = 0,05 fixe a priori). Le test de permutation (inversion de signe, Holm dans la feuille)
    est conserve et rapporte A TITRE D'INFORMATION.
 3. Depart des survivants par le temps (mesure >= 1 ms ; estime seulement si aucune mesure) : pour une
    paire, moyenne geometrique, sur les problemes, du ratio (moyenne geometrique par probleme) des temps
    sur les instances de la feuille resolues par les deux. Un ratio de temps n'est utilise que s'il repose
    sur AU MOINS 3 problemes communs ; sinon, ratio de N_tot pour cette paire (signale). Avec plus de
    2 survivants : score d'un survivant = moyenne geometrique de ses ratios contre les autres survivants ;
    le plus petit score gagne.
Sensibilites : delta in {0 ; 0,02 ; 0,10}, regle de 008c (survivants par permutation + Holm ; depart par le
temps sans le seuil de 3 problemes), delta = 0,05 sans aucun temps estime, delta = 0,05 avec les taches
"faible precision" utilisees comme mesures.
"""
import csv
import json
import os

import numpy as np

from common_data import load_distinct_rows
from common_timing import get_timing
from tree_model_b2 import LEAVES, build_leaf_instances
from etape5_statistiques import sign_flip_pvalue_exact, sign_flip_pvalue_mc, holm
from b1_seuil_noeud2 import n_star_point_estimate, n_star_effective
from time_model import load_timebook
from paths import OUT_DIR

ALPHA = 0.05
DELTA_MAIN = 0.05
MIN_COMMON_PROBLEMS = 3
SEED_MC_B2 = 20260927
N_MC_B2 = 100000
EXACT_MAX_PROBLEMS = 20
EPS = 1e-9


def sign_flip_pvalue(d):
    if len(d) <= EXACT_MAX_PROBLEMS:
        return sign_flip_pvalue_exact(d), f"exact (2^{len(d)})"
    rng = np.random.default_rng(SEED_MC_B2)
    return sign_flip_pvalue_mc(d, N_MC_B2, rng), f"Monte-Carlo ({N_MC_B2}, graine={SEED_MC_B2})"


def index_rows(rows):
    idx = {}
    for r in rows:
        idx.setdefault((r["problem"], r["n"]), {})[r["method"]] = {
            "success": r["status"] == "success", "n_total": int(r["n_total"])}
    return idx


def per_problem_leaf_rates(rows_idx, instances, method):
    by_problem = {}
    for (p, n) in instances:
        d = rows_idx[(p, n)].get(method)
        if d is not None:
            by_problem.setdefault(p, []).append(d["success"])
    return {p: float(np.mean(v)) for p, v in by_problem.items()}


def leaf_success_rate(rows_idx, instances, method):
    rates = per_problem_leaf_rates(rows_idx, instances, method)
    if not rates:
        return None, {}
    return float(np.mean(list(rates.values()))), rates


def pair_ratio(rows_idx, instances, a, b, tb, allow_estimate, min_problems, allow_lp=False):
    """Ratio de cout a/b sur les instances de la feuille resolues par les deux.
    Retourne dict(source, n_problemes, ratio, n_instances) ou None. Source = 'temps' si au moins `min_problems`
    problemes communs ont un temps (mesure ou estime) pour a ET b ; sinon 'N_tot'."""
    by_problem_t, by_problem_n = {}, {}
    for (p, n) in instances:
        da, db = rows_idx[(p, n)].get(a), rows_idx[(p, n)].get(b)
        if da is None or db is None or not (da["success"] and db["success"]):
            continue
        if da["n_total"] > 0 and db["n_total"] > 0:
            by_problem_n.setdefault(p, []).append(da["n_total"] / db["n_total"])
        ta, sa = tb.get(p, n, a, allow_estimate, allow_lp)
        tb_, sb = tb.get(p, n, b, allow_estimate, allow_lp)
        if ta is not None and tb_ is not None and ta > 0 and tb_ > 0:
            by_problem_t.setdefault(p, []).append((ta / tb_, sa, sb))

    def agg(d, f=lambda x: x):
        pg = [float(np.exp(np.mean(np.log([f(x) for x in v])))) for v in d.values()]
        return float(np.exp(np.mean(np.log(pg)))) if pg else None
    if len(by_problem_t) >= min_problems and len(by_problem_t) >= 1:
        n_vals = sum(len(v) for v in by_problem_t.values()) * 2
        n_est = sum((x[1] == "estime") + (x[2] == "estime") for v in by_problem_t.values() for x in v)
        return {"source": "temps", "n_problemes": len(by_problem_t),
                "n_instances": sum(len(v) for v in by_problem_t.values()),
                "ratio": agg(by_problem_t, lambda x: x[0]), "n_valeurs_temps": n_vals, "n_valeurs_estimees": n_est}
    if by_problem_n:
        return {"source": "N_tot", "n_problemes": len(by_problem_n),
                "n_instances": sum(len(v) for v in by_problem_n.values()),
                "ratio": agg(by_problem_n), "n_valeurs_temps": 0, "n_valeurs_estimees": 0,
                "n_problemes_temps_disponibles": len(by_problem_t)}
    return None


def pick_fastest(rows_idx, instances, candidates, tb, allow_estimate, min_problems, allow_lp=False):
    pair_info = {}
    if len(candidates) == 1:
        return candidates[0], {candidates[0]: 1.0}, pair_info
    scores = {}
    for m in candidates:
        ratios = []
        for other in candidates:
            if other == m:
                continue
            info = pair_ratio(rows_idx, instances, m, other, tb, allow_estimate, min_problems, allow_lp)
            pair_info[(m, other)] = info
            if info is not None and info["ratio"] is not None:
                ratios.append(info["ratio"])
        scores[m] = float(np.exp(np.mean(np.log(ratios)))) if ratios else None
    valid = {m: s for m, s in scores.items() if s is not None}
    if not valid:
        return candidates[0], scores, pair_info
    return min(valid, key=valid.get), scores, pair_info


def learn_leaf(rows_idx, instances, candidates, tb, delta=DELTA_MAIN, rule="delta", allow_estimate=True,
               compute_tests=True, allow_lp=False):
    if not instances:  # feuille vide (seuil n* infini : aucune instance "non abordable")
        return {"taux": {}, "plus_robuste": None, "tests": [], "survivants": [], "departage_effectue": False, "scores": {},
                "recommande": None, "paires": {}, "n_paires": 0, "n_paires_ntot": 0, "part_estimee": None,
                "n_valeurs_temps": 0, "n_instances": 0, "n_problemes": 0}
    rates, per_problem = {}, {}
    for m in candidates:
        rate, pp = leaf_success_rate(rows_idx, instances, m)
        rates[m], per_problem[m] = rate, pp
    valid_c = [m for m in candidates if rates[m] is not None]
    most_robust = max(valid_c, key=lambda m: rates[m])

    tests = []
    if compute_tests or rule == "008c":
        others = [m for m in valid_c if m != most_robust]
        pvals = []
        for o in others:
            probs = sorted(set(per_problem[most_robust]) & set(per_problem[o]))
            d = np.array([per_problem[most_robust][p] - per_problem[o][p] for p in probs])
            if len(d) > 0:
                p_raw, meth = sign_flip_pvalue(d)
            else:
                p_raw, meth = 1.0, "n/a"
            tests.append({"candidat": o, "n_problemes": len(probs), "mean_d": float(np.mean(d)) if len(d) else None,
                          "p_brut": p_raw, "methode_p": meth})
            pvals.append(p_raw)
        ph = holm(np.array(pvals)) if pvals else np.array([])
        for t, h in zip(tests, ph):
            t["p_holm"] = float(h)

    if rule == "008c":
        survivors = [most_robust] + [t["candidat"] for t in tests if t["p_holm"] >= ALPHA]
        min_problems = 1
    else:
        survivors = [m for m in valid_c if rates[most_robust] - rates[m] <= delta + EPS]
        min_problems = MIN_COMMON_PROBLEMS
        survivors.sort(key=lambda m: -rates[m])
    winner, scores, pair_info = pick_fastest(rows_idx, instances, survivors, tb, allow_estimate, min_problems, allow_lp)

    n_vals = sum((i or {}).get("n_valeurs_temps", 0) for i in pair_info.values())
    n_est = sum((i or {}).get("n_valeurs_estimees", 0) for i in pair_info.values())
    n_pairs_ntot = sum(1 for i in pair_info.values() if i is not None and i["source"] == "N_tot") // 2
    n_pairs = len(pair_info) // 2
    return {
        "taux": rates, "plus_robuste": most_robust, "tests": tests, "survivants": survivors,
        "departage_effectue": len(survivors) > 1, "scores": scores, "recommande": winner,
        "paires": pair_info, "n_paires": n_pairs, "n_paires_ntot": n_pairs_ntot,
        "part_estimee": (n_est / n_vals) if n_vals else None, "n_valeurs_temps": n_vals,
        "n_instances": len(instances), "n_problemes": len({p for p, n in instances}),
    }


def tree_for_variant(rows, rows_idx, threshold_n, tb, delta, rule, allow_estimate, compute_tests=False, allow_lp=False):
    leaf_data = build_leaf_instances(rows, threshold_n)
    return {leaf: learn_leaf(rows_idx, leaf_data[leaf]["instances"], leaf_data[leaf]["candidats"], tb,
                             delta=delta, rule=rule, allow_estimate=allow_estimate, compute_tests=compute_tests,
                             allow_lp=allow_lp)
            for leaf in LEAVES}, leaf_data


def discrimination(rec, threshold_n):
    disc = [{"noeud": "1 (Hessienne disponible ?)", "discrimine": True,
             "raison": f"ensembles de solveurs disjoints par construction (L1={rec['L1_pas_hessienne']} vs branche Hessienne)"}]
    if rec.get("L4_hess_nonabordable_AB") is None and rec.get("L5_hess_nonabordable_C") is None:
        disc.append({"noeud": "2 (factorisation dense abordable ? n < n*, n* non determinable -> seuil infini)", "discrimine": False,
                     "raison": "feuilles L4 et L5 vides : aucune instance n >= n* (n* non determinable) ; noeud 2 candidat a la suppression"})
        n3_only = rec["L2_hess_abordable_AB"] != rec["L3_hess_abordable_C"]
        disc.append({"noeud": "3 (Hessienne possiblement indefinie, Cat.C ?)", "discrimine": n3_only,
                     "raison": f"L2 (A/B): {rec['L2_hess_abordable_AB']} vs L3 (C): {rec['L3_hess_abordable_C']} "
                               f"[{'change' if n3_only else 'identique'}] ; L4/L5 vides"})
        return disc
    n2_ab = rec["L2_hess_abordable_AB"] != rec["L4_hess_nonabordable_AB"]
    n2_c = rec["L3_hess_abordable_C"] != rec["L5_hess_nonabordable_C"]
    disc.append({"noeud": f"2 (factorisation dense abordable ? n < n* = {threshold_n:.1f})", "discrimine": n2_ab or n2_c,
                 "raison": f"A/B: {rec['L2_hess_abordable_AB']} (n<n*) vs {rec['L4_hess_nonabordable_AB']} (n>=n*) "
                           f"[{'change' if n2_ab else 'identique'}] ; "
                           f"C: {rec['L3_hess_abordable_C']} (n<n*) vs {rec['L5_hess_nonabordable_C']} (n>=n*) "
                           f"[{'change' if n2_c else 'identique'}]"})
    n3_a = rec["L2_hess_abordable_AB"] != rec["L3_hess_abordable_C"]
    n3_b = rec["L4_hess_nonabordable_AB"] != rec["L5_hess_nonabordable_C"]
    disc.append({"noeud": "3 (Hessienne possiblement indefinie, Cat.C ?)", "discrimine": n3_a or n3_b,
                 "raison": f"n<n*: {rec['L2_hess_abordable_AB']} (A/B) vs {rec['L3_hess_abordable_C']} (C) "
                           f"[{'change' if n3_a else 'identique'}] ; "
                           f"n>=n*: {rec['L4_hess_nonabordable_AB']} (A/B) vs {rec['L5_hess_nonabordable_C']} (C) "
                           f"[{'change' if n3_b else 'identique'}]"})
    return disc


VARIANTS = [("delta=0.05 (principale)", dict(delta=0.05, rule="delta", allow_estimate=True)),
            ("delta=0", dict(delta=0.0, rule="delta", allow_estimate=True)),
            ("delta=0.02", dict(delta=0.02, rule="delta", allow_estimate=True)),
            ("delta=0.10", dict(delta=0.10, rule="delta", allow_estimate=True)),
            ("regle 008c (permutation+Holm)", dict(delta=None, rule="008c", allow_estimate=True)),
            ("delta=0.05 sans temps estimes", dict(delta=0.05, rule="delta", allow_estimate=False)),
            ("delta=0.05 faible precision incluse", dict(delta=0.05, rule="delta", allow_estimate=True, allow_lp=True))]


def tree_all_variants(timing, threshold_n, rows=None, rows_idx=None, tb=None):
    rows = rows if rows is not None else load_distinct_rows()
    rows_idx = rows_idx if rows_idx is not None else index_rows(rows)
    if tb is None:
        tb = load_timebook(timing)[0]
    results = {}
    for name, kw in VARIANTS:
        res, _ = tree_for_variant(rows, rows_idx, threshold_n, tb, kw["delta"], kw["rule"], kw["allow_estimate"],
                                  compute_tests=name.startswith("delta=0.05 ("), allow_lp=kw.get("allow_lp", False))
        results[name] = res
    return results


def main():
    rows = load_distinct_rows()
    rows_idx = index_rows(rows)
    timing = get_timing("principale")
    with open(os.path.join(OUT_DIR, "b1_seuil_noeud2.json"), encoding="utf-8") as fh:
        b1 = json.load(fh)["principale"]
    threshold_n = n_star_effective(timing.wall, rows=rows)
    if b1.get("n_star_exact") is None:
        assert threshold_n == float("inf")
        print("Seuil du noeud 2 : n* NON DETERMINABLE (aucun croisement montant) -> seuil +infini (feuilles L4 et L5 vides)")
    else:
        assert abs(threshold_n - b1["n_star_exact"]) < 1e-9
        print(f"Seuil du noeud 2 : n < n* = {threshold_n:.2f} (IC95 bootstrap [{b1['n_star_IC95_bas']}, {b1['n_star_IC95_haut']}]), sans arrondi")
    tb, recs, va, vl, use = load_timebook(timing)

    results = tree_all_variants(timing, threshold_n, rows, rows_idx, tb)
    main_res = results["delta=0.05 (principale)"]
    out_rows, pair_rows = [], []
    for leaf in LEAVES:
        info = main_res[leaf]
        print(f"\n{leaf} (instances={info['n_instances']}, problemes={info['n_problemes']})")
        if info["n_instances"] == 0:
            print("  feuille vide")
            out_rows.append({"leaf": leaf, "n_instances": 0, "n_problemes": 0, "recommande": "(feuille vide)", "plus_robuste": "",
                             "taux_plus_robuste": "", "survivants_delta_0.05": "", "taux_tous_candidats": "{}",
                             "paires_departage_N_tot": "", "part_valeurs_temps_estimees": "", "tests_permutation_info": "[]",
                             "scores_departage": "{}"})
            continue
        print("  taux : " + ", ".join(f"{m}={r:.4f}" for m, r in info["taux"].items()))
        print(f"  plus robuste : {info['plus_robuste']} ; survivants (delta=0.05) : {info['survivants']}")
        for t in info["tests"]:
            print(f"    perm vs {t['candidat']:13s} mean_d={t['mean_d']:+.4f} p_holm={t['p_holm']:.4f}")
        print(f"  => recommande : {info['recommande']}  (paires N_tot : {info['n_paires_ntot']}/{info['n_paires']} ; "
              f"part estimee : {info['part_estimee']})")
        out_rows.append({
            "leaf": leaf, "n_instances": info["n_instances"], "n_problemes": info["n_problemes"],
            "recommande": info["recommande"], "plus_robuste": info["plus_robuste"],
            "taux_plus_robuste": round(info["taux"][info["plus_robuste"]], 4),
            "survivants_delta_0.05": ";".join(info["survivants"]),
            "taux_tous_candidats": json.dumps({k: round(v, 4) for k, v in info["taux"].items()}),
            "paires_departage_N_tot": f"{info['n_paires_ntot']}/{info['n_paires']}",
            "part_valeurs_temps_estimees": round(info["part_estimee"], 3) if info["part_estimee"] is not None else "",
            "tests_permutation_info": json.dumps(info["tests"]),
            "scores_departage": json.dumps({k: (round(v, 4) if v is not None else None) for k, v in info["scores"].items()}),
        })
    with open(os.path.join(OUT_DIR, "b2_feuilles.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)
    for name, _ in VARIANTS:
        for leaf in LEAVES:
            for (a, b), pi in results[name][leaf]["paires"].items():
                if pi is not None and a < b:
                    pair_rows.append({"variante": name, "leaf": leaf, "a": a, "b": b, "source": pi["source"],
                                      "n_problemes": pi["n_problemes"], "n_instances": pi["n_instances"],
                                      "ratio_a_sur_b": round(pi["ratio"], 4),
                                      "part_valeurs_estimees": (round(pi["n_valeurs_estimees"] / pi["n_valeurs_temps"], 3)
                                                                if pi["n_valeurs_temps"] else "")})
    with open(os.path.join(OUT_DIR, "b2_paires_departage.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(pair_rows[0].keys()))
        w.writeheader()
        w.writerows(pair_rows)

    sens_rows = []
    print("\n=== Sensibilite (recommandation par feuille) ===")
    for leaf in LEAVES:
        row = {"leaf": leaf}
        for name, _ in VARIANTS:
            row[name] = results[name][leaf]["recommande"]
            row[name + " | survivants"] = ";".join(results[name][leaf]["survivants"])
        ref = row["delta=0.05 (principale)"]
        row["change_vs_principale"] = ";".join(n for n, _ in VARIANTS[1:] if row[n] != ref)
        sens_rows.append(row)
        print(f"  {leaf}: " + " | ".join(f"{n.split(' ')[0]}={row[n]}" for n, _ in VARIANTS)
              + (f"   <-- CHANGE pour : {row['change_vs_principale']}" if row["change_vs_principale"] else ""))
    with open(os.path.join(OUT_DIR, "b2_sensibilite.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(sens_rows[0].keys()))
        w.writeheader()
        w.writerows(sens_rows)

    rec = {leaf: main_res[leaf]["recommande"] for leaf in LEAVES}
    disc = discrimination(rec, threshold_n)
    print("\n=== Discrimination des noeuds (principale) ===")
    for d in disc:
        print(f"  Noeud {d['noeud']} : {'DISCRIMINE' if d['discrimine'] else 'NE DISCRIMINE PAS'} -- {d['raison']}")
    with open(os.path.join(OUT_DIR, "b2_discrimination_noeuds.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["noeud", "discrimine", "raison"])
        w.writeheader()
        w.writerows(disc)
    disc_var = []
    for name, _ in VARIANTS:
        recv = {leaf: results[name][leaf]["recommande"] for leaf in LEAVES}
        dv = discrimination(recv, threshold_n)
        disc_var.append({"variante": name, "noeud1": dv[0]["discrimine"], "noeud2": dv[1]["discrimine"], "noeud3": dv[2]["discrimine"],
                         "recommandations": json.dumps(recv)})
    with open(os.path.join(OUT_DIR, "b2_discrimination_par_variante.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(disc_var[0].keys()))
        w.writeheader()
        w.writerows(disc_var)


if __name__ == "__main__":
    main()
