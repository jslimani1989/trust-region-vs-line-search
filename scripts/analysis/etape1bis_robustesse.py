# -*- coding: utf-8 -*-
"""
Consigne 009e, priorite 3 : robustesse des conclusions a la qualite des mesures (la nuit v0.5 n'a pas respecte le critere
de +/-10 % sur la mesure de controle : 52 mesures sur 118 hors tolerance en A, 42 sur 118 en B).

Pour n*, le rapport des temps par dimension, les pentes du front de Pareto et le classement des quatre regles du
compromis, trois variantes en plus de l'analyse principale (passage A, mediane) :
  (R1) minimum des repetitions du passage A ;
  (R2) passage A sans les blocs de 20 executions encadres par une mesure de controle hors de +/-10 % de la mediane des
       mesures de controle (lecture "l'une des deux" ; lecture "les deux" = R2b, diagnostic) ;
  (R3) passage B (replication), memes 784 taches.

REGLE FIXEE AVANT DE VOIR LES RESULTATS v0.5 (consigne 009e) : conclusions robustes si, dans R1, R2 (lecture "l'une des
deux") et R3 :
  (i)   n* reste dans l'intervalle de l'analyse principale ;
  (ii)  les solveurs du front de Pareto sont les memes ;
  (iii) l'ordre des quatre regles sur la PERTE DE SUCCES est inchange (ordre faible, egalite a 1e-9 pres).
Les ordres sur le regret sont RAPPORTES SANS CRITERE (008d-bis : ecarts de 2 a 8 %, sous le bruit de mesure).
ADAPTATION consignee avant le calcul des variantes (robustesse/regle_adaptee_prealable.md) : l'analyse principale n'ayant
aucun croisement montant (n* non determinable), le critere (i) est remplace par (i') : n* est aussi non determinable dans
la variante. Convention de calcul : seuil +infini (voir regle_adaptee_prealable.md).
"""
import csv
import json
import os

import numpy as np

from common_data import load_distinct_rows
from common_timing import get_timing, LOW_PREC_S, control_excursions
from b1_seuil_noeud2 import analyse, SEED_BOOT_B1, RATIO_KEY
from b3_validation_LOO import run_loo, RULES
from b4_pareto_temps import load_robustness, front_for_variant
from b2_arbre_decision_temps import index_rows
from paths import OUT_DIR, ROB_DIR

TOL_ORDER = 1e-9
VARIANTS = ["principale", "R1", "R2", "R3"]
DIAGNOSTICS = ["R2b"]


def weak_order(values):
    """values : {regle: valeur}. Ordre faible croissant : liste de groupes (frozenset) ex aequo a TOL_ORDER."""
    items = sorted(values.items(), key=lambda kv: kv[1])
    groups, cur, last = [], [], None
    for k, v in items:
        if last is None or abs(v - last) <= TOL_ORDER:
            cur.append(k)
        else:
            groups.append(frozenset(cur))
            cur = [k]
        last = v if last is None or abs(v - last) > TOL_ORDER else last
    if cur:
        groups.append(frozenset(cur))
    return groups


def fmt_order(groups):
    return " < ".join("{" + ",".join(sorted(g)) + "}" if len(g) > 1 else next(iter(g)) for g in groups)


def run_variant(name, rows, rows_idx, rob, perm_global, boot=True):
    timing = get_timing(name)
    res = {"variante": name, "description": timing.note, "n_taches_mesurees": len(timing.wall),
           "n_taches_faible_precision": len(timing.low_precision_tasks())}
    if hasattr(timing, "n_removed"):
        res.update({"n_executions_retirees": timing.n_removed, "n_blocs_retires": timing.n_blocks_removed,
                    "n_taches_sans_aucune_repetition_restante": len(timing.tasks_lost)})
    # n*
    b1 = analyse(timing.wall, LOW_PREC_S, np.random.default_rng(SEED_BOOT_B1), name, rows=rows, boot=boot)
    res["n_star"] = b1.get("n_star_exact")
    res["n_star_IC95_variante"] = [b1.get("n_star_IC95_bas_exact"), b1.get("n_star_IC95_haut_exact")]
    res["n_star_bootstrap_part_croisement"] = b1.get("bootstrap_part_replicats_avec_croisement_montant")
    res["n_exploitables"] = b1["n_exploitables"]
    res["dimensions_retenues"] = b1["dimensions_retenues"]
    res["par_dimension"] = b1["par_dimension"]
    # front de Pareto (pente du temps reel par iteration)
    scal, pareto = front_for_variant(rows, timing.wall, rob, perm_global)
    res["pareto"] = pareto
    res["front"] = sorted(r["method"] for r in pareto if r["sur_le_front"])
    res["solveurs_pente_estimable"] = sorted(r["method"] for r in pareto)
    # compromis (quatre regles) : n* non determinable -> seuil +infini (convention)
    loo = run_loo(timing, rows, rows_idx, with_ci=False)
    res["plis_ecartes"] = loo["skipped"]
    comp = {s["regle"]: s for s in loo["synth"] if s["portee"] == "toutes"}
    res["compromis"] = {r: {k: comp[r][k] for k in ("perte_moyenne", "taux_accord", "regret_mesure_seul_geo",
                                                    "n_pb_regret_mesure_seul", "n_inst_regret_mesure_seul",
                                                    "regret_mesure_estime_geo", "n_pb_regret_mesure_estime",
                                                    "n_inst_regret_mesure_estime", "regret_mesure_seul_commun_geo",
                                                    "n_pb_commun", "n_inst_commun")} for r in RULES}
    res["n_star_plein_echantillon_utilise"] = None if not np.isfinite(loo["n_star_full"]) else loo["n_star_full"]
    res["plis_avec_n_star_defini"] = sum(1 for f in loo["fold_rows"] if not str(f["n_star_pli"]).startswith("non"))
    res["ordre_perte"] = fmt_order(weak_order({r: comp[r]["perte_moyenne"] for r in RULES}))
    res["ordre_regret_mesure_seul"] = fmt_order(weak_order({r: comp[r]["regret_mesure_seul_geo"] for r in RULES}))
    res["ordre_regret_mesure_estime_info"] = fmt_order(weak_order({r: comp[r]["regret_mesure_estime_geo"] for r in RULES}))
    res["ordre_regret_commun_info"] = fmt_order(weak_order({r: comp[r]["regret_mesure_seul_commun_geo"] for r in RULES}))
    return res


def verdict(main, var):
    out = {"variante": var["variante"]}
    n = var["n_star"]
    out["n_star"] = n
    if main["n_star"] is None:
        out["critere_i"] = "(i') n* non determinable dans la variante, comme dans l'analyse principale"
        out["i_n_star_dans_IC_principale"] = (n is None)
    else:
        lo, hi = main["n_star_IC95_variante"]
        out["critere_i"] = "(i) n* dans l'IC95 de l'analyse principale"
        out["i_n_star_dans_IC_principale"] = (n is not None) and (lo <= n <= hi)
    out["ii_meme_front"] = var["front"] == main["front"]
    out["front"] = var["front"]
    out["iii_ordre_perte_inchange"] = var["ordre_perte"] == main["ordre_perte"]
    out["ordre_perte"] = var["ordre_perte"]
    # rapporte sans critere
    out["ordre_regret_mesure_seul_rapporte_sans_critere"] = var["ordre_regret_mesure_seul"]
    out["ordre_regret_mesure_seul_identique_a_la_principale"] = var["ordre_regret_mesure_seul"] == main["ordre_regret_mesure_seul"]
    crit = [out["i_n_star_dans_IC_principale"], out["ii_meme_front"], out["iii_ordre_perte_inchange"]]
    out["robuste"] = all(c is True for c in crit)
    return out


def main():
    os.makedirs(ROB_DIR, exist_ok=True)
    rows = load_distinct_rows()
    rows_idx = index_rows(rows)
    rob, perm_global = load_robustness()
    med_c, exc = control_excursions()

    results = {}
    for v in VARIANTS + DIAGNOSTICS:
        results[v] = run_variant(v, rows, rows_idx, rob, perm_global)
        print(f"[{v}] n*={results[v]['n_star']} front={results[v]['front']} perte: {results[v].get('ordre_perte')} "
              f"| regret mesure seul: {results[v].get('ordre_regret_mesure_seul')}", flush=True)
    main_res = results["principale"]
    with open(os.path.join(OUT_DIR, "b1_seuil_noeud2.json"), encoding="utf-8") as fh:
        b1 = json.load(fh)["principale"]
    if b1.get("n_star_exact") is None:
        assert main_res["n_star"] is None
    else:
        assert abs(b1["n_star_exact"] - main_res["n_star"]) < 1e-9
        main_res["n_star_IC95_variante"] = [b1["n_star_IC95_bas_exact"], b1["n_star_IC95_haut_exact"]]

    verdicts = {v: verdict(main_res, results[v]) for v in VARIANTS[1:] + DIAGNOSTICS}
    regle_ok = all(verdicts[v]["robuste"] for v in ("R1", "R2", "R3"))
    summary = {"n_star_principale": main_res["n_star"], "IC95_n_star_principale": main_res["n_star_IC95_variante"],
               "front_principale": main_res["front"], "ordre_perte_principale": main_res["ordre_perte"],
               "ordre_regret_mesure_seul_principale_rapporte_sans_critere": main_res["ordre_regret_mesure_seul"],
               "verdicts": verdicts,
               "conclusions_robustes_selon_la_regle_fixee_a_l_avance": regle_ok,
               "regle": "R1, R2 (lecture 'l'une des deux') et R3 : (i/i') n* ; (ii) meme front ; (iii) meme ordre sur la perte de succes ; "
                        "regret rapporte sans critere ; adaptation (i') consignee dans regle_adaptee_prealable.md",
               "mesure_de_controle_passage_A": {"mediane_s": med_c, "n_hors_10pct": sum(1 for x in exc.values() if x[1]),
                                                "n_mesures": len(exc)}}
    with open(os.path.join(ROB_DIR, "verdict_robustesse.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    with open(os.path.join(ROB_DIR, "variantes_detail.json"), "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False, default=str)

    with open(os.path.join(ROB_DIR, "variantes_synthese.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["variante", "n_taches_mesurees", "n_taches_faible_precision", "n_executions_retirees", "n_blocs_retires",
                    "n_exploitables_n_star", "n_star", "part_bootstrap_avec_croisement", "n_star_defini_critere_i",
                    "front", "meme_front", "ordre_perte", "ordre_perte_inchange", "ordre_regret_mesure_seul_sans_critere",
                    "robuste_selon_la_regle"])
        for v in VARIANTS + DIAGNOSTICS:
            r = results[v]
            vd = verdicts.get(v)
            w.writerow([v, r["n_taches_mesurees"], r["n_taches_faible_precision"], r.get("n_executions_retirees", 0),
                        r.get("n_blocs_retires", 0), r["n_exploitables"], "non determinable" if r["n_star"] is None else round(r["n_star"], 2),
                        "" if r["n_star_bootstrap_part_croisement"] is None else round(r["n_star_bootstrap_part_croisement"], 4),
                        vd["i_n_star_dans_IC_principale"] if vd else "reference",
                        ";".join(r["front"]), vd["ii_meme_front"] if vd else "reference", r["ordre_perte"],
                        vd["iii_ordre_perte_inchange"] if vd else "reference", r["ordre_regret_mesure_seul"],
                        vd["robuste"] if vd else "reference"])
    with open(os.path.join(ROB_DIR, "variantes_ratio_par_dimension.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["variante", "n", "n_problemes", "retenue", RATIO_KEY, "IC95_bas", "IC95_haut"])
        for v in VARIANTS + DIAGNOSTICS:
            for n, d in results[v]["par_dimension"].items():
                w.writerow([v, n, d["n_problemes"], int(d["retenue"]), d[RATIO_KEY], d["IC95_bas"], d["IC95_haut"]])
    with open(os.path.join(ROB_DIR, "variantes_pareto.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["variante", "method", "taux_succes", "pente_temps_par_iteration", "IC95_bas_pente", "IC95_haut_pente",
                    "n_problemes_pente", "sur_le_front", "domine_par"])
        for v in VARIANTS + DIAGNOSTICS:
            for r in results[v]["pareto"]:
                w.writerow([v, r["method"], r["taux_succes"], round(r["pente_temps_par_iteration"], 4),
                            round(r["IC95_bas_pente"], 4), round(r["IC95_haut_pente"], 4), r["n_problemes_pente"],
                            r["sur_le_front"], r["domine_par"]])
    with open(os.path.join(ROB_DIR, "variantes_compromis.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["variante", "regle", "perte_moyenne", "regret_mesure_seul_geo", "n_pb_regret_mesure_seul",
                    "n_inst_regret_mesure_seul", "regret_mesure_estime_geo", "n_pb_regret_mesure_estime",
                    "regret_mesure_seul_sous_ensemble_commun_geo", "n_pb_commun", "n_inst_commun"])
        for v in VARIANTS + DIAGNOSTICS:
            c = results[v].get("compromis")
            if not c:
                continue
            for r in RULES:
                x = c[r]
                w.writerow([v, r, round(x["perte_moyenne"], 4), round(x["regret_mesure_seul_geo"], 4),
                            x["n_pb_regret_mesure_seul"], x["n_inst_regret_mesure_seul"],
                            round(x["regret_mesure_estime_geo"], 4), x["n_pb_regret_mesure_estime"],
                            "" if x["regret_mesure_seul_commun_geo"] is None else round(x["regret_mesure_seul_commun_geo"], 4),
                            x["n_pb_commun"], x["n_inst_commun"]])
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
