# -*- coding: utf-8 -*-
"""
Consigne 012, etape 5 : comparaison indicateur par indicateur entre 009e (= variante sans exclusion, reproduite a
l'identique) et les variantes V1 a V4. Lecture seule des sorties ; aucun calcul statistique nouveau (seulement des
comparaisons de valeurs deja calculees : cote du seuil 0,05, ordre, position d'un IC par rapport a 1 ou a 0, front).

Regle « conclusion changee » (consigne 012) : une paire change de cote du seuil 0,05 de Holm ; l'ordre des solveurs
change ; un IC change de position par rapport a 1 ; un solveur entre dans le front ou en sort.
"""
import csv
import json
import os

# Depot publie (v0.5.2) : sorties sous ANALYSIS_OUT (defaut : analysis_output/ a la racine du depot).
_OUT = os.environ.get("ANALYSIS_OUT", os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "analysis_output"))
BASE = os.path.join(_OUT, "sensitivity")  # variantes
REF = _OUT  # sorties de la chaine principale (009e)
VARIANTS = {"V1": "excl_B11_DixonPrice", "V2": "excl_A14", "V3": "excl_A14_B11_DixonPrice",
            "V4": "excl_B3_Ackley_B5_Griewank_B10_Schwefel_C04"}
COLS = ["009e", "V1", "V2", "V3", "V4"]
METHODS = ["Cauchy_Point", "Dogleg", "Steihaug_CG", "BFGS_Armijo", "BFGS_Wolfe", "Newton_BT"]


def path(col, f):
    return os.path.join(REF if col == "009e" else os.path.join(BASE, VARIANTS[col]), f)


def rcsv(col, f):
    with open(path(col, f), encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def rjson(col, f):
    with open(path(col, f), encoding="utf-8") as fh:
        return json.load(fh)


out = []


def add(famille, indicateur, vals, conclusions, regle):
    """vals : {col: texte} ; conclusions : {col: valeur comparable} ; changee si une variante differe de 009e."""
    ch = [c for c in COLS[1:] if conclusions[c] != conclusions["009e"]]
    out.append({"famille": famille, "indicateur": indicateur, **{c: vals[c] for c in COLS},
                "regle": regle, "conclusion_changee": "oui (" + ", ".join(ch) + ")" if ch else "non"})


def side(lo, hi, ref):
    lo, hi = float(lo), float(hi)
    return "<" if hi < ref else (">" if lo > ref else "contient")


def ranking(d):
    """Ordre decroissant ; les ex aequo forment un meme groupe (aucun changement d'ordre artificiel)."""
    vals = sorted(set(d.values()), reverse=True)
    return " > ".join("=".join(sorted(m for m in d if d[m] == v)) for v in vals)


# 1. Taux de succes moyens par probleme (IC BCa) + ordre des solveurs
data = {c: {(r["method"], r["scope"]): r for r in rcsv(c, "etape5_2_IC_taux_succes.csv")} for c in COLS}
for scope in ["global", "A", "B", "C"]:
    for m in METHODS:
        vals = {c: f'{data[c][(m, scope)]["taux_succes_moyen_par_probleme"]} [{data[c][(m, scope)]["IC95_bas"]} ; {data[c][(m, scope)]["IC95_haut"]}] ({data[c][(m, scope)]["n_problemes"]} pb)' for c in COLS}
        add("taux moyen par probleme (BCa)", f"{m} | {scope}", vals, {c: 0 for c in COLS}, "descriptif (l'ordre est juge ci-dessous)")
    ranks = {c: ranking({m: float(data[c][(m, scope)]["taux_succes_moyen_par_probleme"]) for m in METHODS}) for c in COLS}
    add("ordre des solveurs", f"taux moyen par probleme | {scope}", ranks, ranks, "ordre des solveurs")

# 2. Taux en proportion des instances distinctes + ordre
data = {c: {(r["solveur"], r["portee"]): r for r in rcsv(c, "p1_table_succes_156_instances.csv")} for c in COLS}
for scope in ["global", "A", "B", "C"]:
    vals = {c: "; ".join(f'{m}={data[c][(m, scope)]["n_succes_v0.5"]}/{data[c][(m, scope)]["n_instances"]}' for m in METHODS) for c in COLS}
    add("succes en proportion des instances distinctes", scope, vals, {c: 0 for c in COLS}, "descriptif")
    ranks = {c: ranking({m: float(data[c][(m, scope)]["proportion_v0.5"]) for m in METHODS}) for c in COLS}
    add("ordre des solveurs", f"proportion des instances distinctes | {scope}", ranks, ranks, "ordre des solveurs")

# 3. Permutation + Holm (15 paires x 4 familles)
data = {c: {(r["famille"], r["solveur_a"], r["solveur_b"]): r for r in rcsv(c, "etape5_1_tests_permutation.csv")} for c in COLS}
for k in data["009e"]:
    vals = {c: f'p_holm={float(data[c][k]["p_holm"]):.4g} ({data[c][k]["n_problemes"]} pb)' for c in COLS}
    concl = {c: float(data[c][k]["p_holm"]) < 0.05 for c in COLS}
    add("permutation, Holm", f"{k[0]} | {k[1]} vs {k[2]}", vals, concl, "cote du seuil 0,05 (Holm)")

# 4. IC des ecarts de taux entre paires (position par rapport a 0, equivalent additif de « par rapport a 1 »)
data = {c: {(r["scope"], r["solveur_a"], r["solveur_b"]): r for r in rcsv(c, "etape5_2_IC_ecarts_paires.csv")} for c in COLS}
for k in data["009e"]:
    vals = {c: f'{data[c][k]["mean_d"]} [{data[c][k]["IC95_bas"]} ; {data[c][k]["IC95_haut"]}]' for c in COLS}
    concl = {c: side(data[c][k]["IC95_bas"], data[c][k]["IC95_haut"], 0.0) for c in COLS}
    add("IC des ecarts de taux (BCa)", f"{k[0]} | {k[1]} - {k[2]}", vals, concl, "position de l'IC par rapport a 0")

# 5. Couts N_tot des 15 paires (etape 5.3, global et par categorie) + T-L1 (tl1)
data = {c: {(r["scope"], r["solveur_a"], r["solveur_b"]): r for r in rcsv(c, "etape5_3_couts.csv")} for c in COLS}
for k in data["009e"]:
    def fmt(c):
        r = data[c].get(k)
        return "non calculable" if r is None or not r["ratio_geo_moyen"] else f'{r["ratio_geo_moyen"]} [{r["IC95_bas"]} ; {r["IC95_haut"]}] ({r["n_problemes_utilises"]} pb, {r["n_instances_utilisees"]} inst.)'
    vals = {c: fmt(c) for c in COLS}
    concl = {c: (side(data[c][k]["IC95_bas"], data[c][k]["IC95_haut"], 1.0) if data[c].get(k) and data[c][k]["IC95_bas"] else "n.c.") for c in COLS}
    add("cout N_tot (etape 5.3)", f"{k[0]} | {k[1]} / {k[2]}", vals, concl, "position de l'IC par rapport a 1")
data = {c: {(r["solveur_a"], r["solveur_b"]): r for r in rcsv(c, "tl1_ntot_15paires_recalcule.csv")} for c in COLS}
for k in data["009e"]:
    vals = {c: f'{data[c][k]["ratio_geo_moyen_v05"]} [{data[c][k]["IC95_bas_v05"]} ; {data[c][k]["IC95_haut_v05"]}] ({data[c][k]["n_problemes_utilises"]} pb)' for c in COLS}
    concl = {c: side(data[c][k]["IC95_bas_v05"], data[c][k]["IC95_haut_v05"], 1.0) for c in COLS}
    add("cout N_tot (procedure T-L1)", f"global | {k[0]} / {k[1]}", vals, concl, "position de l'IC par rapport a 1")

# 6. Temps Newton-LM / Steihaug-CG par dimension et n* (V2, V3 seulement ; V1 et V4 : Categorie A inchangee)
data = {c: {r["n"]: r for r in rcsv(c, "b1_ratio_par_dimension.csv") if r["retenue"] == "1"} for c in COLS}
for n in sorted(data["009e"], key=int):
    def fmt(c):
        r = data[c].get(n)
        return "non retenue" if r is None else f'{r["ratio_geo_moyen_NewtonLM_sur_SteihaugCG"]} [{r["IC95_bas"]} ; {r["IC95_haut"]}] ({r["n_problemes"]} pb)'
    vals = {c: fmt(c) for c in COLS}
    concl = {c: (side(data[c][n]["IC95_bas"], data[c][n]["IC95_haut"], 1.0) if n in data[c] else "non retenue") for c in COLS}
    add("temps reel Newton-LM / Steihaug-CG", f"n = {n}", vals, concl, "position de l'IC par rapport a 1")
b1 = {c: rjson(c, "b1_seuil_noeud2.json")["principale"] for c in COLS}
vals = {c: ("non determinable" if b1[c]["n_star_exact"] is None else str(b1[c]["n_star_exact"])) +
           f' (croisement dans {b1[c]["bootstrap_part_replicats_avec_croisement_montant"]} des reechantillons ; {b1[c]["n_exploitables"]} inst. exploitables)' for c in COLS}
add("seuil n*", "n* (Categorie A)", vals, {c: b1[c]["n_star_exact"] is None for c in COLS}, "determinable ou non")

# 7. Arbre appris (feuilles)
data = {c: {r["leaf"]: r for r in rcsv(c, "b2_feuilles.csv")} for c in COLS}
for leaf in data["009e"]:
    vals = {c: f'{data[c][leaf]["recommande"]} ({data[c][leaf]["n_instances"]} inst., {data[c][leaf]["n_problemes"]} pb ; survivants {data[c][leaf]["survivants_delta_0.05"]})' for c in COLS}
    add("arbre appris (delta = 0,05)", leaf, vals, {c: data[c][leaf]["recommande"] for c in COLS}, "solveur recommande (ordre)")

# 8. Validation hors echantillon : perte et regret par regle, ordre des regles
data = {c: {r["regle"]: r for r in rcsv(c, "b3_synthese.csv") if r["portee"] == "toutes"} for c in COLS}
for rule in data["009e"]:
    vals = {c: f'perte {data[c][rule]["perte_moyenne"]} {data[c][rule]["perte_IC95"]} ; regret mesure seul {data[c][rule]["regret_mesure_seul_geo"]} {data[c][rule]["regret_mesure_seul_IC95"]} ; regret mesure+estime {data[c][rule]["regret_mesure_estime_geo"]} {data[c][rule]["regret_mesure_estime_IC95"]}' for c in COLS}
    add("validation hors echantillon", rule, vals, {c: 0 for c in COLS}, "descriptif (l'ordre est juge ci-dessous)")
for key, lab in [("perte_moyenne", "perte de succes"), ("regret_mesure_seul_geo", "regret mesure seul"),
                 ("regret_mesure_estime_geo", "regret mesure + estime")]:
    ranks = {c: ranking({r: -float(data[c][r][key]) for r in data[c]}) for c in COLS}  # meilleur (plus petit) en tete
    add("ordre des regles", lab, ranks, ranks, "ordre des regles (meilleure en tete)")

# 9. Front de Pareto
data = {c: {r["method"]: r for r in rcsv(c, "b4_pareto_front.csv")} for c in COLS}
for m in METHODS:
    vals = {c: f'succes {data[c][m]["taux_succes"]} ; pente {data[c][m]["pente_temps_par_iteration"]} [{data[c][m]["IC95_bas_pente"]} ; {data[c][m]["IC95_haut_pente"]}] ; front={data[c][m]["sur_le_front"]}' for c in COLS}
    add("front de Pareto", m, vals, {c: data[c][m]["sur_le_front"] for c in COLS}, "entree ou sortie du front")

# 10. Supplement : McNemar exact + Holm (instances distinctes)
data = {c: {(r["solveur_a"], r["solveur_b"]): r for r in rcsv(c, "etape5_4_mcnemar.csv")} for c in COLS}
for k in data["009e"]:
    vals = {c: f'p_holm={float(data[c][k]["p_holm"]):.4g} (n01={data[c][k]["n01_b_seul"]}, n10={data[c][k]["n10_a_seul"]})' for c in COLS}
    add("McNemar exact, Holm (supplement)", f"{k[0]} vs {k[1]}", vals, {c: float(data[c][k]["p_holm"]) < 0.05 for c in COLS}, "cote du seuil 0,05 (Holm)")

# 11. Departs multiples (hors liste de l'etape 3 : informatif ; B11 et C04 figurent dans le sous-ensemble)
data = {c: {r["method"]: r for r in rcsv(c, "p5_departs_multiples_global.csv")} for c in COLS}
vals = {c: "; ".join(f'{m}={data[c][m]["n_succes_v0.5"]}/{data[c][m]["n_departs"]}' for m in METHODS) for c in COLS}
add("departs multiples (informatif)", "succes global", vals, {c: 0 for c in COLS}, "descriptif")
ranks = {c: ranking({m: float(data[c][m]["taux_v0.5"]) for m in METHODS}) for c in COLS}
add("departs multiples (informatif)", "ordre des solveurs (global)", ranks, ranks, "ordre des solveurs")

with open(os.path.join(BASE, "comparaison_variantes.csv"), "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=list(out[0].keys()), delimiter=";")
    w.writeheader()
    w.writerows(out)
ch = [r for r in out if r["conclusion_changee"] != "non"]
print(f"{len(out)} indicateurs ; conclusions changees : {len(ch)}")
for r in ch:
    print(f'- [{r["famille"]}] {r["indicateur"]} -> {r["conclusion_changee"]}')
    for c in COLS:
        print(f'    {c}: {r[c]}')
