# -*- coding: utf-8 -*-
"""
Consigne 008c, B2 : structure de l'arbre (identique a 007/tree_model.py dans
sa forme -- 3 noeuds, 5 feuilles), mais seuil du noeud 2 parametrable (n*,
issu de B1, au lieu de la valeur provisoire 500 de 007) et regle
d'apprentissage des feuilles refaite (permutation + Holm + depart par le
temps, cf. b2_arbre_decision_temps.py).
"""
NO_HESS = ["BFGS_Armijo", "BFGS_Wolfe"]
HESS = ["Cauchy_Point", "Dogleg", "Steihaug_CG", "Newton_BT"]

LEAVES = ["L1_pas_hessienne", "L2_hess_abordable_AB", "L3_hess_abordable_C",
          "L4_hess_nonabordable_AB", "L5_hess_nonabordable_C"]


def leaf_for(category, n, threshold_n):
    afford = int(n) < threshold_n
    indefinite = (category == "C")
    if afford and not indefinite:
        leaf_hess = "L2_hess_abordable_AB"
    elif afford and indefinite:
        leaf_hess = "L3_hess_abordable_C"
    elif not afford and not indefinite:
        leaf_hess = "L4_hess_nonabordable_AB"
    else:
        leaf_hess = "L5_hess_nonabordable_C"
    return "L1_pas_hessienne", leaf_hess


def build_leaf_instances(rows, threshold_n, exclude_problem=None):
    leaf_data = {leaf: {"instances": []} for leaf in LEAVES}
    seen = set()
    for r in rows:
        p, n = r["problem"], r["n"]
        if exclude_problem is not None and p == exclude_problem:
            continue
        key = (p, n)
        if key in seen:
            continue
        seen.add(key)
        leaf_nohess, leaf_hess = leaf_for(r["category"], n, threshold_n)
        leaf_data[leaf_nohess]["instances"].append((p, n))
        leaf_data[leaf_hess]["instances"].append((p, n))
    leaf_data["L1_pas_hessienne"]["candidats"] = NO_HESS
    for leaf in LEAVES[1:]:
        leaf_data[leaf]["candidats"] = HESS
    return leaf_data
