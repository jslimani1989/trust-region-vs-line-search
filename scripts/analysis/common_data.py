# -*- coding: utf-8 -*-
"""
Module partage (consigne 009e, repris de 008d-bis) : chargement des donnees et
calcul de s(p, m) = proportion des instances distinctes du probleme p resolues
par la methode m. Unite statistique = le probleme (36 grappes : 15 A, 11 B, 10 C).
Aucune execution de solveur ; lecture seule de resultats_benchmark_v0.5.csv
(v0.5 : 1080 lignes, colonne code_version ; premiere occurrence de chaque instance distincte).
"""
import csv

from paths import BENCH_CSV

V02_CSV = BENCH_CSV  # nom historique (008d)

METHODS = ["Cauchy_Point", "Dogleg", "Steihaug_CG", "BFGS_Armijo", "BFGS_Wolfe", "Newton_BT"]
CATEGORIES = ["A", "B", "C"]

_CACHE = {}


def load_distinct_rows():
    if "rows" not in _CACHE:
        with open(V02_CSV, encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        idx = {}
        for r in rows:
            idx.setdefault((r["problem"], r["n"], r["method"]), r)
        _CACHE["rows"] = list(idx.values())
    return list(_CACHE["rows"])


def problems_by_category(rows):
    cats = {}
    for r in rows:
        cats.setdefault(r["category"], set()).add(r["problem"])
    return {c: sorted(cats[c]) for c in cats}


def success_matrix(rows):
    """Retourne s[p][m] = proportion d'instances distinctes de p resolues par m,
    et n_instances[p] = nombre d'instances distinctes de p."""
    by_pm = {}
    for r in rows:
        key = (r["problem"], r["method"])
        by_pm.setdefault(key, []).append(r["status"] == "success")
    n_inst = {}
    for r in rows:
        n_inst.setdefault(r["problem"], set()).add(r["n"])
    n_inst = {p: len(v) for p, v in n_inst.items()}

    s = {}
    for (p, m), vals in by_pm.items():
        s.setdefault(p, {})[m] = sum(vals) / len(vals)
    return s, n_inst


def ntot_by_instance(rows):
    """Retourne ntot[p][n][m] = N_tot, et succes[p][n][m] = bool."""
    ntot = {}
    succ = {}
    for r in rows:
        ntot.setdefault(r["problem"], {}).setdefault(r["n"], {})[r["method"]] = int(r["n_total"])
        succ.setdefault(r["problem"], {}).setdefault(r["n"], {})[r["method"]] = (r["status"] == "success")
    return ntot, succ


def all_problems(rows):
    return sorted({r["problem"] for r in rows})
