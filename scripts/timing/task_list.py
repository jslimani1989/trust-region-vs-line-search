# -*- coding: utf-8 -*-
"""
Consigne 009d, etape 7 : MEMES blocs que 009c (A3, A3bis, A3ter ; 784 taches), code fige du tag v0.5.
(Historique : consigne 008e, reprise a l'identique de 008c, A3/A3bis) : enumeration des
taches distinctes a chronometrer. Construit directement a partir du registre
de problemes du code fige (tag v0.3.1, instrumentation en moins), independamment
de resultats_benchmark_v0.3.csv (dont les temps ne doivent pas etre utilises).

A3    : Categorie A, TOUTES les instances (problem,n) distinctes, 6 solveurs.
A3bis : Categories B et C, instances (problem,n) avec n>=500 seulement,
        Newton_BT et Steihaug_CG seulement.
A3ter : (009c, nouveau) Categories B et C, instances (problem,n) avec n<500,
        les 6 solveurs.
"""
import os
import sys

WORKTREE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # racine du depot (code des solveurs)

METHODS_A3 = ["Cauchy_Point", "Dogleg", "Steihaug_CG", "BFGS_Armijo", "BFGS_Wolfe", "Newton_BT"]
METHODS_A3BIS = ["Newton_BT", "Steihaug_CG"]
DIMS_NOMINAL = [10, 50, 100, 500, 1000]
A3BIS_MIN_N = 500


def _fw():
    if WORKTREE not in sys.path:
        sys.path.insert(0, WORKTREE)
    import framework_united_final as fw
    return fw


def distinct_instances(fw, categories):
    """Deduplique la grille nominale (certains problemes a dimension fixe
    apparaissent plusieurs fois pour les memes (pid,n))."""
    specs = fw.build_problem_specs(dims=DIMS_NOMINAL, categories=categories)
    seen = {}
    for pid, n in specs:
        cat = pid[0].upper()
        seen.setdefault((pid, n), cat)
    return sorted(seen.items())  # [((pid,n), cat), ...] trie pour reproductibilite avant tirage au hasard


def build_tasks():
    """Retourne la liste des taches distinctes (dicts), bloc A3 et A3bis."""
    fw = _fw()
    tasks = []

    catA = distinct_instances(fw, ["A"])
    assert len(catA) == 59, f"attendu 59 instances Cat.A distinctes, obtenu {len(catA)}"
    for (pid, n), cat in catA:
        for m in METHODS_A3:
            tasks.append({"bloc": "A3", "problem": pid, "n": n, "category": cat, "method": m})
    assert len([t for t in tasks if t["bloc"] == "A3"]) == 354

    catBC = distinct_instances(fw, ["B", "C"])
    catBC_ge500 = [((pid, n), cat) for (pid, n), cat in catBC if n >= A3BIS_MIN_N]
    for (pid, n), cat in catBC_ge500:
        for m in METHODS_A3BIS:
            tasks.append({"bloc": "A3bis", "problem": pid, "n": n, "category": cat, "method": m})
    n_a3bis = len([t for t in tasks if t["bloc"] == "A3bis"])

    # 009c : A3ter = Categories B et C, TOUTES les instances a n < 500, 6 solveurs
    catBC_lt500 = [((pid, n), cat) for (pid, n), cat in catBC if n < A3BIS_MIN_N]
    for (pid, n), cat in catBC_lt500:
        for m in METHODS_A3:
            tasks.append({"bloc": "A3ter", "problem": pid, "n": n, "category": cat, "method": m})
    n_a3ter = len([t for t in tasks if t["bloc"] == "A3ter"])
    assert len(catBC_lt500) + len(catBC_ge500) == len(catBC), "B/C : n<500 et n>=500 doivent partitionner les instances"
    assert n_a3ter == 6 * len(catBC_lt500)
    print(f"A3 : {len([t for t in tasks if t['bloc']=='A3'])} taches (59 instances x 6 solveurs)")
    print(f"A3bis : {n_a3bis} taches ({len(catBC_ge500)} instances B/C n>=500 x 2 solveurs)")
    print(f"A3ter : {n_a3ter} taches ({len(catBC_lt500)} instances B/C n<500 x 6 solveurs)")

    for t in tasks:
        t["task_id"] = f"{t['bloc']}|{t['problem']}|{t['n']}|{t['method']}"
    return tasks


if __name__ == "__main__":
    tasks = build_tasks()
    print(f"Total taches distinctes : {len(tasks)}")
