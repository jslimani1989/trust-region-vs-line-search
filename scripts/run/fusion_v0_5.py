# -*- coding: utf-8 -*-
"""
Consigne 009d, Etape 5.1 (fin) : construit resultats_benchmark_v0.5.csv = lignes v0.4 de Cauchy Point et de
Steihaug-CG (code inchange) + les 720 nouvelles lignes des quatre solveurs modifies, avec la colonne `code_version`
("v0.4" pour les 360 lignes reprises, "v0.5" pour les 720 nouvelles). Controles : 1080 lignes, memes creneaux que
resultats_benchmark_v0.4.csv (doublons de creneaux compris), aucune ligne manquante. Lecture seule des deux entrees.
"""
import csv
import os
import sys
from collections import defaultdict, deque

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # racine du depot
V04 = os.path.join(REPO, "results", "reference", "resultats_benchmark_v0.4.csv")
NEW = os.path.join(REPO, "rerun", "resultats_benchmark_v0.5_nouvelles_lignes.csv")
OUT = os.path.join(REPO, "rerun", "resultats_benchmark_v0.5.csv")  # a comparer a results/resultats_benchmark_v0.5.csv
UNCHANGED = {"Cauchy_Point", "Steihaug_CG"}


def main():
    with open(V04, encoding="utf-8") as fh:
        rd = csv.DictReader(fh)
        fields04 = rd.fieldnames
        rows04 = list(rd)
    with open(NEW, encoding="utf-8") as fh:
        new = list(csv.DictReader(fh))
    queues = defaultdict(deque)
    for r in new:
        queues[(r["problem"], r["n"], r["method"])].append(r)
    out = []
    for r in rows04:
        k = (r["problem"], r["n"], r["method"])
        if r["method"] in UNCHANGED:
            r2 = dict(r)
            r2["code_version"] = "v0.4"
        else:
            if not queues[k]:
                sys.exit(f"ARRET : ligne v0.5 manquante pour {k}")
            r2 = dict(queues[k].popleft())
            r2["code_version"] = "v0.5"
        out.append(r2)
    left = sum(len(q) for q in queues.values())
    if left:
        sys.exit(f"ARRET : {left} nouvelles lignes sans creneau correspondant")
    assert len(out) == 1080, len(out)
    fields = list(fields04) + ["code_version"]
    with open(OUT, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(out)
    n04 = sum(1 for r in out if r["code_version"] == "v0.4")
    print(f"{len(out)} lignes ecrites dans {OUT} ({n04} v0.4 : Cauchy Point et Steihaug-CG ; {len(out) - n04} v0.5)")


if __name__ == "__main__":
    main()
