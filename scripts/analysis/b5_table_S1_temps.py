# -*- coding: utf-8 -*-
"""
Consigne 009e (reprise de 008d-bis), etape 3 (S1) : table S1 par probleme (11 problemes x 5 dimensions, format valide en
008b/008d) reconstruite depuis resultats_benchmark_v0.5.csv, avec la colonne
de TEMPS REEL median de Newton-LM (nuit v0.5, passage A) et la marque "faible precision" lorsque ce temps est < 1 ms.
"""
import csv
import os
import statistics as st

from common_timing import get_timing, load_timing_rows, LOW_PREC_S
from paths import OUT_DIR, BENCH_CSV

BENCH_V05 = BENCH_CSV
FIVE_DIM_CAT_A = ["A01", "A02", "A03", "A04", "A05", "A07", "A09", "A10", "A12", "A14", "A15"]


def main():
    with open(BENCH_V05, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    wall = get_timing("principale").wall
    n_rep = {}
    for r in load_timing_rows():
        k = (r["problem"], r["n"], r["method"])
        n_rep[k] = n_rep.get(k, 0) + 1
    catA_nb = [r for r in rows if r["category"] == "A" and r["method"] == "Newton_BT"]
    dims = {p: sorted({r["n"] for r in catA_nb if r["problem"] == p}, key=int) for p in FIVE_DIM_CAT_A}

    out, n_low = [], 0
    for rank in range(5):
        for p in FIVE_DIM_CAT_A:
            n = dims[p][rank]
            r = next(rr for rr in catA_nb if rr["problem"] == p and rr["n"] == n)
            ok = r["status"] == "success"
            t = wall.get((p, n, "Newton_BT"))
            low = t is not None and t < LOW_PREC_S
            n_low += low
            out.append({"rang": rank + 1, "problem": p, "n": n, "n_iter": r["n_iter"] if ok else "ECHEC",
                        "status": r["status"],
                        "note": "A09 : n=8 (rang1) / n=48 (rang2), pas 10/50" if p == "A09" and rank < 2 else "",
                        "n_total_v05": r["n_total"] if ok else "",
                        "temps_reel_median_NewtonLM_s": f"{t:.6g}" if t is not None else "non determinable",
                        "faible_precision": "oui (< 1 ms)" if low else "non",
                        "n_repetitions": n_rep.get((p, n, "Newton_BT"), 0)})
    with open(OUT_DIR + r"\b5_table_S1_avec_temps_reel.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)

    synth = []
    for rank in range(5):
        cells = [r for r in out if r["rang"] == rank + 1]
        vals = [int(r["n_iter"]) for r in cells if r["status"] == "success"]
        nts = [int(r["n_total_v05"]) for r in cells if r["status"] == "success"]
        ts = [float(r["temps_reel_median_NewtonLM_s"]) for r in cells]
        synth.append({"rang": rank + 1, "n_succes": len(vals), "n_total": len(cells),
                      "mediane_n_iter": st.median(vals) if vals else None,
                      "mediane_N_tot_v05": st.median(nts) if nts else None,
                      "mediane_temps_reel_s": f"{st.median(ts):.6g}",
                      "cellules_faible_precision": sum(1 for r in cells if r["faible_precision"] != "non")})
    with open(OUT_DIR + r"\b5_table_S1_synthese.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(synth[0].keys()))
        w.writeheader()
        w.writerows(synth)
    print(f"{len(out)} cellules, dont {n_low} faible precision")
    for s in synth:
        print(s)
    for r in out:
        print(f"  rang={r['rang']} {r['problem']:4s} n={r['n']:>5s} n_iter={r['n_iter']:>6s} "
              f"t={r['temps_reel_median_NewtonLM_s']:>10s} s  {r['faible_precision']}")


if __name__ == "__main__":
    main()
