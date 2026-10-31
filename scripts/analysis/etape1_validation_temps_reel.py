# -*- coding: utf-8 -*-
"""
Consigne 009e (reprise de 008d-bis), etape 1 (reprise de 008d, etape 1) sur les mesures v0.5 (passage A).
1. Validation : sur les taches dont le temps CPU MEDIAN depasse 0,1 s, rapport temps reel median /
   temps CPU median (mediane, 5e et 95e centiles). Mediane hors [0,95 ; 1,05] -> ARRET.
2. Statistique retenue = mediane des repetitions du temps reel ; taches dont cette mediane < 1 ms =
   "faible precision".
Le rapport 009d (par execution, > 0,1 s de temps reel) est rappele pour comparaison.
"""
import csv
import json
import os
import sys

import numpy as np

from common_timing import get_timing, load_timing_rows, LOW_PREC_S
from paths import OUT_DIR


def main():
    timing = get_timing("principale")
    rows = load_timing_rows()
    wall, cpu = timing.wall, timing.cpu
    tasks = sorted(wall)

    ratios, ratio_rows = [], []
    for k in tasks:
        if cpu[k] > 0.1:
            r = wall[k] / cpu[k]
            ratios.append(r)
            ratio_rows.append({"problem": k[0], "n": k[1], "method": k[2],
                               "cpu_median_s": cpu[k], "wall_median_s": wall[k], "ratio_reel_sur_cpu": round(r, 5)})
    med, p5, p95 = (float(x) for x in np.percentile(ratios, [50, 5, 95]))
    stop = not (0.95 <= med <= 1.05)

    per_exec = [float(r["wall_perf_counter_s"]) / float(r["cpu_process_time_s"]) for r in rows
                if float(r["wall_perf_counter_s"]) > 0.1 and float(r["cpu_process_time_s"]) > 0]
    e_med, e_p5, e_p95 = (float(x) for x in np.percentile(per_exec, [50, 5, 95]))

    low = sorted(k for k in tasks if wall[k] < LOW_PREC_S)
    with open(os.path.join(OUT_DIR, "etape1_ratio_reel_cpu_taches.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(ratio_rows[0].keys()))
        w.writeheader()
        w.writerows(ratio_rows)
    with open(os.path.join(OUT_DIR, "etape1_faible_precision.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["problem", "n", "method", "wall_median_s", "cpu_median_s", "bloc"])
        for k in low:
            w.writerow([k[0], k[1], k[2], wall[k], cpu[k], timing.bloc[k]])

    res = {
        "source": "revision/timing_v0.5/timing_catA_v0.5_A.csv (passage A, nuit du 2026-10-11)",
        "validation_taches_cpu_median_sup_0.1s": {
            "n_taches": len(ratios), "ratio_mediane": round(med, 4), "ratio_p5": round(p5, 4),
            "ratio_p95": round(p95, 4), "intervalle_exige": [0.95, 1.05], "arret": stop},
        "rappel_par_execution_reel_sup_0.1s": {
            "n_executions": len(per_exec), "ratio_mediane": round(e_med, 4), "ratio_p5": round(e_p5, 4),
            "ratio_p95": round(e_p95, 4)},
        "taches_total": len(tasks), "taches_faible_precision_reel_median_sous_1ms": len(low),
        "taches_faible_precision_par_bloc": {b: sum(1 for k in low if timing.bloc[k] == b) for b in ("A3", "A3bis", "A3ter")},
        "taches_cpu_median_nul": sum(1 for k in tasks if cpu[k] == 0),
        "seuil_faible_precision_s": LOW_PREC_S,
    }
    with open(os.path.join(OUT_DIR, "etape1_validation.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2, ensure_ascii=False)
    print(json.dumps(res, indent=2, ensure_ascii=False))
    if stop:
        print("\n*** ARRET : mediane hors [0,95 ; 1,05] ***")
        sys.exit(2)


if __name__ == "__main__":
    main()
