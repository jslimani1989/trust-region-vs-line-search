# -*- coding: utf-8 -*-
"""
Consigne 008c, A1 : identification du coeur logique le plus rapide, en
executant la mesure de controle (A2) successivement sur chacun des
processeurs logiques (masque d'affinite Windows), un seul a la fois.
A executer avant le lancement de la nuit de mesure (machine deja au repos,
mais duree breve : ~16 x quelques dixiemes de seconde).

Sortie : revision/timing_v0.5/a1_core_selection.csv (une ligne par coeur
logique) et impression du coeur retenu (le plus rapide, temps reel
perf_counter le plus court).
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import csv
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
import common_pin as cp
import control_measurement as cm

OUT_CSV = os.path.join(os.environ.get("TIMING_OUT", os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "rerun", "timing")),
                       "a1_core_selection.csv")  # depot publie (v0.5.2)
os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)


def main():
    n_cpu = cp.n_logical_processors()
    print(f"Processeurs logiques detectes : {n_cpu}")
    rows = []
    for cpu_idx in range(n_cpu):
        try:
            cp.set_affinity(cpu_idx)
        except OSError as e:
            print(f"  cpu {cpu_idx}: echec affinite ({e}), ignore")
            continue
        # une repetition "a blanc" pour stabiliser les caches / la frequence avant la mesure retenue
        cm.run_control_measurement()
        res = cm.run_control_measurement()
        res["cpu_index"] = cpu_idx
        rows.append(res)
        print(f"  cpu {cpu_idx:2d} : cpu_process_time_s={res['cpu_process_time_s']:.5f}  "
              f"wall_perf_counter_s={res['wall_perf_counter_s']:.5f}")

    best = min(rows, key=lambda r: r["wall_perf_counter_s"])
    print(f"\nCoeur logique retenu (le plus rapide, perf_counter) : cpu {best['cpu_index']} "
          f"(wall={best['wall_perf_counter_s']:.5f}s)")

    fieldnames = ["cpu_index", "timestamp_iso", "cpu_process_time_s", "wall_perf_counter_s",
                  "n_rep", "matrix_size", "seed", "checksum"]
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"Ecrit : {OUT_CSV}")

    # remet l'affinite sur le coeur retenu pour la suite immediate (sera re-fixee
    # explicitement par le harnais principal de toute facon)
    cp.set_affinity(best["cpu_index"])
    return best["cpu_index"]


if __name__ == "__main__":
    main()
