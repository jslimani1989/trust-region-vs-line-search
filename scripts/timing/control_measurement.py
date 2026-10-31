# -*- coding: utf-8 -*-
"""
Consigne 008c, A2 : mesure de controle (produit matriciel fixe, graine fixee),
utilisee (i) pour choisir le coeur logique le plus rapide (A1), et (ii) comme
sonde de derive de frequence pendant la nuit (avant la tache 1, puis toutes
les 20 taches, A4.2).

Deux matrices 500x500 fixees UNE SEULE FOIS pour toute la session (memes
matrices reutilisees a chaque controle, sinon la mesure ne sonderait pas la
meme charge de travail). Repete N_REP fois par controle pour attenuer le bruit
de resolution de l'horloge ; la valeur rapportee est la somme sur les N_REP
repetitions (choix de mise en oeuvre, consigne a signaler dans le rapport).
"""
import time

import numpy as np

CONTROL_SEED = 20260927  # graine fixee (meme convention que 007/008b : date du jour)
MATRIX_SIZE = 500
N_REP = 10

_rng = np.random.default_rng(CONTROL_SEED)
_A = _rng.standard_normal((MATRIX_SIZE, MATRIX_SIZE))
_B = _rng.standard_normal((MATRIX_SIZE, MATRIX_SIZE))


def run_control_measurement():
    """Execute le produit matriciel fixe N_REP fois. Retourne un dict avec les
    temps CPU processus (time.process_time) et reel (time.perf_counter),
    cumules sur les N_REP repetitions, et un horodatage ISO."""
    t_cpu0 = time.process_time()
    t_wall0 = time.perf_counter()
    checksum = 0.0
    for _ in range(N_REP):
        C = _A @ _B
        checksum += float(C[0, 0])
    t_cpu1 = time.process_time()
    t_wall1 = time.perf_counter()
    return {
        "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "cpu_process_time_s": t_cpu1 - t_cpu0,
        "wall_perf_counter_s": t_wall1 - t_wall0,
        "n_rep": N_REP,
        "matrix_size": MATRIX_SIZE,
        "seed": CONTROL_SEED,
        "checksum": checksum,  # trace de coherence (memes matrices -> meme checksum a chaque fois)
    }
