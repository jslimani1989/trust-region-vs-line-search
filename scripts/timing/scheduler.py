# -*- coding: utf-8 -*-
"""
Consigne 008c, A3 (repris par 009d, etape 7) : ordonnancement.
- Passe 1 : ordre aleatoire (graine notee) des 784 taches distinctes, une execution (repetition 1) de chacune.
- Regle des 60s : le nombre TOTAL de repetitions d'une tache est determine par la duree (temps CPU processus) de sa
  1ere execution -> 3 si <60s, 1 sinon. Ce nombre n'est connu qu'apres la passe 1 ; d'ou une passe 2 distincte pour les
  repetitions 2 et 3 des taches "rapides".
- Passe 2 : chaque tache "rapide" apparait deux fois (repetitions 2 et 3), construite comme deux "tours" independamment
  melanges (graines notees) et concatenes : aucune tache n'a ses deux occurrences adjacentes (sauf cas degenere).

009d : DEUX PASSAGES successifs dans la meme nuit, par un seul processus :
  passage A (mesure principale) puis passage B (replication, autre ordre), graines distinctes entre elles et distinctes
  de toutes celles de 008c/008e/009b/009c (20260927/28/29, 20261003/04/05). Chaque passage a ses propres fichiers de
  sortie, son propre rang d'execution et sa propre cadence de mesure de controle. Motif : sur v0.5 il n'y aura pas
  d'autre nuit pour verifier que les conclusions ne dependent pas de la nuit (variante R3 de 008d-bis).
"""
import random

SEEDS = {
    "A": {"order": 20261011, "tour_a": 20261012, "tour_b": 20261013},
    "B": {"order": 20261014, "tour_a": 20261015, "tour_b": 20261016},
}
SLOW_THRESHOLD_S = 60.0


def build_phase1_order(tasks, passage="A"):
    task_ids = [t["task_id"] for t in tasks]
    rng = random.Random(SEEDS[passage]["order"])
    order = task_ids[:]
    rng.shuffle(order)
    return order


def _no_adjacent_duplicates(seq):
    return all(seq[i] != seq[i + 1] for i in range(len(seq) - 1))


def build_phase2_order(fast_task_ids, last_phase1_task_id=None, passage="A"):
    """fast_task_ids : ids des taches dont la 1ere execution a dure <60s
    (chacune doit encore etre executee 2 fois : repetitions 2 et 3)."""
    if not fast_task_ids:
        return []
    tour_a = fast_task_ids[:]
    random.Random(SEEDS[passage]["tour_a"]).shuffle(tour_a)
    tour_b = fast_task_ids[:]
    random.Random(SEEDS[passage]["tour_b"]).shuffle(tour_b)
    order = tour_a + tour_b

    # garde-fou : jonction interne tour_a/tour_b
    if len(order) >= 2 and len(set(fast_task_ids)) > 1:
        i = len(tour_a)
        j = i
        while j < len(order) and order[j] == order[i - 1]:
            j += 1
        if j != i and j < len(order):
            order[i], order[j] = order[j], order[i]

    # garde-fou : jonction avec la fin de la passe 1
    if last_phase1_task_id is not None and order and order[0] == last_phase1_task_id and len(order) > 1:
        for k in range(1, len(order)):
            if order[k] != order[0]:
                order[0], order[k] = order[k], order[0]
                break

    return order


def classify_fast_slow(phase1_rows_by_task_id):
    """phase1_rows_by_task_id : {task_id: row_dict_de_la_repetition_1}.
    Critere de la regle des 60s : temps CPU processus (cpu_process_time_s)."""
    fast, slow = [], []
    for task_id, row in phase1_rows_by_task_id.items():
        t = float(row["cpu_process_time_s"])
        (fast if t < SLOW_THRESHOLD_S else slow).append(task_id)
    return fast, slow
