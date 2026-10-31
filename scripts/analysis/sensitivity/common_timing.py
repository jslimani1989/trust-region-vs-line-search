# -*- coding: utf-8 -*-
"""
Consigne 009e : chargement des temps mesures et variantes de mesure (reprise de 008d-bis).

Source principale : revision/timing_v0.5/timing_catA_v0.5_A.csv (passage A de la nuit du 2026-10-11, MESURE PRINCIPALE).
Statistique de temps d'une tache = MEDIANE des repetitions du TEMPS REEL (perf_counter, colonne wall_perf_counter_s).
Les taches dont cette statistique est < 1 ms sont "faible precision" (exclues des rapports de temps dans l'analyse
principale ; une tache mesuree sous 1 ms n'est jamais remplacee par une estimation : decision 4).

Variantes de la priorite 3 (robustesse ; regle fixee avant de voir les resultats v0.5) :
  principale : passage A, mediane des repetitions
  R1         : passage A, MINIMUM des repetitions
  R2         : passage A, mediane, apres exclusion des executions dont le rang tombe dans un bloc de 20 executions ENCADRE
               par une mesure de controle hors de +/-10 % de la mediane des mesures de controle du passage A
               (lecture "l'une des deux" : bloc retire si l'UNE des mesures de controle presentes qui l'encadrent est
               hors tolerance)
  R2b        : lecture "les deux" : bloc retire si TOUTES les mesures de controle presentes qui l'encadrent le sont
  R3         : passage B (replication), mediane, memes 784 taches
Difference avec 008d-bis : la nuit v0.5 est un processus unique (pid 21804, rapport 009d) ; la regle "rangs 124 a 134"
(demarrage d'un second processus, 009c) ne s'applique pas et n'est pas reprise (consigne 009e, priorite 3).

Lecture seule.
"""
import csv
import statistics as st

from paths import TIMING_A_CSV, TIMING_B_CSV, CONTROL_A_CSV, keep

LOW_PREC_S = 1e-3
CONTROL_TOL = 0.10
BLOCK = 20
OVERLAP_RANKS = ()  # pas de second processus en v0.5

_CACHE = {}


def load_rows(path):
    with open(path, encoding="utf-8") as fh:
        return [r for r in csv.DictReader(fh) if keep(r)]  # 012


def load_timing_rows(path=TIMING_A_CSV):
    return load_rows(path)


def key_of(r):
    return (r["problem"], r["n"], r["method"])


class Timing:
    """Statistique de temps par tache ; wall[(p, n, m)] en secondes (n est une chaine, comme dans les CSV)."""

    def __init__(self, rows, stat="median", label="principale", note=""):
        assert stat in ("median", "min")
        self.rows, self.stat, self.label, self.note = rows, stat, label, note
        by_task = {}
        for r in rows:
            by_task.setdefault(key_of(r), []).append(r)
        agg = st.median if stat == "median" else min
        self.wall = {k: agg([float(r["wall_perf_counter_s"]) for r in v]) for k, v in by_task.items()}
        self.cpu = {k: st.median([float(r["cpu_process_time_s"]) for r in v]) for k, v in by_task.items()}
        self.n_rep = {k: len(v) for k, v in by_task.items()}
        self.bloc = {k: v[0]["bloc"] for k, v in by_task.items()}

    def is_low_precision(self, k):
        t = self.wall.get(k)
        return t is not None and t < LOW_PREC_S

    def low_precision_tasks(self):
        return sorted(k for k, t in self.wall.items() if t < LOW_PREC_S)


def control_excursions(control_rows=None):
    """{rang_avant: (ecart relatif a la mediane, hors_tolerance)} sur la mesure de controle (temps reel), passage A."""
    control_rows = control_rows if control_rows is not None else load_rows(CONTROL_A_CSV)
    walls = [float(c["wall_perf_counter_s"]) for c in control_rows]
    med = st.median(walls)
    out = {}
    for c, w in zip(control_rows, walls):
        rel = w / med - 1.0
        out[int(c["global_exec_rank_before"])] = (rel, abs(rel) > CONTROL_TOL)
    return med, out


def perturbed_ranks(n_exec, reading="either"):
    """Rangs d'execution (1..n_exec) retires par R2.

    Bloc b = rangs 20b+1 .. 20b+20, encadre par la mesure de controle de rang_avant = 20b (avant) et celle de
    rang_avant = 20(b+1) (apres ; absente pour le dernier bloc partiel). reading='either' : bloc retire si l'une
    des mesures presentes qui l'encadrent est hors de +/-10 % de la mediane des mesures de controle (lecture
    retenue) ; reading='both' : si toutes les mesures presentes qui l'encadrent le sont (lecture alternative)."""
    _, exc = control_excursions()
    removed = set(OVERLAP_RANKS)
    removed_blocks = set()
    for r in range(1, n_exec + 1):
        b = (r - 1) // BLOCK
        flags = [exc[x][1] for x in (BLOCK * b, BLOCK * (b + 1)) if x in exc]
        hit = any(flags) if reading == "either" else (bool(flags) and all(flags))
        if hit:
            removed.add(r)
            removed_blocks.add(b)
    return {r for r in removed if 1 <= r <= n_exec}, removed_blocks


def get_timing(variant="principale"):
    """Retourne l'objet Timing d'une variante (mis en cache)."""
    if variant in _CACHE:
        return _CACHE[variant]
    rows_a = load_timing_rows(TIMING_A_CSV)
    if variant == "principale":
        t = Timing(rows_a, "median", "principale", "passage A, mediane des repetitions du temps reel")
    elif variant == "R1":
        t = Timing(rows_a, "min", "R1", "passage A, minimum des repetitions du temps reel")
    elif variant in ("R2", "R2b"):
        reading = "either" if variant == "R2" else "both"
        n_exec = max(int(r["global_exec_rank"]) for r in rows_a)
        removed, blocks = perturbed_ranks(n_exec, reading)
        kept = [r for r in rows_a if int(r["global_exec_rank"]) not in removed]
        t = Timing(kept, "median", variant,
                   f"passage A, mediane apres retrait de {len(rows_a) - len(kept)} executions sur {len(rows_a)} "
                   f"(lecture '{reading}' : {len(blocks)} blocs de 20)")
        t.n_removed = len(rows_a) - len(kept)
        t.n_blocks_removed = len(blocks)
        t.tasks_lost = sorted({key_of(r) for r in rows_a} - set(t.wall))
    elif variant == "R3":
        rows_b = load_timing_rows(TIMING_B_CSV)
        t = Timing(rows_b, "median", "R3", "passage B (replication), mediane, memes 784 taches")
    else:
        raise ValueError(variant)
    _CACHE[variant] = t
    return t


# --- compatibilite avec les noms de 008d (analyse principale) -------------------------------------------------
def median_wall_by_task(rows=None):
    return get_timing("principale").wall if rows is None else Timing(rows, "median").wall


def median_cpu_by_task(rows=None):
    return get_timing("principale").cpu if rows is None else Timing(rows, "median").cpu


def min_wall_by_task(rows=None):
    return get_timing("R1").wall if rows is None else Timing(rows, "min").wall


def is_low_precision(t):
    return t is not None and t < LOW_PREC_S


def measured_reliable(times):
    return {k: t for k, t in times.items() if t >= LOW_PREC_S}


def task_blocs(rows=None):
    return get_timing("principale").bloc
