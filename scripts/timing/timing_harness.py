# -*- coding: utf-8 -*-
"""
Consigne 009d, etape 7 (repris de 009c : nuit propre et elargie) -- harnais principal de
chronometrage (A3, A3bis, A3ter ; 784 taches, SIX solveurs), SANS instrumentation (instrument=False) et
AVEC memoire du comptage des oracles (memoize=True) sur le tag v0.5. UN SEUL processus Python, du debut a
la fin de la nuit de mesure, pour DEUX passages successifs :
  passage A (mesure principale) puis passage B (replication, autre graine d'ordre), dans des fichiers de
  sortie distincts (suffixes _A et _B), enchaines par une seule commande (--passages A,B, defaut).
A executer avec le repertoire de travail regle sur revision/run_v0.5/tag_v0.5_worktree/ (le code fige du tag
v0.5 est importe depuis ce chemin, cf. WORKTREE ; le commit du worktree doit etre celui du tag). Statistique de
temps primaire = temps reel (perf_counter), le temps CPU processus reste egalement enregistre. Plafond par
execution (defaut 30 min, --cap-seconds) : au-dela, la tache est interrompue et notee "non_mesuree".

Usage :
    python timing_harness.py --core <i>                    # nuit : passage A puis passage B (voir TIMING_DIR)
    python timing_harness.py --dry-run --limit 6           # essai a blanc, passage A seul, sortie jetable _DRYRUN
    python timing_harness.py --passages A --core <i>       # un seul passage (reprise, essai)

Resumable : relit les sorties CSV existantes au demarrage, ne rejoue pas les executions deja enregistrees
(relancer la MEME commande apres une interruption : le passage A reprend, puis le passage B demarre apres la fin
complete du passage A).
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import argparse
import csv
import ctypes
import json
import subprocess
import sys
import threading
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
import common_pin as cp
import control_measurement as cm
import task_list
import scheduler

WORKTREE = task_list.WORKTREE
TIMING_DIR = os.environ.get("TIMING_OUT", os.path.join(WORKTREE, "rerun", "timing"))  # sorties d'une nouvelle mesure
os.makedirs(TIMING_DIR, exist_ok=True)
CAP_SECONDS_DEFAULT = 1800.0  # 009c : plafond par execution (30 minutes)


class ExecutionCap(BaseException):
    """Plafond par execution atteint (009c). Derive de BaseException : le
    `except Exception` de run_one (framework_united_final.py) ne l'attrape pas."""


def arm_cap(cap_s):
    """Declenche ExecutionCap dans le thread principal apres cap_s secondes
    (PyThreadState_SetAsyncExc : l'exception est levee des que le thread
    principal execute du bytecode Python). Le minuteur est cree AVANT le
    demarrage des chronometres. Retourne (minuteur, etat)."""
    main_id = threading.get_ident()
    state = {"done": False, "fired": False}

    def fire():
        if not state["done"]:
            state["fired"] = True
            ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_ulong(main_id), ctypes.py_object(ExecutionCap))

    timer = threading.Timer(cap_s, fire)
    timer.daemon = True
    timer.start()
    return timer, state

FIELDNAMES = [
    "bloc", "problem", "n", "category", "method", "task_id",
    "phase", "rep_index", "global_exec_rank",
    "status", "n_iter", "n_total",
    "cpu_process_time_s", "wall_perf_counter_s", "timestamp_iso",
    "error_msg",
]

CONTROL_FIELDNAMES = ["global_exec_rank_before", "timestamp_iso", "cpu_process_time_s",
                       "wall_perf_counter_s", "n_rep", "matrix_size", "seed", "checksum"]


def log(log_path, msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(log_path, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def load_existing(out_csv):
    if not os.path.exists(out_csv):
        return []
    with open(out_csv, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def load_or_build_plan_phase1(plan_path, tasks, passage="A"):
    if os.path.exists(plan_path):
        with open(plan_path, encoding="utf-8") as fh:
            return json.load(fh)
    order = scheduler.build_phase1_order(tasks, passage)
    with open(plan_path, "w", encoding="utf-8") as fh:
        json.dump({"passage": passage, "seed": scheduler.SEEDS[passage]["order"], "order": order}, fh, indent=2)
    return {"passage": passage, "seed": scheduler.SEEDS[passage]["order"], "order": order}


def load_or_build_plan_phase2(plan_path, existing_rows, passage="A"):
    if os.path.exists(plan_path):
        with open(plan_path, encoding="utf-8") as fh:
            return json.load(fh)
    phase1_rep1 = {r["task_id"]: r for r in existing_rows if r["phase"] == "1" and int(r["rep_index"]) == 1}
    fast_ids, slow_ids = scheduler.classify_fast_slow(phase1_rep1)
    phase1_order_rows = [r for r in existing_rows if r["phase"] == "1"]
    last_task_id = max(phase1_order_rows, key=lambda r: int(r["global_exec_rank"]))["task_id"] if phase1_order_rows else None
    order = scheduler.build_phase2_order(fast_ids, last_phase1_task_id=last_task_id, passage=passage)
    plan = {
        "passage": passage, "seed_tour_a": scheduler.SEEDS[passage]["tour_a"], "seed_tour_b": scheduler.SEEDS[passage]["tour_b"],
        "fast_task_ids": fast_ids, "slow_task_ids": slow_ids, "order": order,
        "threshold_s": scheduler.SLOW_THRESHOLD_S,
    }
    with open(plan_path, "w", encoding="utf-8") as fh:
        json.dump(plan, fh, indent=2)
    return plan


def run_and_record(fw, task, phase, rep_index, global_rank, writer, fh_out, log_path, cap_s=CAP_SECONDS_DEFAULT):
    pb = fw._get_problem(task["problem"], task["n"])
    timer, cap_state = arm_cap(cap_s) if cap_s else (None, {"done": False})
    t_cpu0 = time.process_time()
    t_wall0 = time.perf_counter()
    try:
        res = fw.run_one(pb, task["method"], instrument=False, memoize=True)
        err = ""
    except ExecutionCap:
        res = {"status": "non_mesuree", "n_iter": "", "n_total": ""}
        err = f"plafond de {cap_s:.0f} s atteint : tache interrompue, non mesuree"
    except Exception as e:  # filet de securite : run_one capture deja l'essentiel, mais au cas ou
        res = {"status": "exception", "n_iter": 0, "n_total": 0}
        err = f"{type(e).__name__}: {e}"
    t_cpu1 = time.process_time()
    t_wall1 = time.perf_counter()
    cap_state["done"] = True
    if timer is not None:
        timer.cancel()

    row = {
        "bloc": task["bloc"], "problem": task["problem"], "n": task["n"], "category": task["category"],
        "method": task["method"], "task_id": task["task_id"], "phase": phase, "rep_index": rep_index,
        "global_exec_rank": global_rank,
        "status": res["status"], "n_iter": res["n_iter"], "n_total": res["n_total"],
        "cpu_process_time_s": t_cpu1 - t_cpu0, "wall_perf_counter_s": t_wall1 - t_wall0,
        "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%S"), "error_msg": err,
    }
    writer.writerow(row)
    fh_out.flush()
    log(log_path, f"[{global_rank}] phase{phase} rep{rep_index} {task['task_id']} -> "
                   f"status={row['status']} n_iter={row['n_iter']} n_total={row['n_total']} "
                   f"cpu_s={row['cpu_process_time_s']:.4f} wall_s={row['wall_perf_counter_s']:.4f}")
    return row


def maybe_control_measurement(global_rank_after, control_csv, log_path):
    """Avant la 1ere tache (global_rank_after==0) puis toutes les 20 executions."""
    if global_rank_after % 20 != 0 and global_rank_after != 0:
        return
    res = cm.run_control_measurement()
    res["global_exec_rank_before"] = global_rank_after
    write_header = not os.path.exists(control_csv)
    with open(control_csv, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=CONTROL_FIELDNAMES)
        if write_header:
            w.writeheader()
        w.writerow(res)
    log(log_path, f"[controle apres tache {global_rank_after}] "
                   f"cpu_s={res['cpu_process_time_s']:.5f} wall_s={res['wall_perf_counter_s']:.5f}")


def passage_paths(passage, suffix):
    tag = f"v0.5_{passage}{suffix}"
    return {
        "out_csv": os.path.join(TIMING_DIR, f"timing_catA_{tag}.csv"),
        "control_csv": os.path.join(TIMING_DIR, f"control_measurement_{tag}.csv"),
        "log_path": os.path.join(TIMING_DIR, f"run_log_{tag}.txt"),
        "plan1_path": os.path.join(TIMING_DIR, f"plan_phase1_{tag}.json"),
        "plan2_path": os.path.join(TIMING_DIR, f"plan_phase2_{tag}.json"),
    }


def worktree_commit_ok():
    """Depot publie (v0.5.2) : le tag v0.5 n'y figure pas ; le code doit avoir l'empreinte (hash git des fichiers)
    enregistree dans results/resultats_benchmark_v0.5_meta.json. Retourne (ok, hash_worktree, empreinte_ok)."""
    h_wt = subprocess.run(["git", "rev-parse", "HEAD"], cwd=WORKTREE, capture_output=True, text=True).stdout.strip()
    with open(os.path.join(WORKTREE, "results", "resultats_benchmark_v0.5_meta.json"), encoding="utf-8") as fh:
        ref = json.load(fh)["empreinte_code"]
    fp = {f: subprocess.run(["git", "rev-parse", f"HEAD:{f}"], cwd=WORKTREE, capture_output=True, text=True).stdout.strip()
          for f in ref}
    return bool(h_wt) and fp == ref, h_wt, "empreinte conforme" if fp == ref else "empreinte differente"


def run_passage(passage, args, fw, tasks, suffix):
    """Un passage complet (passe 1 puis passe 2), reprise possible. Retourne True si le passage est COMPLET."""
    P = passage_paths(passage, suffix)
    out_csv, control_csv, log_path = P["out_csv"], P["control_csv"], P["log_path"]
    tasks_by_id = {t["task_id"]: t for t in tasks}
    log(log_path, f"=== Passage {passage} : sortie {out_csv} ; graines {scheduler.SEEDS[passage]} ===")

    existing_rows = load_existing(out_csv)
    done_pairs = {(r["task_id"], int(r["rep_index"])) for r in existing_rows}
    global_rank = len(existing_rows)
    n_written_this_run = 0

    fh_out = open(out_csv, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(fh_out, fieldnames=FIELDNAMES)
    if not existing_rows:
        writer.writeheader()
        fh_out.flush()

    if global_rank == 0:
        maybe_control_measurement(0, control_csv, log_path)

    def limit_reached():
        return args.limit is not None and n_written_this_run >= args.limit

    # ── Passe 1 ──────────────────────────────────────────────────────────
    plan1 = load_or_build_plan_phase1(P["plan1_path"], tasks, passage)
    for task_id in plan1["order"]:
        if limit_reached():
            break
        if (task_id, 1) in done_pairs:
            continue
        task = tasks_by_id[task_id]
        global_rank += 1
        run_and_record(fw, task, phase=1, rep_index=1, global_rank=global_rank,
                       writer=writer, fh_out=fh_out, log_path=log_path, cap_s=args.cap_seconds)
        done_pairs.add((task_id, 1))
        n_written_this_run += 1
        maybe_control_measurement(global_rank, control_csv, log_path)

    phase1_complete = all((tid, 1) in done_pairs for tid in plan1["order"])
    if not phase1_complete:
        log(log_path, f"Arret (limite atteinte ou interruption) avant fin de la passe 1 du passage {passage}. "
                      f"{sum(1 for tid in plan1['order'] if (tid, 1) in done_pairs)}/{len(plan1['order'])} faites.")
        fh_out.close()
        return False

    log(log_path, f"Passage {passage} : passe 1 terminee ({len(plan1['order'])} premieres executions).")

    # ── Passe 2 (taches rapides : repetitions 2 et 3) ───────────────────
    existing_rows_now = load_existing(out_csv)
    plan2 = load_or_build_plan_phase2(P["plan2_path"], existing_rows_now, passage)
    log(log_path, f"Passage {passage} : passe 2 : {len(plan2['fast_task_ids'])} taches rapides (x2 repetitions), "
                  f"{len(plan2['slow_task_ids'])} taches lentes (1 seule repetition, deja faite).")

    rep_used = {tid: {1} for tid in plan1["order"]}
    for tid, rep in done_pairs:
        rep_used.setdefault(tid, set()).add(rep)

    for task_id in plan2["order"]:
        if limit_reached():
            break
        used = rep_used.setdefault(task_id, {1})
        next_rep = 2 if 2 not in used else (3 if 3 not in used else None)
        if next_rep is None or (task_id, next_rep) in done_pairs:
            continue
        task = tasks_by_id[task_id]
        global_rank += 1
        run_and_record(fw, task, phase=2, rep_index=next_rep, global_rank=global_rank,
                       writer=writer, fh_out=fh_out, log_path=log_path, cap_s=args.cap_seconds)
        done_pairs.add((task_id, next_rep))
        used.add(next_rep)
        n_written_this_run += 1
        maybe_control_measurement(global_rank, control_csv, log_path)

    phase2_complete = all((tid, 2) in done_pairs and (tid, 3) in done_pairs for tid in plan2["fast_task_ids"])
    if phase2_complete:
        log(log_path, f"PASSAGE {passage} TERMINE. {global_rank} executions au total ecrites dans {out_csv}.")
    else:
        n_ok = sum(1 for tid in plan2["fast_task_ids"] if (tid, 2) in done_pairs and (tid, 3) in done_pairs)
        log(log_path, f"Arret avant fin de la passe 2 du passage {passage}. {n_ok}/{len(plan2['fast_task_ids'])} taches rapides completees.")
    fh_out.close()
    return phase2_complete


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="ecrit dans des fichiers _DRYRUN separes (passage A seul par defaut)")
    ap.add_argument("--limit", type=int, default=None, help="limite le nb d'executions PAR PASSAGE (tests)")
    ap.add_argument("--core", type=int, default=None, help="coeur logique choisi (A1) ; si omis, pas d'affinite forcee")
    ap.add_argument("--no-priority", action="store_true", help="ne pas passer en priorite haute (tests)")
    ap.add_argument("--passages", default=None, help="passages a executer, ex. A,B (defaut : A,B ; en --dry-run : A)")
    ap.add_argument("--cap-seconds", type=float, default=CAP_SECONDS_DEFAULT,
                    help="plafond par execution en secondes (defaut 1800 = 30 min) ; 0 = pas de plafond")
    args = ap.parse_args()

    suffix = "_DRYRUN" if args.dry_run else ""
    passages = (args.passages or ("A" if args.dry_run else "A,B")).split(",")
    assert all(p in scheduler.SEEDS for p in passages), passages
    log_path = passage_paths(passages[0], suffix)["log_path"]

    ok, h_wt, h_tag = worktree_commit_ok()
    if not ok:
        log(log_path, f"ARRET : le worktree ({h_wt[:8]}) n'est pas sur le commit du tag v0.5 ({h_tag[:8]}).")
        sys.exit(2)

    if args.core is not None:
        cp.set_affinity(args.core)
    if not args.no_priority:
        cp.set_high_priority()
    log(log_path, f"Demarrage. commit={h_wt[:8]} (tag v0.5) affinite={cp.get_affinity()} priorite={cp.get_priority_name()} "
                  f"pid={os.getpid()} dry_run={args.dry_run} limit={args.limit} passages={passages} cap_s={args.cap_seconds}")

    sys.path.insert(0, WORKTREE)
    import framework_united_final as fw

    tasks = task_list.build_tasks()
    assert len(tasks) == 784, len(tasks)

    for i, passage in enumerate(passages):
        complete = run_passage(passage, args, fw, tasks, suffix)
        if not complete:
            log(log_path, f"Passage {passage} incomplet : les passages suivants ne sont pas lances "
                          f"(relancer la meme commande pour reprendre).")
            return
        if i + 1 < len(passages):
            log(log_path, f"Passage {passage} complet -> enchainement sur le passage {passages[i + 1]} (meme processus).")
    log(log_path, f"TOUS LES PASSAGES TERMINES : {passages}.")


if __name__ == "__main__":
    main()
