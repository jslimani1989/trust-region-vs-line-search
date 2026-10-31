# -*- coding: utf-8 -*-
"""
Consigne 009d, Etape 5.2 : departs multiples des QUATRE solveurs modifies (meme protocole et memes graines que
run_multistart.py : graine maitresse 20260927, memes problemes, memes dimensions, memes departs). Les tirages xi sont
generes dans le processus principal, dans l'ordre fixe categorie -> probleme -> dimension -> sigma -> k, de sorte que
les departs sont IDENTIQUES a ceux de results_multistart_v0.4.csv (le generateur ne depend pas des solveurs).
Ecrit revision/run_v0.5/results_multistart_v0.5.csv (memes colonnes que v0.4 ; 4 solveurs x 462 departs = 1848
lignes), memoize=True, instrument=True, K_max=1000, tolerance 1e-6. REPRISE POSSIBLE (relancer la meme commande).
Usage : python run_multistart_v0_5.py [--workers 8] [--limit N --out CHEMIN.csv]  (--limit/--out : essais)
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import argparse
import csv
import json
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # racine du depot
WORKTREE = REPO  # le code des solveurs est a la racine du depot
RERUN = os.path.join(REPO, "rerun")  # sorties d'une nouvelle execution (results/ n'est pas modifie)
os.makedirs(RERUN, exist_ok=True)
CODE_META_JSON = os.path.join(REPO, "results", "resultats_benchmark_v0.5_meta.json")
OUT_CSV = os.path.join(RERUN, "results_multistart_v0.5.csv")
LOG_PATH = os.path.join(RERUN, "run_multistart_v0.5.log")
MODIFIED = ["Dogleg", "BFGS_Armijo", "BFGS_Wolfe", "Newton_BT"]


def log(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def git(*args, cwd):
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True).stdout.strip()


def _worker(cat, pid, n_req, radius_label, draw_idx, x0, method):
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    sys.path.insert(0, WORKTREE)
    import framework_united_final as fw
    fw.KMAX = 1000
    pb0 = fw._get_problem(pid, n_req)
    pb_local = dict(pb0)
    pb_local["x0"] = x0
    row = fw.run_one(pb_local, method)
    return cat, pid, pb0["n"], radius_label, draw_idx, method, row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=OUT_CSV)
    args = ap.parse_args()

    head_wt = git("rev-parse", "HEAD", cwd=WORKTREE)
    # Depot publie (v0.5.2) : le tag v0.5 n'y figure pas. Le garde-fou compare l'empreinte (hash git) des fichiers de
    # code a celle de l'execution d'origine, enregistree dans results/resultats_benchmark_v0.5_meta.json.
    with open(CODE_META_JSON, encoding="utf-8") as _fh:
        _ref = json.load(_fh)["empreinte_code"]
    if {f: git("rev-parse", f"HEAD:{f}", cwd=WORKTREE) for f in _ref} != _ref:
        sys.exit("ARRET : le code n'a pas l'empreinte de l'execution d'origine (results/resultats_benchmark_v0.5_meta.json).")

    sys.path.insert(0, WORKTREE)
    import run_multistart as ms
    import framework_united_final as fw
    fw.KMAX = ms.KMAX

    rng = np.random.default_rng(ms.MASTER_SEED)
    selected = ms.select_problems(fw, rng)
    log(f"=== multistart v0.5 (graine maitresse={ms.MASTER_SEED}, commit {head_wt[:8]}) ===")
    for cat in ("A", "B", "C"):
        log(f"  Categorie {cat} : {selected[cat]}")

    # 1) generation de TOUS les departs, dans l'ordre fixe de run_multistart.py (rng consomme comme en v0.3/v0.4)
    starts_all = []  # (cat, pid, n_req, n_actual, radius_label, draw_idx, x0, lam_min_x0)
    for cat in ("A", "B", "C"):
        for pid in selected[cat]:
            pbs_by_n = {}
            for n_req in ms.TARGET_DIMS:
                pb_try = fw._get_problem(pid, n_req)
                pbs_by_n[pb_try["n"]] = (n_req, pb_try)
            for n_actual, (n_req, pb0) in sorted(pbs_by_n.items()):
                x0_nominal = np.asarray(pb0["x0"], dtype=float)
                scale = max(1.0, float(np.max(np.abs(x0_nominal))))
                starts = [("nominal", 0, x0_nominal.copy())]
                for sigma in ms.SIGMAS:
                    for k in range(ms.N_DRAWS):
                        xi = rng.standard_normal(n_actual)
                        starts.append((f"sigma={sigma}", k, x0_nominal + sigma * scale * xi))
                for radius_label, draw_idx, x0_pert in starts:
                    lam = None
                    if cat == "C":
                        H0 = pb0["hess_f"](x0_pert)
                        lam = float(np.linalg.eigvalsh(0.5 * (H0 + H0.T)).min())
                    starts_all.append((cat, pid, n_req, n_actual, radius_label, draw_idx, x0_pert, lam))
    log(f"{len(starts_all)} departs generes ({len(starts_all) * len(MODIFIED)} executions a faire au total)")

    done = set()
    write_header = not os.path.exists(args.out)
    if not write_header:
        with open(args.out, encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                done.add((r["category"], r["problem"], r["n"], r["radius"], r["draw"], r["method"]))
        log(f"Reprise : {len(done)} lignes deja ecrites.")
    todo = []
    for (cat, pid, n_req, n_actual, radius_label, draw_idx, x0, lam) in starts_all:
        for m in MODIFIED:
            if (cat, pid, str(n_actual), radius_label, str(draw_idx), m) not in done:
                todo.append((cat, pid, n_req, n_actual, radius_label, draw_idx, x0, lam, m))
    if args.limit:
        todo = todo[:args.limit]
    lam_of = {(c, p, na, rl, di): lam for (c, p, nr, na, rl, di, x0, lam) in starts_all}

    fh = open(args.out, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(fh, fieldnames=ms.CSV_FIELDS)
    if write_header:
        writer.writeheader()
        fh.flush()
    t0, n_done = time.time(), 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(_worker, cat, pid, n_req, rl, di, x0, m) for (cat, pid, n_req, na, rl, di, x0, lam, m) in todo]
        for fut in as_completed(futs):
            cat, pid, n_actual, rl, di, m, row = fut.result()
            writer.writerow({"category": cat, "problem": pid, "n": n_actual, "radius": rl, "draw": di, "method": m,
                             "status": row["status"], "converged": row["converged"], "n_iter": row["n_iter"],
                             "n_f_evals": row["n_f_evals"], "n_g_evals": row["n_g_evals"], "n_h_evals": row["n_h_evals"],
                             "n_total": row["n_total"], "f_final": row["f_final"], "grad_norm": row["grad_norm"],
                             "lambda_min_x0": lam_of[(cat, pid, n_actual, rl, di)]})
            fh.flush()
            n_done += 1
            if n_done % 50 == 0 or n_done == len(futs):
                log(f"[{len(done) + n_done}/{len(starts_all) * len(MODIFIED)}] (elapsed={time.time() - t0:.0f}s)")
    fh.close()
    log(f"TERMINE. {n_done} lignes ecrites cette execution dans {args.out}.")


if __name__ == "__main__":
    main()
