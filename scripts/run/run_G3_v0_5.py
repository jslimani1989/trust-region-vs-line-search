# -*- coding: utf-8 -*-
"""
Consigne 009d, Etape 5.3 : experience kappa-controlee G3 pour les solveurs modifies qui figurent dans G3 (Dogleg,
BFGS-Armijo, BFGS-Wolfe, Newton-LM), meme protocole et memes graines que run_G3.py (K_max = 5000, 20 graines,
panneau A : n = 100, kappa in {1e2..1e5} ; panneau B : kappa = 1e4, n in {10, 50, 100, 500, 1000} ; graines SEED_Q(n),
SEED_X0(k)). run_G3.py est importe depuis le worktree v0.5 SANS modification : seule la liste METHODS est restreinte
aux quatre solveurs (la famille de problemes et les departs ne dependent pas des solveurs). Cellules independantes
(9 cellules), calculees en parallele ; ecriture cellule par cellule ; REPRISE POSSIBLE (une cellule deja complete
est sautee).
Ecrit revision/run_v0.5/results_G3_v0.5.csv (memes colonnes que results_G3_v0.3.csv ; 9 cellules x 20 graines x
4 solveurs = 720 lignes). Cauchy Point et Steihaug-CG (code inchange) ne sont pas recalcules : leurs lignes restent
celles de run_v0.3/results_G3_v0.3.csv (les iterations externes ne dependent pas de la couche de comptage).
Usage : python run_G3_v0_5.py [--workers 4] [--seeds 20] [--out CHEMIN.csv]  (--seeds/--out : essais)
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

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # racine du depot
WORKTREE = REPO  # le code des solveurs est a la racine du depot
RERUN = os.path.join(REPO, "rerun")  # sorties d'une nouvelle execution (results/ n'est pas modifie)
os.makedirs(RERUN, exist_ok=True)
CODE_META_JSON = os.path.join(REPO, "results", "resultats_benchmark_v0.5_meta.json")
OUT_CSV = os.path.join(RERUN, "results_G3_v0.5.csv")
LOG_PATH = os.path.join(RERUN, "run_G3_v0.5.log")
MODIFIED = ["dogleg", "bfgs_armijo", "bfgs_wolfe", "newton_bt"]


def log(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def git(*args, cwd):
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True).stdout.strip()


def _cell_worker(panel, kappa, n, n_seeds):
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    sys.path.insert(0, WORKTREE)
    import run_G3 as g3
    g3.METHODS = list(MODIFIED)  # run_cell lit la variable globale METHODS du module
    solve = g3.make_real_registry()
    rows = g3.run_cell(panel, kappa, n, n_seeds, solve, g3.KMAX)
    return panel, kappa, n, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seeds", type=int, default=20)
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
    import run_G3 as g3

    cells = [("A", k, g3.PANEL_A["n"]) for k in g3.PANEL_A["kappas"]] + [("B", g3.PANEL_B["kappa"], n) for n in g3.PANEL_B["ns"]]
    done_cells = {}
    write_header = not os.path.exists(args.out)
    if not write_header:
        with open(args.out, encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                key = (r["panel"], float(r["kappa"]), int(r["n"]))
                done_cells[key] = done_cells.get(key, 0) + 1
    expected = args.seeds * len(MODIFIED)
    todo = [c for c in cells if done_cells.get((c[0], float(c[1]), c[2]), 0) < expected]
    log(f"G3 v0.5 (commit {head_wt[:8]}) : {len(cells)} cellules, {len(todo)} a calculer, {args.seeds} graines, sortie {args.out}")
    if len(todo) < len(cells) and any(0 < done_cells.get((c[0], float(c[1]), c[2]), 0) < expected for c in cells):
        sys.exit("ARRET : cellule partiellement ecrite dans la sortie ; supprimer ses lignes avant de reprendre.")

    fh = open(args.out, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(fh, fieldnames=g3.CSV_FIELDS)
    if write_header:
        writer.writeheader()
        fh.flush()
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_cell_worker, p, k, n, args.seeds): (p, k, n) for (p, k, n) in sorted(todo, key=lambda c: -c[2])}
        for fut in as_completed(futs):
            panel, kappa, n, rows = fut.result()
            for r in rows:
                out = {k: r[k] for k in g3.CSV_FIELDS}
                if out["cg_inner_total"] is None:
                    out["cg_inner_total"] = ""
                writer.writerow(out)
            fh.flush()
            log(f"cellule {panel} kappa={kappa:.0e} n={n} : {len(rows)} lignes (elapsed={time.time() - t0:.0f}s)")
    fh.close()
    log("TERMINE.")


if __name__ == "__main__":
    main()
