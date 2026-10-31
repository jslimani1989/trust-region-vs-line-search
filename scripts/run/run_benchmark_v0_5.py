# -*- coding: utf-8 -*-
"""
Consigne 009d, Etape 5.1 : nouvelle execution des QUATRE solveurs modifies (BFGS-Armijo, BFGS-Wolfe, Newton-LM,
Dogleg) sur le tag v0.5 : 4 x 180 creneaux = 720 taches, memoize=True, instrument=True, K_max=1000, tolerance 1e-6.
Ecrit revision/run_v0.5/resultats_benchmark_v0.5_nouvelles_lignes.csv (colonnes de v0.4 + code_version), ligne par
ligne, par le processus principal seul ; REPRISE POSSIBLE (relancer la meme commande). Le fichier final
resultats_benchmark_v0.5.csv (lignes v0.4 de Cauchy Point et Steihaug-CG + 720 nouvelles lignes) est produit par
fusion_v0_5.py.

Garde-fous : le worktree doit etre sur le commit du tag v0.5 ; le commit et l'EMPREINTE du code (hash git des quatre
fichiers de code) sont enregistres dans resultats_benchmark_v0.5_meta.json au premier lancement ; une reprise est refusee
si l'empreinte change (un deplacement du tag qui ne touche que les tests ne bloque pas ; apres un deplacement du tag,
mettre le worktree a jour : `git -C <worktree> checkout --detach v0.5`).

Parallelisme autorise (aucun temps n'est utilise dans cette consigne). Taches triees par cout decroissant (cpu_s de
v0.4) pour limiter la queue de distribution. Usage :
    python run_benchmark_v0_5.py [--workers 10] [--limit N --out CHEMIN.csv]   (--limit/--out : essais)
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
OUT_CSV = os.path.join(RERUN, "resultats_benchmark_v0.5_nouvelles_lignes.csv")
META_JSON = os.path.join(RERUN, "resultats_benchmark_v0.5_meta.json")
LOG_PATH = os.path.join(RERUN, "run_benchmark_v0.5.log")
V04_CSV = os.path.join(REPO, "results", "reference", "resultats_benchmark_v0.4.csv")  # ordre des taches seulement
MODIFIED = ["Dogleg", "BFGS_Armijo", "BFGS_Wolfe", "Newton_BT"]
CODE_VERSION = "v0.5"
CODE_FILES = ["framework_united_final.py", "trust_region.py", "implementation_methode_lineaires.py",
              "benchmark_all_categories_no_levy_validation.py"]


def log(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def git(*args, cwd):
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True).stdout.strip()


def _worker(pid, n, m):
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    sys.path.insert(0, WORKTREE)
    import framework_united_final as fw
    pb = fw._get_problem(pid, n)
    row = fw.run_one(pb, m)  # instrument=True, memoize=True (defauts)
    row.pop("lm_trace", None)
    row["code_version"] = CODE_VERSION
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=OUT_CSV)
    args = ap.parse_args()
    smoke = args.out != OUT_CSV

    head_wt = git("rev-parse", "HEAD", cwd=WORKTREE)
    # Depot publie (v0.5.2) : le tag v0.5 n'y figure pas. Le garde-fou compare l'empreinte (hash git) des fichiers de
    # code a celle de l'execution d'origine, enregistree dans results/resultats_benchmark_v0.5_meta.json.
    with open(CODE_META_JSON, encoding="utf-8") as _fh:
        _ref = json.load(_fh)["empreinte_code"]
    if {f: git("rev-parse", f"HEAD:{f}", cwd=WORKTREE) for f in _ref} != _ref:
        sys.exit("ARRET : le code n'a pas l'empreinte de l'execution d'origine (results/resultats_benchmark_v0.5_meta.json).")
    fp = {f: git("rev-parse", f"HEAD:{f}", cwd=WORKTREE) for f in CODE_FILES}
    if not smoke:
        if os.path.exists(META_JSON):
            with open(META_JSON, encoding="utf-8") as fh:
                prev = json.load(fh)
            if prev["empreinte_code"] != fp:
                sys.exit("ARRET : le code des solveurs/problemes/framework a change depuis le premier lancement "
                         "(empreinte des fichiers differente) ; supprimer le csv partiel et le meta pour repartir.")
            if prev["commit"] != head_wt:
                log(f"Note : le commit du tag a change ({prev['commit'][:8]} -> {head_wt[:8]}) sans toucher aux fichiers de code : reprise autorisee.")
        else:
            with open(META_JSON, "w", encoding="utf-8") as fh:
                json.dump({"commit": head_wt, "tag": "v0.5", "empreinte_code": fp, "debut": time.strftime("%Y-%m-%d %H:%M:%S")}, fh, indent=2)

    sys.path.insert(0, WORKTREE)
    import framework_united_final as fw

    specs = fw.build_problem_specs(dims=[10, 50, 100, 500, 1000], categories=["A", "B", "C"])
    assert len(specs) == 180, len(specs)
    tasks = [(pid, n, m) for (pid, n) in specs for m in MODIFIED]
    assert len(tasks) == 720, len(tasks)

    cost = {}
    with open(V04_CSV, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            cost[(r["problem"], r["n"], r["method"])] = float(r["cpu_s"] or 0.0)
    tasks.sort(key=lambda t: -cost.get((t[0], str(t[1]), t[2]), 0.0))

    fields = list(fw.FIELDS) + ["code_version"]
    # Les 180 creneaux contiennent des doublons (problemes a dimension fixe : meme instance sur plusieurs creneaux) ;
    # comme en v0.4, chaque creneau est execute et ecrit. Reprise par decompte (cle -> lignes deja ecrites).
    from collections import Counter
    done_count = Counter()
    write_header = not os.path.exists(args.out)
    if not write_header:
        with open(args.out, encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                done_count[(r["problem"], r["n"], r["method"])] += 1
        log(f"Reprise : {sum(done_count.values())} lignes deja ecrites, {len(tasks) - sum(done_count.values())} restantes.")
    todo, remaining = [], Counter(done_count)
    for (pid, n, m) in tasks:
        k = (pid, str(n), m)
        if remaining[k] > 0:
            remaining[k] -= 1
        else:
            todo.append((pid, n, m))
    done = list(range(sum(done_count.values())))  # pour le compteur d'avancement
    if args.limit:
        todo = todo[-args.limit:]  # essais : les taches les moins couteuses

    log(f"Debut : commit {head_wt[:8]} (tag v0.5), {len(todo)} taches, {args.workers} processus, sortie {args.out}")
    fh = open(args.out, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
    if write_header:
        writer.writeheader()
        fh.flush()

    t_start = time.time()
    n_done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(_worker, pid, n, m): (pid, n, m) for (pid, n, m) in todo}
        for fut in as_completed(futures):
            pid, n, m = futures[fut]
            row = fut.result()
            writer.writerow(row)
            fh.flush()
            n_done += 1
            log(f"[{len(done) + n_done}/{len(tasks)}] {pid} n={n} {m} -> status={row['status']} n_iter={row['n_iter']} "
                f"n_f={row['n_f_evals']} n_g={row['n_g_evals']} n_h={row['n_h_evals']} n_total={row['n_total']} "
                f"(elapsed={time.time() - t_start:.1f}s)")
    fh.close()
    log(f"TERMINE. {n_done} lignes ecrites cette execution ({len(done) + n_done} au total) dans {args.out}.")


if __name__ == "__main__":
    main()
