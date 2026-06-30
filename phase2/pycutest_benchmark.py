#!/usr/bin/env python3
"""
pycutest_benchmark.py — 
============================
Validation CUTEst : 6 problèmes Tier 1 × 6 solveurs × n variables.
Réutilise run_one() du framework — cohérence totale avec resultats_benchmark.csv.

Sorties :
  results_cutest.csv   — même schéma que resultats_benchmark.csv
  audit_cutest.csv     — comparaison benchmark vs CUTEst (delta en pp)

Usage :
  python pycutest_benchmark.py
  python pycutest_benchmark.py --bench resultats_benchmark.csv
  python pycutest_benchmark.py --skip-audit   (si benchmark indisponible)

Migration plan : pycutest/CUTEst Tier 1 problems
"""

import sys
import os
import time
import argparse
import warnings
import numpy as np
import pandas as pd

# ── pycutest ─────────────────────────────────────────────────────────────────
try:
    import pycutest
except ImportError:
    sys.exit("pycutest non trouvé. Lancez : conda activate cutest")

# ── Framework (même répertoire) ───────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from framework_united_final import run_one
except ImportError:
    sys.exit("framework_united_final.py introuvable. Placez ce script dans le même dossier.")

# ── Constantes ────────────────────────────────────────────────────────────────
KMAX         = 1000
TOL          = 1e-6
WARN_PP      = 5.0    # seuil WARN  : > 5 points de pourcentage
CRITICAL_PP  = 15.0   # seuil CRIT  : > 15 points de pourcentage

METHOD_NAMES = ["Cauchy_Point", "Dogleg", "Steihaug_CG",
                "BFGS_Armijo", "BFGS_Wolfe", "Newton_BT"]

# ── Définition des 6 problèmes Tier 1 ────────────────────────────────────────
# (cutest_name, bench_id, bench_category, n_list, sif_params_fn)
# sif_params_fn(n) → dict passé à pycutest.import_problem(sifParams=...)
TIER1 = [
    # Problèmes à dimension fixe
    ("BEALE",    "A08", "A", [2],
     lambda n: {}),

    ("FREUROTH", "A11", "A", [2],
     lambda n: {"N": 2}),   # N=2 → 2 paires FR → n=4... vérifier au runtime

    ("WOODS",    "A06", "A", [4],
     lambda n: {"N": 1}),   # N=1 → n=4 (Wood standard)

    # Problèmes scalables
    ("BROYDN3D", "B6_Broydn3D", "B", [10, 50, 100, 500, 1000],
     lambda n: {"N": n}),

    ("BRYBND",   "B7_BryBnd",   "B", [10, 50, 100, 500, 1000],
     lambda n: {"N": n}),

    ("CHAINWOO", "B8_ChainWoo", "B", [10, 50, 100, 500, 1000],
     lambda n: {"N": n}),   # N doit être multiple de 4 — ajusté au runtime
]


# ══════════════════════════════════════════════════════════════════════════════
# IMPORT PYCUTEST + ADAPTATION EN PB DICT
# ══════════════════════════════════════════════════════════════════════════════

def _nearest_valid_n(cutest_name: str, n: int) -> int:
    """Ajuste n au multiple valide le plus proche (ex. CHAINWOO : multiple de 4)."""
    if cutest_name == "CHAINWOO":
        return max(4, (n // 4) * 4)   # multiple de 4 inférieur, min 4
    return n


def import_tier1(cutest_name: str, sif_params: dict):
    """Importe un problème pycutest avec gestion d'erreur."""
    try:
        if sif_params:
            prob = pycutest.import_problem(cutest_name, sifParams=sif_params)
        else:
            prob = pycutest.import_problem(cutest_name)
        return prob, None
    except Exception as e:
        return None, str(e)


def make_pb(prob, bench_id: str, category: str) -> dict:
    """Convertit un objet pycutest en pb-dict compatible run_one().

    Closures explicites sur _prob pour éviter les captures tardives en boucle.
    """
    _prob = prob

    def f_fn(x):
        return float(_prob.obj(x))

    def g_fn(x):
        _, g = _prob.obj(x, gradient=True)
        return np.array(g, dtype=float)

    def h_fn(x):
        H = _prob.hess(x)
        return np.array(H, dtype=float)

    return {
        "id":       bench_id,
        "name":     f"{_prob.name} (CUTEst n={_prob.n})",
        "category": category,
        "n":        int(_prob.n),
        "f":        f_fn,
        "grad_f":   g_fn,
        "hess_f":   h_fn,
        "x0":       _prob.x0.copy(),
        "f_opt":    0.0,
    }


# ══════════════════════════════════════════════════════════════════════════════
# RUNNER PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

def run_benchmark(verbose: bool = True) -> pd.DataFrame:
    """Lance les 6 méthodes sur les 6 problèmes Tier 1 et retourne les résultats."""
    rows = []
    total = sum(len(ns) for _, _, _, ns, _ in TIER1) * len(METHOD_NAMES)
    done  = 0

    for cutest_name, bench_id, category, n_list, sif_fn in TIER1:
        for n_target in n_list:

            # Ajustement n pour contraintes internes (ex. CHAINWOO multiple de 4)
            n_valid = _nearest_valid_n(cutest_name, n_target)
            if n_valid != n_target and verbose:
                print(f"  [{cutest_name}] n={n_target} → ajusté à n={n_valid} "
                      f"(contrainte {cutest_name})")

            sif_params = sif_fn(n_valid)

            # Import pycutest (compile si non en cache — ~30s première fois)
            if verbose:
                print(f"Importation {cutest_name} n={n_valid} sifParams={sif_params} ...",
                      end=" ", flush=True)
            prob, err = import_tier1(cutest_name, sif_params)
            if prob is None:
                print(f"ERREUR: {err}")
                for method in METHOD_NAMES:
                    rows.append({
                        "problem": bench_id, "name": cutest_name,
                        "category": category, "method": method,
                        "n": n_valid, "status": "import_error",
                        "converged": False, "n_iter": 0,
                        "n_f_evals": 0, "n_g_evals": 0, "n_h_evals": 0,
                        "n_total": 0, "budget_ratio": np.nan,
                        "cpu_s": 0.0, "f_final": np.nan,
                        "f_error": np.nan, "grad_norm": np.nan,
                        "error_msg": err,
                    })
                    done += 1
                continue

            if verbose:
                print(f"OK (n={prob.n})")

            # Vérifier que la dimension obtenue correspond à ce qu'on voulait
            if prob.n != n_valid and verbose:
                print(f"  Attention: n demandé={n_valid}, n obtenu={prob.n}")

            pb = make_pb(prob, bench_id, category)

            # Exécuter les 6 méthodes
            for method in METHOD_NAMES:
                result = run_one(pb, method)
                rows.append(result)
                done += 1

                if verbose:
                    conv = "✓" if result["converged"] else "✗"
                    print(f"  [{done:3d}/{total}] {method:<14s} "
                          f"conv={conv} iter={result['n_iter']:4d} "
                          f"N_tot={result['n_total']:6d} "
                          f"||∇f||={result['grad_norm']:.2e}")

    df = pd.DataFrame(rows)
    # Retirer la colonne cg_inner_total si présente (spécifique framework)
    if "cg_inner_total" in df.columns:
        df = df.drop(columns=["cg_inner_total"])
    return df


# ══════════════════════════════════════════════════════════════════════════════
# AUDIT — COMPARAISON BENCHMARK vs CUTEST
# ══════════════════════════════════════════════════════════════════════════════

def run_audit(df_cutest: pd.DataFrame,
              bench_path: str) -> pd.DataFrame:
    """Compare les taux de convergence CUTEst vs benchmark Phase 1.

    Seuils (SPEC  §3) :
      OK       : |delta| ≤ 5 pp
      WARN     : 5 < |delta| ≤ 15 pp
      CRITICAL : |delta| > 15 pp
    """
    df_bench = pd.read_csv(bench_path)

    # Taux convergence benchmark par (problem, method)
    bench_conv = (df_bench.groupby(["problem", "method"])["converged"]
                           .mean()
                           .reset_index()
                           .rename(columns={"converged": "conv_bench"}))

    # Taux convergence CUTEst par (problem, method)
    cutest_conv = (df_cutest.groupby(["problem", "method"])["converged"]
                             .mean()
                             .reset_index()
                             .rename(columns={"converged": "conv_cutest"}))

    audit = bench_conv.merge(cutest_conv, on=["problem", "method"], how="outer")
    audit["delta_pp"] = (audit["conv_cutest"] - audit["conv_bench"]).round(4) * 100

    def flag(delta):
        if pd.isna(delta):       return "MISSING"
        if abs(delta) <= WARN_PP:    return "OK"
        if abs(delta) <= CRITICAL_PP: return "WARN"
        return "CRITICAL"

    audit["flag"] = audit["delta_pp"].apply(flag)
    audit["conv_bench"]  = audit["conv_bench"].round(3)
    audit["conv_cutest"] = audit["conv_cutest"].round(3)
    audit["delta_pp"]    = audit["delta_pp"].round(1)
    return audit.sort_values(["problem", "method"]).reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
# RÉSUMÉ CONSOLE
# ══════════════════════════════════════════════════════════════════════════════

def print_summary(df_cutest: pd.DataFrame,
                  df_audit: pd.DataFrame | None) -> None:
    print("\n" + "=" * 65)
    print("RÉSUMÉ — Benchmark pycutest Tier 1")
    print("=" * 65)

    print("\nTaux de convergence par problème et méthode :")
    conv = (df_cutest.groupby(["problem", "method"])["converged"]
                     .mean().round(3).unstack("method"))
    print(conv.to_string())

    if df_audit is not None:
        print("\n" + "-" * 65)
        print("AUDIT — Écarts benchmark vs CUTEst (points de pourcentage)")
        print("-" * 65)
        warns    = df_audit[df_audit["flag"] == "WARN"]
        crits    = df_audit[df_audit["flag"] == "CRITICAL"]
        missings = df_audit[df_audit["flag"] == "MISSING"]

        print(f"  OK       : {(df_audit['flag']=='OK').sum()} paires")
        print(f"  WARN     : {len(warns)} paires  (> {WARN_PP} pp)")
        print(f"  CRITICAL : {len(crits)} paires  (> {CRITICAL_PP} pp)")
        print(f"  MISSING  : {len(missings)} paires")

        if not crits.empty:
            print("\nPaires CRITICAL :")
            print(crits[["problem","method","conv_bench","conv_cutest","delta_pp"]].to_string(index=False))

        if not warns.empty:
            print("\nPaires WARN :")
            print(warns[["problem","method","conv_bench","conv_cutest","delta_pp"]].to_string(index=False))


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="pycutest benchmark — ")
    parser.add_argument("--bench",       default="resultats_benchmark.csv",
                        help="Fichier benchmark Phase 1")
    parser.add_argument("--out-cutest",  default="results_cutest.csv")
    parser.add_argument("--out-audit",   default="audit_cutest.csv")
    parser.add_argument("--skip-audit",  action="store_true",
                        help="Ne pas comparer avec le benchmark Phase 1")
    parser.add_argument("--quiet",       action="store_true")
    args = parser.parse_args()

    verbose = not args.quiet
    print("=" * 65)
    print(" — pycutest benchmark Tier 1")
    print(f"Problèmes : {len(TIER1)}  |  Méthodes : {len(METHOD_NAMES)}")
    print("=" * 65)

    # ── Exécution ─────────────────────────────────────────────────────────────
    df_cutest = run_benchmark(verbose=verbose)
    df_cutest.to_csv(args.out_cutest, index=False)
    print(f"\nRésultats écrits : {args.out_cutest}  ({len(df_cutest)} lignes)")

    # ── Audit ─────────────────────────────────────────────────────────────────
    df_audit = None
    if not args.skip_audit and os.path.exists(args.bench):
        df_audit = run_audit(df_cutest, args.bench)
        df_audit.to_csv(args.out_audit, index=False)
        print(f"Audit écrit      : {args.out_audit}  ({len(df_audit)} lignes)")
    elif not args.skip_audit:
        print(f"Benchmark introuvable ({args.bench}) — audit ignoré.")

    print_summary(df_cutest, df_audit)


if __name__ == "__main__":
    main()

