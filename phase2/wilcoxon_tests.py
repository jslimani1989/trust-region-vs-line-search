#!/usr/bin/env python3
"""
wilcoxon_tests.py — Tests statistiques G7 (B28)
================================================
Wilcoxon signed-rank pairé + correction Holm–Bonferroni.
3 familles séparées (Cat. A, B, C). 8 paires ciblées par défaut.

Protocole : PROTOCOL_G7_WILCOXON.md (verrouillé B27)
Source    : resultats_benchmark.csv
Sorties   :
  wilcoxon_results.csv          — résultats détaillés
  wilcoxon_table.tex            — table LaTeX §4
  decision_tree_annotations.txt — annotations §6

Usage :
  python wilcoxon_tests.py                        # 8 paires ciblées
  python wilcoxon_tests.py --pairs all            # 15 paires exhaustives
  python wilcoxon_tests.py --input autre.csv
"""

import sys
import argparse
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from itertools import combinations

# ── Paramètres protocole (W1–W5) ──────────────────────────────────────────────
ALPHA       = 0.05
METRIC      = "n_total"       # W2 : N_tot uniquement
CATEGORIES  = ["A", "B", "C"] # W5 : 3 familles séparées

# Noms exacts dans le CSV
METHODS = ["Cauchy_Point", "Dogleg", "Steihaug_CG",
           "BFGS_Armijo", "BFGS_Wolfe", "Newton_BT"]

# 8 paires ciblées (W3 = "targeted") — alignées sur les claims du papier
PAIRS_TARGETED = [
    ("P1", "Steihaug_CG",  "Cauchy_Point", "Steihaug-CG vs Cauchy-Point (intra-TR scalability)"),
    ("P2", "Steihaug_CG",  "Dogleg",       "Steihaug-CG vs Dogleg (intra-TR, Hessian available)"),
    ("P3", "Steihaug_CG",  "BFGS_Armijo",  "Steihaug-CG vs BFGS-Armijo (inter-family Pareto)"),
    ("P4", "Steihaug_CG",  "BFGS_Wolfe",   "Steihaug-CG vs BFGS-Wolfe (Pareto main competitor)"),
    ("P5", "Steihaug_CG",  "Newton_BT",    "Steihaug-CG vs Newton-BT (Cat. C saddle claim)"),
    ("P6", "Dogleg",       "Cauchy_Point", "Dogleg vs Cauchy-Point (intra-TR sub-model)"),
    ("P7", "BFGS_Wolfe",   "BFGS_Armijo",  "BFGS-Wolfe vs BFGS-Armijo (Wolfe conditions)"),
    ("P8", "BFGS_Wolfe",   "Newton_BT",    "BFGS-Wolfe vs Newton-BT (intra-LS, Hessian cost)"),
]

# 15 paires exhaustives (W3 = "all")
PAIRS_ALL = [
    (f"P{i+1:02d}", a, b, f"{a} vs {b}")
    for i, (a, b) in enumerate(combinations(METHODS, 2))
]


# ══════════════════════════════════════════════════════════════════════════════
# CHARGEMENT ET PRÉPARATION (W1 : médiane par instance)
# ══════════════════════════════════════════════════════════════════════════════

def load_and_prepare(path: str) -> pd.DataFrame:
    """Charge le CSV, valide le schéma, déduplique par médiane (W1=A).

    Retourne un DataFrame pivoté :
        index = (category, problem, n)
        columns = méthodes
        values = n_total médian sur les éventuels doublons de seeds
    """
    df = pd.read_csv(path)

    # ── Validation schéma ─────────────────────────────────────────────────────
    required = {"problem", "category", "method", "n_total", "converged", "n"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes dans {path}: {missing}")

    found_methods = set(df["method"].unique())
    missing_m = set(METHODS) - found_methods
    if missing_m:
        raise ValueError(f"Méthodes manquantes dans le CSV: {missing_m}")

    # ── W1 : médiane sur les doublons (seeds sans colonne explicite) ──────────
    # Certains (problem, n, method) ont jusqu'à 5 lignes identiques (artefact CSV)
    agg = (df.groupby(["category", "problem", "n", "method"], sort=False)[METRIC]
             .median()
             .reset_index())

    # ── Pivot : une colonne par méthode ──────────────────────────────────────
    pivot = agg.pivot_table(index=["category", "problem", "n"],
                            columns="method",
                            values=METRIC)
    pivot.columns.name = None
    pivot = pivot.reset_index()

    # Vérifier alignement : toutes les méthodes présentes pour chaque instance
    for m in METHODS:
        if m not in pivot.columns:
            raise ValueError(f"Méthode {m} absente du pivot.")
        n_missing = pivot[m].isna().sum()
        if n_missing > 0:
            warnings.warn(f"{n_missing} valeurs manquantes pour {m}.")

    return pivot


# ══════════════════════════════════════════════════════════════════════════════
# CORRECTION HOLM–BONFERRONI (step-down)
# ══════════════════════════════════════════════════════════════════════════════

def holm_correction(p_values: np.ndarray, alpha: float = ALPHA):
    """Retourne p-valeurs corrigées Holm (step-down).

    p_corr[i] = min(1, max(p_(j) * (m - j + 1)  for j <= i))
    où les indices sont dans l'ordre croissant des p brutes.

    Compatible avec statsmodels.stats.multitest.multipletests(method='holm')
    mais sans dépendance externe.
    """
    m    = len(p_values)
    order = np.argsort(p_values)
    p_sorted = p_values[order]

    p_corr_sorted = np.empty(m)
    running_max   = 0.0
    for i, p in enumerate(p_sorted):
        adjusted    = p * (m - i)
        running_max = max(running_max, adjusted)
        p_corr_sorted[i] = min(1.0, running_max)

    # Remettre dans l'ordre original
    p_corr = np.empty(m)
    p_corr[order] = p_corr_sorted
    return p_corr


# ══════════════════════════════════════════════════════════════════════════════
# TEST WILCOXON SIGNED-RANK (pairé)
# ══════════════════════════════════════════════════════════════════════════════

def run_wilcoxon(x: np.ndarray, y: np.ndarray):
    """Wilcoxon signed-rank bilatéral sur x vs y.

    Retourne (statistic, p_value). Gère les cas dégénérés :
    - Toutes différences nulles → p=1.0 (pas de différence détectable)
    - n < 10 → zero_method='zsplit' pour test exact asymptotique
    """
    diff = x - y
    if np.all(diff == 0):
        return np.nan, 1.0
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = stats.wilcoxon(x, y, alternative="two-sided",
                                 zero_method="wilcox", method="auto")
        return float(res.statistic), float(res.pvalue)
    except ValueError as e:
        warnings.warn(f"Wilcoxon erreur: {e}")
        return np.nan, 1.0


# ══════════════════════════════════════════════════════════════════════════════
# ANNOTATION
# ══════════════════════════════════════════════════════════════════════════════

def annotate(p_holm: float) -> str:
    if   p_holm < 0.01:  return "★★"
    elif p_holm < 0.05:  return "★"
    else:                 return "ns"


def direction(median_diff: float) -> str:
    """A < B signifie A est meilleure (moins d'oracles)."""
    if   median_diff < -1e-9: return "A<B"
    elif median_diff >  1e-9: return "A>B"
    else:                      return "="


# ══════════════════════════════════════════════════════════════════════════════
# ANALYSE PAR CATÉGORIE (famille Holm indépendante)
# ══════════════════════════════════════════════════════════════════════════════

def analyse_category(pivot_cat: pd.DataFrame, pairs: list, cat: str) -> pd.DataFrame:
    """Lance les tests Wilcoxon sur une catégorie, applique Holm.

    Retourne un DataFrame de résultats pour cette catégorie.
    """
    rows = []

    for pair_id, mA, mB, label in pairs:
        # Instances avec les deux méthodes non-nulles
        mask = pivot_cat[mA].notna() & pivot_cat[mB].notna()
        sub  = pivot_cat[mask]

        if len(sub) < 4:
            warnings.warn(f"Cat {cat} {pair_id}: seulement {len(sub)} instances — test ignoré.")
            rows.append({
                "category": cat, "pair_id": pair_id,
                "method_A": mA,  "method_B": mB, "label": label,
                "n_instances": len(sub), "median_A": np.nan, "median_B": np.nan,
                "median_diff": np.nan, "statistic": np.nan,
                "p_value": np.nan, "p_holm": np.nan,
                "significant": False, "annotation": "ns", "direction": "="
            })
            continue

        xA = sub[mA].values
        xB = sub[mB].values
        stat, pval = run_wilcoxon(xA, xB)

        rows.append({
            "category":    cat,
            "pair_id":     pair_id,
            "method_A":    mA,
            "method_B":    mB,
            "label":       label,
            "n_instances": len(sub),
            "median_A":    round(float(np.median(xA)), 2),
            "median_B":    round(float(np.median(xB)), 2),
            "median_diff": round(float(np.median(xA - xB)), 2),
            "statistic":   round(stat, 4) if not np.isnan(stat) else np.nan,
            "p_value":     round(pval, 6),
            "p_holm":      np.nan,   # rempli après
            "significant": False,
            "annotation":  "ns",
            "direction":   "="
        })

    df_res = pd.DataFrame(rows)

    # ── Holm sur la famille de cette catégorie ────────────────────────────────
    valid = df_res["p_value"].notna()
    if valid.any():
        p_raw   = df_res.loc[valid, "p_value"].values
        p_corr  = holm_correction(p_raw, ALPHA)
        df_res.loc[valid, "p_holm"] = np.round(p_corr, 6)
        df_res.loc[valid, "significant"] = p_corr < ALPHA
        df_res.loc[valid, "annotation"]  = [annotate(p) for p in p_corr]
        df_res.loc[valid, "direction"]   = [
            direction(d) for d in df_res.loc[valid, "median_diff"]
        ]

    return df_res


# ══════════════════════════════════════════════════════════════════════════════
# SORTIE 1 — wilcoxon_results.csv
# ══════════════════════════════════════════════════════════════════════════════

def write_csv(results: pd.DataFrame, path: str) -> None:
    cols = ["category", "pair_id", "method_A", "method_B",
            "n_instances", "median_A", "median_B", "median_diff",
            "statistic", "p_value", "p_holm", "significant",
            "annotation", "direction"]
    results[cols].to_csv(path, index=False)
    print(f"Écrit : {path}  ({len(results)} lignes)")


# ══════════════════════════════════════════════════════════════════════════════
# SORTIE 2 — wilcoxon_table.tex
# ══════════════════════════════════════════════════════════════════════════════

def write_latex(results: pd.DataFrame, path: str, pairs: list) -> None:
    """Table LaTeX §4 : lignes = paires, colonnes = catégories A/B/C.

    Format cellule : median_A vs median_B  annotation
    """
    METHOD_LABEL = {
        "Steihaug_CG":  "Steihaug--CG",
        "Cauchy_Point": "Cauchy",
        "Dogleg":       "Dogleg",
        "BFGS_Armijo":  "BFGS--Armijo",
        "BFGS_Wolfe":   "BFGS--Wolfe",
        "Newton_BT":    "Newton--BT",
    }

    def cell(row):
        if pd.isna(row["p_holm"]):
            return "--"
        mA  = f"{row['median_A']:.0f}"
        mB  = f"{row['median_B']:.0f}"
        ann = row["annotation"]
        arrow = "$\\downarrow$" if row["direction"] == "A<B" else \
                "$\\uparrow$"   if row["direction"] == "A>B" else ""
        return f"{mA} vs {mB} {ann}{arrow}"

    pair_ids = [p[0] for p in pairs]

    lines = []
    lines.append("% Table §4 — Wilcoxon signed-rank + Holm correction")
    lines.append("% ★★ p_holm<0.01 · ★ p_holm<0.05 · ns not significant")
    lines.append("% ↓ method A better · ↑ method B better")
    lines.append(r"\begin{table}[ht]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{l l c c c}")
    lines.append(r"\toprule")
    lines.append(r"ID & Pair & Cat.~A & Cat.~B & Cat.~C \\")
    lines.append(r"\midrule")

    for pid, mA, mB, label in pairs:
        row_A = results[(results["pair_id"] == pid) & (results["category"] == "A")]
        row_B = results[(results["pair_id"] == pid) & (results["category"] == "B")]
        row_C = results[(results["pair_id"] == pid) & (results["category"] == "C")]

        ca = cell(row_A.iloc[0]) if len(row_A) else "--"
        cb = cell(row_B.iloc[0]) if len(row_B) else "--"
        cc = cell(row_C.iloc[0]) if len(row_C) else "--"

        lA = METHOD_LABEL.get(mA, mA)
        lB = METHOD_LABEL.get(mB, mB)
        pair_str = f"{lA} vs {lB}"
        lines.append(f"  {pid} & {pair_str} & {ca} & {cb} & {cc} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\caption{Pairwise Wilcoxon signed-rank tests on $N_{\rm tot}$ "
                 r"(median over instances, Holm--Bonferroni correction, $\alpha=0.05$). "
                 r"$\downarrow$: first method uses fewer oracle evaluations.}")
    lines.append(r"\label{tab:wilcoxon}")
    lines.append(r"\end{table}")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Écrit : {path}")


# ══════════════════════════════════════════════════════════════════════════════
# SORTIE 3 — decision_tree_annotations.txt
# ══════════════════════════════════════════════════════════════════════════════

def write_tree_annotations(results: pd.DataFrame, path: str) -> None:
    """Mappe les résultats aux 4 nœuds de l'arbre de décision §6."""

    def fmt(pid, cat):
        r = results[(results["pair_id"] == pid) & (results["category"] == cat)]
        if len(r) == 0:
            return "n/a"
        r = r.iloc[0]
        ann = r["annotation"]
        d   = r["direction"]
        return f"{ann} ({d})"

    def sig_cats(pid):
        cats = []
        for c in ["A", "B", "C"]:
            r = results[(results["pair_id"] == pid) & (results["category"] == c)]
            if len(r) and r.iloc[0]["significant"]:
                cats.append(c)
        return cats if cats else ["none"]

    lines = []
    lines.append("Decision Tree Annotations — §6")
    lines.append("Source: wilcoxon_tests.py (B28), protocol PROTOCOL_G7_WILCOXON.md")
    lines.append("=" * 60)

    lines.append("\nNode 1 — Is ∇²f available? (Second-order methods)")
    lines.append("  Tests bearing on this node: P2 (TR), P5 (LS)")
    lines.append(f"  P2 Steihaug-CG vs Dogleg    : A={fmt('P2','A')}  B={fmt('P2','B')}  C={fmt('P2','C')}")
    lines.append(f"  P5 Steihaug-CG vs Newton-BT : A={fmt('P5','A')}  B={fmt('P5','B')}  C={fmt('P5','C')}")
    lines.append(f"  → Steihaug-CG significantly better on Cat.: {sig_cats('P4')}")

    lines.append("\nNode 2 — n ≥ 500? (Scalability branch)")
    lines.append("  Tests bearing on this node: P1, P3, P4")
    lines.append(f"  P1 Steihaug-CG vs Cauchy    : A={fmt('P1','A')}  B={fmt('P1','B')}  C={fmt('P1','C')}")
    lines.append(f"  P4 Steihaug-CG vs BFGS-Wolfe: A={fmt('P4','A')}  B={fmt('P4','B')}  C={fmt('P4','C')}")
    lines.append(f"  → Pareto claim (P4) significant on Cat.: {sig_cats('P4')}")

    lines.append("\nNode 3 — H possibly indefinite? (Non-convex branch)")
    lines.append("  Tests bearing on this node: P5, P8")
    lines.append(f"  P5 Steihaug-CG vs Newton-BT : C={fmt('P5','C')}")
    lines.append(f"  P8 BFGS-Wolfe vs Newton-BT  : C={fmt('P8','C')}")
    lines.append(f"  → Newton-BT significantly worse on Cat. C: "
                 f"{'Yes' if 'C' in sig_cats('P5') else 'No'}")

    lines.append("\nNode 4 — Saddle-type geometry? (Cat. C sub-branch)")
    lines.append("  Central claim: Steihaug-CG dominates on C01/C08/C09/C10")
    lines.append(f"  P5 on Cat. C: {fmt('P5','C')}")
    lines.append(f"  P2 on Cat. C: {fmt('P2','C')}")
    lines.append("  → Empirical support for decision-tree leaf:")
    lines.append("    'If saddle-type → Steihaug-CG'")

    lines.append("\n" + "=" * 60)
    lines.append("Intra-family comparisons (supporting evidence)")
    lines.append(f"  P6 Dogleg vs Cauchy-Point: A={fmt('P6','A')}  B={fmt('P6','B')}  C={fmt('P6','C')}")
    lines.append(f"  P7 BFGS-Wolfe vs BFGS-Armijo: A={fmt('P7','A')}  B={fmt('P7','B')}  C={fmt('P7','C')}")
    lines.append(f"  P8 BFGS-Wolfe vs Newton-BT: A={fmt('P8','A')}  B={fmt('P8','B')}  C={fmt('P8','C')}")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Écrit : {path}")


# ══════════════════════════════════════════════════════════════════════════════
# RÉSUMÉ CONSOLE
# ══════════════════════════════════════════════════════════════════════════════

def print_summary(results: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("RÉSUMÉ — Wilcoxon + Holm (α=0.05)")
    print("=" * 60)
    for cat in CATEGORIES:
        sub = results[results["category"] == cat]
        n_sig    = sub["significant"].sum()
        n_total  = sub["significant"].notna().sum()
        n_double = (sub["annotation"] == "★★").sum()
        print(f"\nCat. {cat}  ({sub['n_instances'].iloc[0] if len(sub) else '?'} instances) :"
              f"  {n_sig}/{n_total} paires significatives"
              f"  ({n_double} avec ★★)")
        for _, row in sub.iterrows():
            ann = row["annotation"]
            d   = row["direction"]
            print(f"  {row['pair_id']:4s} {row['method_A']:15s} vs {row['method_B']:15s}"
                  f"  p_holm={row['p_holm']:.4f}  {ann:3s}  {d}")

    # Critères GO/NO-GO
    print("\n" + "-" * 60)
    print("GO / NO-GO (SPEC §7)")
    sig_count = results.groupby("pair_id")["significant"].any().sum()
    p4_cats   = results[(results["pair_id"]=="P4") & results["significant"]]["category"].tolist()
    p5_c      = results[(results["pair_id"]=="P5") & (results["category"]=="C")]["significant"]
    p5_c_sig  = bool(p5_c.values[0]) if len(p5_c) else False

    print(f"  C1 (≥5 paires sig. au moins une cat.) : {sig_count}/8 → {'GO ✓' if sig_count>=5 else 'NO-GO ✗'}")
    print(f"  C2 (P4 sig. sur ≥2 cat.)              : {p4_cats} → {'GO ✓' if len(p4_cats)>=2 else 'NO-GO ✗'}")
    print(f"  C3 (P5 ★★ sur Cat. C)                 : {p5_c_sig} → {'GO ✓' if p5_c_sig else 'NO-GO ✗'}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Wilcoxon + Holm — G7")
    parser.add_argument("--input",  default="resultats_benchmark.csv")
    parser.add_argument("--pairs",  choices=["targeted", "all"], default="targeted")
    parser.add_argument("--out-csv", default="wilcoxon_results.csv")
    parser.add_argument("--out-tex", default="wilcoxon_table.tex")
    parser.add_argument("--out-tree", default="decision_tree_annotations.txt")
    args = parser.parse_args()

    pairs = PAIRS_TARGETED if args.pairs == "targeted" else PAIRS_ALL
    print(f"Mode : {args.pairs} ({len(pairs)} paires × {len(CATEGORIES)} catégories"
          f" = {len(pairs)*len(CATEGORIES)} tests)")

    # Chargement
    pivot = load_and_prepare(args.input)
    print(f"Données chargées : {len(pivot)} instances après déduplication")
    for cat in CATEGORIES:
        n = (pivot["category"] == cat).sum()
        print(f"  Cat. {cat} : {n} instances")

    # Tests par catégorie
    all_results = []
    for cat in CATEGORIES:
        pivot_cat = pivot[pivot["category"] == cat].copy()
        df_cat    = analyse_category(pivot_cat, pairs, cat)
        all_results.append(df_cat)

    results = pd.concat(all_results, ignore_index=True)

    # Sorties
    write_csv(results, args.out_csv)
    write_latex(results, args.out_tex, pairs)
    write_tree_annotations(results, args.out_tree)
    print_summary(results)


if __name__ == "__main__":
    main()
