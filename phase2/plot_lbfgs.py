#!/usr/bin/env python3
"""
plot_lbfgs.py — Figure §5 (B25)
=================================
Panneau gauche  : N_tot vs n  à κ=1e4  (5 courbes : Steihaug-CG, BFGS-Wolfe depuis A07
                                          + L-BFGS-{5,10,20} médiane 20 seeds)
Panneau droit   : Taux convergence vs κ à n=100 (L-BFGS-{5,10,20} uniquement —
                  quadratiques rotées SPEC_G4, non comparables au benchmark diagonal)

Sources :
  results_lbfgs.csv       — B24 (L-BFGS expérience G4)
  resultats_benchmark.csv — Phase 1 (Steihaug_CG, BFGS_Wolfe sur A07)

Sortie : figure_lbfgs_S5.pdf  +  figure_lbfgs_S5.png
"""

import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# ── Chemins ──────────────────────────────────────────────────────────────────
LBFGS_CSV    = "results_lbfgs.csv"
BENCH_CSV    = "resultats_benchmark.csv"
OUT_PDF      = "figure_lbfgs_S5.pdf"
OUT_PNG      = "figure_lbfgs_S5.png"

# ── Palette (cohérente avec article conférence) ───────────────────────────────
COLORS = {
    "Steihaug_CG": "#1F77B4",    # bleu
    "BFGS_Wolfe":  "#FF7F0E",    # orange
    "LBFGS-5":     "#2CA02C",    # vert
    "LBFGS-10":    "#D62728",    # rouge
    "LBFGS-20":    "#9467BD",    # violet
}
MARKERS = {
    "Steihaug_CG": "s",
    "BFGS_Wolfe":  "D",
    "LBFGS-5":     "^",
    "LBFGS-10":    "o",
    "LBFGS-20":    "v",
}
LABELS = {
    "Steihaug_CG": "Steihaug–CG (A07)",
    "BFGS_Wolfe":  "BFGS–Wolfe (A07)",
    "LBFGS-5":     "L-BFGS, $m=5$",
    "LBFGS-10":    "L-BFGS, $m=10$",
    "LBFGS-20":    "L-BFGS, $m=20$",
}

# ── Style global ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "serif",
    "font.size":         10,
    "axes.titlesize":    11,
    "axes.labelsize":    10,
    "legend.fontsize":   8.5,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "lines.linewidth":   1.8,
    "lines.markersize":  7,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.dpi":        150,
})

KMAX         = 1000
NS_B         = [10, 50, 100, 500, 1000]
KAPPAS_A     = [1e2, 1e3, 1e4, 1e5]
N_PANEL_A    = 100      # n fixé pour panneau A
KAPPA_PANEL_B = 1e4     # κ fixé pour panneau B


# ══════════════════════════════════════════════════════════════════════════════
# CHARGEMENT ET PRÉPARATION DES DONNÉES
# ══════════════════════════════════════════════════════════════════════════════

def load_lbfgs(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["method"] = df["method"].str.strip()
    return df


def load_benchmark_a07(path: str) -> pd.DataFrame:
    """Extrait Steihaug_CG et BFGS_Wolfe sur A07 (quadratique géométrique κ=1e4)."""
    df = pd.read_csv(path)
    a07 = df[(df["problem"] == "A07") &
             (df["method"].isin(["Steihaug_CG", "BFGS_Wolfe"]))].copy()
    a07 = a07[["method", "n", "n_total", "converged"]].reset_index(drop=True)
    return a07


def aggregate_panel_B(df_lbfgs: pd.DataFrame) -> pd.DataFrame:
    """Médiane N_tot sur 20 seeds, Panel B (n-sweep, κ=1e4)."""
    pB = df_lbfgs[(df_lbfgs["panel"] == "B")].copy()
    agg = (pB.groupby(["method", "n"])
             .agg(ntot_median=("ntot", "median"),
                  ntot_q25=("ntot", lambda x: x.quantile(0.25)),
                  ntot_q75=("ntot", lambda x: x.quantile(0.75)),
                  conv_rate=("converged", "mean"))
             .reset_index())
    return agg


def aggregate_panel_A(df_lbfgs: pd.DataFrame) -> pd.DataFrame:
    """Taux convergence vs κ, Panel A (κ-sweep, n=100)."""
    pA = df_lbfgs[(df_lbfgs["panel"] == "A")].copy()
    agg = (pA.groupby(["method", "kappa"])
             .agg(conv_rate=("converged", "mean"),
                  iter_median=("iter_outer", "median"))
             .reset_index())
    return agg


# ══════════════════════════════════════════════════════════════════════════════
# PANNEAU GAUCHE — N_tot vs n  (κ = 1e4)
# ══════════════════════════════════════════════════════════════════════════════

def plot_panel_B(ax, df_pB: pd.DataFrame, df_a07: pd.DataFrame) -> None:
    """Panneau B : N_tot médian vs n à κ = 1e4.
    Points ouverts (non remplis) = budget épuisé (conv_rate < 0.5).
    """
    # ── Méthodes référence (A07, instance unique) ─────────────────────────────
    for method in ["Steihaug_CG", "BFGS_Wolfe"]:
        sub = df_a07[df_a07["method"] == method].sort_values("n")
        ns       = sub["n"].values
        ntots    = sub["n_total"].values
        convs    = sub["converged"].values

        # séparer converged / non-converged
        mask_ok  = convs
        mask_bad = ~convs

        # ligne continue sur toutes les valeurs
        ax.plot(ns, ntots,
                color=COLORS[method],
                marker=MARKERS[method],
                linestyle="-",
                label=LABELS[method],
                zorder=3)

        # marqueurs ouverts pour non-convergés
        if mask_bad.any():
            ax.plot(ns[mask_bad], ntots[mask_bad],
                    marker=MARKERS[method],
                    color="white",
                    markeredgecolor=COLORS[method],
                    markeredgewidth=1.8,
                    linestyle="none",
                    markersize=9,
                    zorder=4)

    # ── L-BFGS (médiane ± IQR, 20 seeds) ─────────────────────────────────────
    for method in ["LBFGS-5", "LBFGS-10", "LBFGS-20"]:
        sub = df_pB[df_pB["method"] == method].sort_values("n")
        ns        = sub["n"].values
        med       = sub["ntot_median"].values
        q25       = sub["ntot_q25"].values
        q75       = sub["ntot_q75"].values
        conv_rate = sub["conv_rate"].values

        mask_ok  = conv_rate >= 0.5   # majorité convergée
        mask_bad = ~mask_ok

        ax.plot(ns, med,
                color=COLORS[method],
                marker=MARKERS[method],
                linestyle="--",
                label=LABELS[method],
                zorder=3)

        # IQR en ruban (seulement où majorité converge)
        if mask_ok.any():
            ax.fill_between(ns[mask_ok], q25[mask_ok], q75[mask_ok],
                            color=COLORS[method], alpha=0.12, zorder=2)

        # marqueurs ouverts pour points budget-épuisé (majorité)
        if mask_bad.any():
            ax.plot(ns[mask_bad], med[mask_bad],
                    marker=MARKERS[method],
                    color="white",
                    markeredgecolor=COLORS[method],
                    markeredgewidth=1.8,
                    linestyle="none",
                    markersize=9,
                    zorder=4)

    # ── Ligne budget ──────────────────────────────────────────────────────────
    # Budget approximatif = KMAX × coût moyen par iter (~9-10 oracles) — indiquer visuellement
    ax.axhline(y=9000, color="gray", linestyle=":", linewidth=1.2, alpha=0.7)
    ax.text(110, 7800, f"Budget ($K_{{\\max}}$={KMAX})", fontsize=7.5,
            color="gray", va="top", ha="left")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Dimension $n$")
    ax.set_ylabel("Median $N_{\\mathrm{tot}}$")
    ax.set_title(r"(a) $N_{\rm tot}$ vs $n$ — $\kappa = 10^4$")
    ax.set_xticks(NS_B)
    ax.set_xticklabels([str(n) for n in NS_B])
    ax.set_xlim(7, 1500)

    # Légende
    # marqueur ouvert = budget épuisé
    dummy_bad = Line2D([0], [0], marker="o", color="w",
                       markeredgecolor="gray", markeredgewidth=1.5,
                       markersize=7, label="Budget exhausted ($K_{\\max}$)")
    handles, lbs = ax.get_legend_handles_labels()
    ax.legend(handles + [dummy_bad], lbs + ["Budget exhausted"],
              bbox_to_anchor=(120, 700),
              bbox_transform=ax.transData,
              loc="upper left",
              framealpha=0.92, edgecolor="0.8",
              ncol=1, fontsize=8)

    ax.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.5)
    ax.minorticks_off()


# ══════════════════════════════════════════════════════════════════════════════
# PANNEAU DROIT — Taux convergence vs κ  (n = 100)
# ══════════════════════════════════════════════════════════════════════════════

def plot_panel_A(ax, df_pA: pd.DataFrame) -> None:
    """Panneau A : taux convergence (%) vs κ à n=100, L-BFGS uniquement.

    Note : quadratiques rotées SPEC_G4 ≠ problèmes diagonaux benchmark —
           Steihaug_CG et BFGS_Wolfe non superposés (classes différentes).
    """
    kappas_plot = np.array(KAPPAS_A)

    for method in ["LBFGS-5", "LBFGS-10", "LBFGS-20"]:
        sub = df_pA[df_pA["method"] == method].sort_values("kappa")
        kap   = sub["kappa"].values
        conv  = sub["conv_rate"].values * 100.0   # en %

        ax.plot(kap, conv,
                color=COLORS[method],
                marker=MARKERS[method],
                linestyle="--",
                label=LABELS[method])

        # Annoter les valeurs 0% et 100%
        for k, c in zip(kap, conv):
            if c in (0.0, 100.0):
                ax.annotate(f"{c:.0f}%",
                            xy=(k, c), xytext=(0, 6),
                            textcoords="offset points",
                            ha="center", fontsize=7, color=COLORS[method])

    # Ligne seuil à κ = 1e4
    ax.axvline(x=1e4, color="gray", linestyle=":", linewidth=1.2, alpha=0.7)
    ax.text(1e4 * 1.15, 52, "$\\kappa = 10^4$\n(threshold)", fontsize=7.5,
            color="gray", va="center")

    ax.set_xscale("log")
    ax.set_xlabel("Condition number $\\kappa$")
    ax.set_ylabel("Convergence rate (%)")
    ax.set_title(r"(b) Convergence vs $\kappa$ — $n = 100$")
    ax.set_xticks(KAPPAS_A)
    ax.set_xticklabels([r"$10^2$", r"$10^3$", r"$10^4$", r"$10^5$"])
    ax.set_ylim(-5, 110)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])

    ax.legend(loc="center left", framealpha=0.9, edgecolor="0.8", fontsize=8)
    ax.grid(True, which="major", linestyle=":", linewidth=0.5, alpha=0.5)

    ax.annotate(
        "Rotated quadratics (SPEC G4)\n(Steihaug-CG / BFGS-Wolfe\nnot overlaid — distinct families)",
        xy=(0.97, 0.05), xycoords="axes fraction",
        ha="right", va="bottom", fontsize=7,
        color="0.5",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.8", alpha=0.8)
    )


# ══════════════════════════════════════════════════════════════════════════════
# ASSEMBLAGE FIGURE
# ══════════════════════════════════════════════════════════════════════════════

def make_figure(lbfgs_path: str = LBFGS_CSV,
                bench_path: str = BENCH_CSV,
                out_pdf: str   = OUT_PDF,
                out_png: str   = OUT_PNG) -> None:

    # Chargement
    df_lbfgs = load_lbfgs(lbfgs_path)
    df_a07   = load_benchmark_a07(bench_path)

    # Agrégation
    df_pB = aggregate_panel_B(df_lbfgs)
    df_pA = aggregate_panel_A(df_lbfgs)

    print("Panel B — N_tot médian L-BFGS (κ=1e4):")
    print(df_pB.pivot(index="n", columns="method", values="ntot_median").to_string())
    print()
    print("Panel B — N_tot A07 (Steihaug_CG, BFGS_Wolfe):")
    print(df_a07.to_string(index=False))
    print()
    print("Panel A — taux convergence L-BFGS (n=100):")
    print(df_pA.pivot(index="kappa", columns="method", values="conv_rate")
               .map(lambda x: f"{100*x:.0f}%").to_string())

    # Figure
    fig, (ax_B, ax_A) = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.subplots_adjust(wspace=0.35, left=0.08, right=0.97,
                        top=0.90, bottom=0.13)

    plot_panel_B(ax_B, df_pB, df_a07)
    plot_panel_A(ax_A, df_pA)

    fig.suptitle(
        "Figure §5 — L-BFGS vs Steihaug–CG / BFGS–Wolfe on ill-conditioned problems (Cat. A)",
        fontsize=10.5, fontweight="bold", y=0.98
    )

    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight", dpi=200)
    print(f"\nFigure enregistrée : {out_pdf}  |  {out_png}")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    lbfgs_path = sys.argv[1] if len(sys.argv) > 1 else LBFGS_CSV
    bench_path = sys.argv[2] if len(sys.argv) > 2 else BENCH_CSV
    make_figure(lbfgs_path, bench_path)
