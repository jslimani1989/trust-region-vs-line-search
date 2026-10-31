#!/usr/bin/env python3
"""
plot_G3.py -- Figure pivot G3 (deux panneaux), peuple Section 3 du papier.
Adapte de revision/R1_v02/plot_G3_en.py (libelles anglais, consigne 005/R3.10)
pour la consigne 008b (etape 1d/Execution B) : SEULE modification par
rapport a plot_G3_en.py, la legende de la courbe Newton est renommee
« Newton-LM » (au lieu de « Newton-BT »), pour que la figure emploie la
meme terminologie que le texte du manuscrit (coherent avec le constat du
rapport 005b S3.2 : le code sous-jacent est bien regularise Levenberg-
Marquardt, « Newton-BT » etait un nom trompeur). Aucune autre ligne de
calcul, de selection de donnees, de couleur ou de style n'est modifiee.

Entree : results_G3_v0.3.csv (produit par run_G3.py, cf. etape 1d).
Sortie : fig_G3_v0.3.pdf + .png (300 dpi).
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D

# -- Palette et style (inchange, sauf le libelle Newton -> "Newton-LM") -----
STYLE = {
    "steihaug":    dict(color="#1565C0", lw=2.5, ls="-",  zorder=5, label="Steihaug-CG"),
    "bfgs_wolfe":  dict(color="#C62828", lw=2.5, ls="-",  zorder=4, label="BFGS–Wolfe"),
    "cauchy":      dict(color="#6D4C41", lw=2.5, ls="-",  zorder=3, label="Cauchy Point"),
    "dogleg":      dict(color="#2E7D32", lw=1.2, ls="--", zorder=2, label="Dogleg"),
    "newton_bt":   dict(color="#6A1B9A", lw=1.2, ls=":",  zorder=2, label="Newton-LM"),
    "bfgs_armijo": dict(color="#E65100", lw=1.0, ls="-.", zorder=1, label="BFGS–Armijo"),
}
ALPHA_IQR = 0.12
KMAX_G3   = 5000

def agg(df, x_col, x_vals):
    """Median + Q25/Q75 par (method, x_val)."""
    records = {}
    for method in STYLE:
        med, q25, q75 = [], [], []
        for xv in x_vals:
            sub = df[(df["method"] == method) & (df[x_col] == xv)]["iter_outer"]
            med.append(sub.median())
            q25.append(sub.quantile(0.25))
            q75.append(sub.quantile(0.75))
        records[method] = dict(x=np.array(x_vals, dtype=float),
                               med=np.array(med), q25=np.array(q25), q75=np.array(q75))
    return records

def draw_panel(ax, data, x_vals, x_label, x_log=True,
               kmax_line=True, annotate_sep=None):
    """Trace les courbes + bandes IQR sur ax."""
    for method, st in STYLE.items():
        d = data[method]
        x = d["x"]
        ax.plot(x, d["med"], **st)
        ax.fill_between(x, d["q25"], d["q75"],
                        color=st["color"], alpha=ALPHA_IQR)

    if kmax_line:
        ax.axhline(KMAX_G3, color="black", lw=0.8, ls="--", alpha=0.4, zorder=0)
        ax.text(x_vals[-1], KMAX_G3 * 1.05, f"$K_{{\\max}}$ = {KMAX_G3}",
                ha="right", va="bottom", fontsize=7.5, color="gray")

    if x_log:
        ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(x_label, fontsize=10)
    ax.set_ylabel("Outer iterations (median)", fontsize=10)
    ax.yaxis.set_minor_formatter(ticker.NullFormatter())
    ax.grid(True, which="major", ls="--", lw=0.4, alpha=0.5)
    ax.grid(True, which="minor", ls=":",  lw=0.3, alpha=0.3)
    ax.tick_params(labelsize=8.5)

    if annotate_sep is not None:
        x_last = x_vals[-1]
        st_last  = data["steihaug"]["med"][-1]
        bfw_last = data["bfgs_wolfe"]["med"][-1]
        factor   = bfw_last / st_last
        ax.annotate("",
            xy=(x_last * 1.02, st_last), xytext=(x_last * 1.02, bfw_last),
            arrowprops=dict(arrowstyle="<->", color="#444444", lw=1.0),
            annotation_clip=False)
        ax.text(x_last * 1.04, np.sqrt(st_last * bfw_last),
                f"×{factor:.0f}", fontsize=8, va="center",
                color="#444444")

def build_figure(csv_path, out_stem):
    df = pd.read_csv(csv_path)

    df["cg_inner_total"] = pd.to_numeric(df["cg_inner_total"], errors="coerce")

    A = df[df["panel"] == "A"]
    B = df[df["panel"] == "B"]

    kappas = sorted(A["kappa"].unique())
    ns     = sorted(B["n"].unique())

    data_A = agg(A, "kappa", kappas)
    data_B = agg(B, "n",     ns)

    fig, (axB, axA) = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.subplots_adjust(wspace=0.32, left=0.08, right=0.88, top=0.88, bottom=0.14)

    # --- Panel B (headline) ---
    draw_panel(axB, data_B, ns,
               x_label="Dimension $n$  ($\\kappa = 10^4$ fixed)",
               x_log=True, kmax_line=True, annotate_sep=True)
    axB.set_title("(a)  Scalability in $n$  — headline panel",
                  fontsize=9.5, fontweight="bold", pad=6)
    st10   = data_B["steihaug"]["med"][0]
    st1000 = data_B["steihaug"]["med"][-1]
    axB.annotate(f"Steihaug: ×{st1000/st10:.1f} over $n\\in[10,1000]$",
                 xy=(ns[-1], st1000), xytext=(ns[1], st1000 * 2.5),
                 fontsize=7.5, color=STYLE["steihaug"]["color"],
                 arrowprops=dict(arrowstyle="->", color=STYLE["steihaug"]["color"],
                                 lw=0.8))

    # --- Panel A (support) ---
    draw_panel(axA, data_A, kappas,
               x_label="Conditioning $\\kappa$  ($n = 100$ fixed)",
               x_log=True, kmax_line=True, annotate_sep=False)
    axA.set_title("(b)  Sensitivity to conditioning $\\kappa$  — supporting panel",
                  fontsize=9.5, fontweight="bold", pad=6)
    k5    = 1e5
    idx5  = list(kappas).index(k5)
    st5   = data_A["steihaug"]["med"][idx5]
    bfw5  = data_A["bfgs_wolfe"]["med"][idx5]
    axA.annotate(f"×{bfw5/st5:.1f} at $\\kappa=10^5$",
                 xy=(k5, np.sqrt(st5 * bfw5)),
                 xytext=(k5 / 8, np.sqrt(st5 * bfw5) * 2),
                 fontsize=7.5, color="#444444",
                 arrowprops=dict(arrowstyle="->", color="#444444", lw=0.8))
    axA.text(kappas[0] * 1.2, 120,
             "BFGS: $n$-step termination\non quadratics (mild in $\\kappa$)",
             fontsize=6.5, color=STYLE["bfgs_wolfe"]["color"], alpha=0.85,
             va="top")

    legend_elements = [
        Line2D([0], [0], **{k: v for k, v in st.items() if k != "label"},
               label=st["label"])
        for method, st in STYLE.items()
    ]
    fig.legend(handles=legend_elements, loc="center right",
               bbox_to_anchor=(1.0, 0.5), fontsize=8.5,
               frameon=True, framealpha=0.9, edgecolor="#CCCCCC",
               title="Method", title_fontsize=8.5)

    fig.suptitle(
        "Figure G3 — Steihaug-CG: invariance in $n$ (×1.5) vs BFGS-Wolfe: growth in $n$ (×55)  "
        "·  $f(x)=\\frac{1}{2}x^\\top A x$, log-uniform spectrum, 20 draws, $K_{\\max}=5000$",
        fontsize=8.5, y=0.97, color="#333333"
    )

    for fmt in ("pdf", "png"):
        path = f"{out_stem}.{fmt}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Saved: {path}")

    print("\n=== CHIFFRES SEC.3 (verification) ===")
    print(f"Panel B - Steihaug  : n=10->{data_B['steihaug']['med'][0]:.0f} iters, "
          f"n=1000->{data_B['steihaug']['med'][-1]:.0f} iters "
          f"(x{data_B['steihaug']['med'][-1]/data_B['steihaug']['med'][0]:.2f})")
    print(f"Panel B - BFGS-Wolfe: n=10->{data_B['bfgs_wolfe']['med'][0]:.0f} iters, "
          f"n=1000->{data_B['bfgs_wolfe']['med'][-1]:.0f} iters "
          f"(x{data_B['bfgs_wolfe']['med'][-1]/data_B['bfgs_wolfe']['med'][0]:.1f})")

    plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="results_G3_v0.3.csv")
    ap.add_argument("--out",   default="fig_G3_v0.3")
    args = ap.parse_args()
    build_figure(args.input, args.out)

if __name__ == "__main__":
    main()
