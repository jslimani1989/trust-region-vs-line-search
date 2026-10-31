# -*- coding: utf-8 -*-
"""
Consigne 009e, priorite 7.3 : figures en anglais pour le manuscrit.
PDF vectoriel, polices incorporees (Type 42 / TrueType), AUCUN titre integre a la figure, largeur = largeur de texte du
manuscrit (\\the\\textwidth = 466.62503 pt, releve dans tables/compile_check.log), toutes les polices >= 8 pt (verifie par
une assertion sur chaque objet texte apres le dessin).

  fig_compromis_measured.pdf             : perte de succes en fonction du regret, temps mesures seuls (pour le texte)
  fig_compromis_measured_estimated.pdf   : meme figure, temps mesures + estimes (pour le supplement)
  fig_pareto.pdf                         : front de Pareto (succes x pente log-log du temps reel par iteration)
  fig_G3_manuscript.pdf                  : figure G3 (memes donnees et agregation que plot_G3.py : mediane et IQR sur 20 graines),
                                           depuis results_G3_v0.5_complet.csv, sans titre integre, a la largeur de texte

Les deux regles « robustesse d'abord » et « temps d'abord » ont des valeurs IDENTIQUES (n* non determinable : seuil infini) :
elles sont representees par UN seul marqueur etiquete des deux noms (pas de marqueurs superposes).
Lecture seule des CSV d'analyse ; aucun nombre nouveau.
"""
import csv
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib._mathtext as _mathtext
_mathtext.SHRINK_FACTOR = 1.0  # correction consigne 010b : exposants et indices au meme corps que le texte (defaut 0,7 -> 5,6 pt)
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from paths import OUT_DIR, FIG_DIR, G3_COMPLET_CSV

TEXTWIDTH_PT = 466.62503
FIG_W = TEXTWIDTH_PT / 72.27  # pouces
MIN_FONT = 8.0

matplotlib.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "DejaVu Sans", "font.size": 8.5,
    "axes.labelsize": 8.5, "axes.titlesize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "mathtext.default": "regular", "axes.linewidth": 0.8, "savefig.bbox": None,
})
INK, INK2, MUTED, GRID = "#0b0b0b", "#444444", "#898781", "#e1e0d9"


def check_fonts(fig):
    fig.canvas.draw()
    small = []
    for t in fig.findobj(matplotlib.text.Text):
        if t.get_visible() and t.get_text().strip() and t.get_fontsize() < MIN_FONT - 1e-9:
            small.append((t.get_text(), t.get_fontsize()))
    assert not small, f"polices < {MIN_FONT} pt : {small}"


def save(fig, stem):
    os.makedirs(FIG_DIR, exist_ok=True)
    check_fonts(fig)
    fig.savefig(os.path.join(FIG_DIR, stem + ".pdf"))
    fig.savefig(os.path.join(FIG_DIR, stem + ".png"), dpi=200)
    plt.close(fig)
    print("ecrit", stem)


# ---------------------------------------------------------------------------------------------- compromis
def fig_compromis(kind):
    """kind : 'measured' ou 'measured_estimated'."""
    with open(os.path.join(OUT_DIR, "b3_synthese.csv"), encoding="utf-8") as fh:
        rows = {r["regle"]: r for r in csv.DictReader(fh) if r["portee"] == "toutes"}
    kx, kci = (("regret_mesure_seul_geo", "regret_mesure_seul_IC95") if kind == "measured"
               else ("regret_mesure_estime_geo", "regret_mesure_estime_IC95"))
    naive, noeud = rows["naive"], rows["un_noeud"]
    # valeurs ponctuelles identiques ; les IC bootstrap ne different que par le bruit de reechantillonnage (flux aleatoire
    # distinct pour chaque regle) : on trace l'IC de « robustesse d'abord »
    assert naive[kx] == noeud[kx] and naive["perte_moyenne"] == noeud["perte_moyenne"], \
        "robustesse d'abord et temps d'abord ne sont plus identiques : revoir la figure"
    pts = [("naive", "Robustness first = time first\n($n^*$ not determinable)", "#eb6834", "s"),
           ("arbre", "Learned tree ($\\delta=0.05$)", "#2a78d6", "o"),
           ("article", "Article rule\n($n\\geq 500$ and C01$'$, C08$'$, C09$'$, C10$'$)", "#1baf7a", "D")]
    fig, ax = plt.subplots(figsize=(FIG_W, 3.1))
    ax.set_facecolor("white")
    xs, ys = [], []
    for key, label, col, mk in pts:
        r = rows[key]
        x, y = float(r[kx]), float(r["perte_moyenne"])
        xci, yci = json.loads(r[kci]), json.loads(r["perte_IC95"])
        ax.errorbar(x, y, xerr=[[x - xci[0]], [xci[1] - x]], yerr=[[y - yci[0]], [yci[1] - y]], fmt="none", ecolor=col,
                    elinewidth=1.3, alpha=0.6, capsize=0, zorder=2)
        ax.plot(x, y, mk, ms=8, mfc=col, mec="white", mew=1.5, zorder=3)
        xs += xci
        ys += yci
    # etiquettes directes (coordonnees de donnees) reliees au point par un trait fin
    lo_x, hi_x = min(xs), max(xs)
    x_naive = float(rows["naive"][kx])
    label_xy = {"naive": (x_naive, -0.0125), "arbre": (lo_x + 0.02 * (hi_x - lo_x), 0.040),
                "article": (lo_x + 0.50 * (hi_x - lo_x), 0.078)}
    for key, label, col, mk in pts:
        r = rows[key]
        x, y = float(r[kx]), float(r["perte_moyenne"])
        ax.annotate(label, (x, y), xytext=label_xy[key], textcoords="data", fontsize=8, color=INK2,
                    ha="center" if key == "naive" else "left", va="center",
                    arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.7, "shrinkA": 2, "shrinkB": 6})
    ax.set_xlabel("Time regret (geometric mean; 1 = fastest successful candidate)" +
                  (", measured times only" if kind == "measured" else ", measured + estimated times"))
    ax.set_ylabel("Mean success loss per problem")
    ax.grid(color=GRID, lw=0.7, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_ylim(-0.026, 0.092)
    ax.set_yticks([0.0, 0.02, 0.04, 0.06, 0.08])
    pad = 0.05 * (hi_x - lo_x)
    ax.set_xlim(lo_x - pad, hi_x + pad)
    fig.subplots_adjust(left=0.115, right=0.985, bottom=0.17, top=0.97)
    save(fig, "fig_compromis_" + kind)


# ---------------------------------------------------------------------------------------------- Pareto
def fig_pareto():
    with open(os.path.join(OUT_DIR, "b4_pareto_front.csv"), encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    MARK = {"Cauchy_Point": "v", "Dogleg": "<", "Steihaug_CG": "o", "BFGS_Armijo": "s", "BFGS_Wolfe": "D", "Newton_BT": "^"}
    LABEL = {"Cauchy_Point": "Cauchy Point", "Dogleg": "Dogleg", "Steihaug_CG": "Steihaug-CG", "BFGS_Armijo": "BFGS-Armijo",
             "BFGS_Wolfe": "BFGS-Wolfe", "Newton_BT": "Newton-LM"}
    # decalages d'etiquette en points (x, y) et alignement, choisis pour eviter tout recouvrement
    OFFS = {"Cauchy_Point": (-10, 6, "right"), "Dogleg": (12, -12, "left"), "Steihaug_CG": (-10, 12, "right"),
            "BFGS_Armijo": (-12, -3, "right"), "BFGS_Wolfe": (12, -12, "left"), "Newton_BT": (14, -9, "left")}
    FRONT, OTHER = "#2a78d6", "#898781"
    fig, ax = plt.subplots(figsize=(FIG_W, 3.4))
    for r in rows:
        m = r["method"]
        x, y = float(r["pente_temps_par_iteration"]), float(r["taux_succes"])
        xl, xh = float(r["IC95_bas_pente"]), float(r["IC95_haut_pente"])
        yl, yh = float(r["IC95_bas_succes"]), float(r["IC95_haut_succes"])
        front = r["sur_le_front"] == "True"
        col = FRONT if front else OTHER
        ax.errorbar(x, y, xerr=[[x - xl], [xh - x]], yerr=[[y - yl], [yh - y]], fmt="none", ecolor=col, elinewidth=1.3, alpha=0.55,
                    capsize=0, zorder=2)
        ax.plot(x, y, MARK[m], ms=8, mfc=col, mec="white", mew=1.5, zorder=3)
        dx, dy, ha = OFFS[m]
        ax.annotate(LABEL[m] + ("  (front)" if front else ""), (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=8,
                    color=INK if front else INK2, ha=ha, va="center")
    ax.set_xlabel("Empirical log–log slope of wall time per iteration (n = 100–1000)")
    ax.set_ylabel("Success rate (mean over 36 problems)")
    ax.grid(color=GRID, lw=0.7, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_ylim(0.2, 1.03)
    ax.set_xlim(1.2, 2.4)
    fig.subplots_adjust(left=0.115, right=0.985, bottom=0.15, top=0.97)
    save(fig, "fig_pareto")


# ---------------------------------------------------------------------------------------------- G3
STYLE = {
    "steihaug": dict(color="#1565C0", lw=2.0, ls="-", zorder=5, label="Steihaug-CG"),
    "bfgs_wolfe": dict(color="#C62828", lw=2.0, ls="-", zorder=4, label="BFGS–Wolfe"),
    "cauchy": dict(color="#6D4C41", lw=2.0, ls="-", zorder=3, label="Cauchy Point"),
    "dogleg": dict(color="#2E7D32", lw=1.2, ls="--", zorder=2, label="Dogleg"),
    "newton_bt": dict(color="#6A1B9A", lw=1.2, ls=":", zorder=2, label="Newton-LM"),
    "bfgs_armijo": dict(color="#E65100", lw=1.0, ls="-.", zorder=1, label="BFGS–Armijo"),
}
KMAX_G3 = 5000


def agg(df, x_col, x_vals):
    out = {}
    for method in STYLE:
        med, q25, q75 = [], [], []
        for xv in x_vals:
            sub = df[(df["method"] == method) & (df[x_col] == xv)]["iter_outer"]
            med.append(sub.median())
            q25.append(sub.quantile(0.25))
            q75.append(sub.quantile(0.75))
        out[method] = dict(x=np.array(x_vals, dtype=float), med=np.array(med), q25=np.array(q25), q75=np.array(q75))
    return out


def draw_panel(ax, data, x_vals, x_label):
    for method, st in STYLE.items():
        d = data[method]
        ax.plot(d["x"], d["med"], **st)
        ax.fill_between(d["x"], d["q25"], d["q75"], color=st["color"], alpha=0.12)
    ax.axhline(KMAX_G3, color="black", lw=0.8, ls="--", alpha=0.4, zorder=0)
    ax.text(x_vals[0], KMAX_G3 * 1.08, f"$K_{{\\max}}$ = {KMAX_G3}", ha="left", va="bottom", fontsize=8, color="gray")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(0.7, 2.5e4)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Outer iterations (median)")
    ax.yaxis.set_minor_formatter(ticker.NullFormatter())
    ax.grid(True, which="major", ls="--", lw=0.4, alpha=0.5)
    ax.grid(True, which="minor", ls=":", lw=0.3, alpha=0.3)


def fig_g3():
    df = pd.read_csv(G3_COMPLET_CSV)
    A, B = df[df["panel"] == "A"], df[df["panel"] == "B"]
    kappas, ns = sorted(A["kappa"].unique()), sorted(B["n"].unique())
    data_A, data_B = agg(A, "kappa", kappas), agg(B, "n", ns)
    fig, (axB, axA) = plt.subplots(1, 2, figsize=(FIG_W, 3.3))
    fig.subplots_adjust(left=0.095, right=0.985, top=0.93, bottom=0.30, wspace=0.30)
    draw_panel(axB, data_B, ns, "Dimension $n$  ($\\kappa = 10^4$ fixed)")
    axB.set_title("(a) Scalability in $n$", fontsize=9, fontweight="bold", pad=4)
    x_last = ns[-1]
    st_last, bfw_last = data_B["steihaug"]["med"][-1], data_B["bfgs_wolfe"]["med"][-1]
    axB.annotate("", xy=(x_last * 1.03, st_last), xytext=(x_last * 1.03, bfw_last),
                 arrowprops=dict(arrowstyle="<->", color="#444444", lw=1.0), annotation_clip=False)
    axB.text(x_last * 1.07, math.sqrt(st_last * bfw_last), f"×{bfw_last / st_last:.0f}", fontsize=8, va="center", color="#444444",
             clip_on=False)
    st10, st1000 = data_B["steihaug"]["med"][0], data_B["steihaug"]["med"][-1]
    axB.text(ns[0] * 1.15, 2.2, f"Steihaug-CG: ×{st1000 / st10:.1f} over $n\\in[10,1000]$", fontsize=8,
             color=STYLE["steihaug"]["color"], ha="left", va="center")
    draw_panel(axA, data_A, kappas, "Conditioning $\\kappa$  ($n = 100$ fixed)")
    axA.set_title("(b) Sensitivity to conditioning $\\kappa$", fontsize=9, fontweight="bold", pad=4)
    k5 = 1e5
    idx5 = list(kappas).index(k5)
    st5, bfw5 = data_A["steihaug"]["med"][idx5], data_A["bfgs_wolfe"]["med"][idx5]
    axA.annotate(f"×{bfw5 / st5:.1f} at $\\kappa=10^5$", xy=(k5, math.sqrt(st5 * bfw5)), xytext=(2e3, 34),
                 fontsize=8, color="#444444", va="center", arrowprops=dict(arrowstyle="->", color="#444444", lw=0.8))
    handles = [Line2D([0], [0], **{k: v for k, v in st.items() if k != "label"}, label=st["label"]) for st in STYLE.values()]
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.0), frameon=False, fontsize=8, columnspacing=1.6)
    save(fig, "fig_G3_manuscript")


def main():
    fig_compromis("measured")
    fig_compromis("measured_estimated")
    fig_pareto()
    fig_g3()


if __name__ == "__main__":
    main()
