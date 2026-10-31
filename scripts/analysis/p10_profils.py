# -*- coding: utf-8 -*-
"""
Consigne 010, priorite 1 : profils de performance (Dolan-More) et profils de donnees (More-Wild) sur v0.5.

DEFINITIONS (citees du manuscrit article_SciAfrican_R1_v05.tex, l. 412-413 et 738-777, et de analysis/data_profiles.py du depot) :
  - ensemble P : les instances DISTINCTES (probleme, dimension) : |P| = 156 (global), 59 (A), 55 (B), 42 (C) (legende l. 765-775) ;
  - cout : le cout d'oracle unifie N_tot = N_f + N_g + N_h (colonne n_total, N_tot v0.5, comptage avec memoire) ;
  - succes / critere de convergence : status == "success" (norme du gradient <= 1e-6 dans K_max = 1000 iterations) ;
  - profil de performance (l. 741-746) : pour chaque methode, fraction des problemes resolus a un facteur tau du meilleur N_tot observe
    sur ce probleme (parmi les six solveurs ayant reussi) ; la valeur en tau = 1 est la frequence de victoire, le plateau (tau -> infini)
    est le taux de succes ; une instance non resolue par une methode n'est jamais comptee pour elle ;
  - profil de donnees (l. 747-751 ; data_profiles.py) : alpha_{p,s} = N_tot / (n + 1) si convergence, infini sinon ;
    d_s(alpha) = |{p : alpha_{p,s} <= alpha}| / |P| ; axes : alpha en echelle logarithmique, fraction des problemes resolus.
Le fichier d'origine fig_data_profiles.pdf (inchange depuis la v1) ne contient en fait que les quatre panneaux de profils de donnees
(Global, A, B, C) ; la legende et le texte decrivent aussi des profils de performance : la figure v0.5 porte les deux, par ligne
(Global, A, B, C), profil de performance a gauche et profil de donnees a droite (legende l. 645 de la v1 : « Performance profiles
(Dolan-More, left) and data profiles (More-Wild, right) per category »).

Sorties : figures/fig_profiles_v05.pdf (+ .png), fig_profiles_v05_courbes.csv (points des echelons), fig_profiles_v05_resume.csv.
Aucune execution de solveur ; aucun nombre n'est tape a la main.
"""
import csv
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib._mathtext as _mathtext
_mathtext.SHRINK_FACTOR = 1.0  # exposants et indices au meme corps que le texte (defaut 0,7 -> 5,6 pt) : aucun texte sous 8 pt
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

from common_data import load_distinct_rows, METHODS
from paths import FIG_DIR, OUT_DIR

TEXTWIDTH_PT = 466.62503
FIG_W = TEXTWIDTH_PT / 72.27
MIN_FONT = 8.0
matplotlib.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "DejaVu Sans", "font.size": 8.5, "axes.labelsize": 8.5,
    "axes.titlesize": 8.5, "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8, "mathtext.default": "regular",
    "axes.linewidth": 0.8,
})
# styles de l'original (couleur, trait, epaisseur) : memes que fig_data_profiles.pdf de la v1
STYLE = {
    "Cauchy_Point": ("#6D4C41", "-", 1.0, "Cauchy Point"),
    "Dogleg": ("#2E7D32", "--", 1.4, "Dogleg"),
    "Steihaug_CG": ("#1565C0", "-", 2.2, "Steihaug-CG"),
    "BFGS_Armijo": ("#E65100", "-.", 1.4, "BFGS-Armijo"),
    "BFGS_Wolfe": ("#C62828", "-", 2.0, "BFGS-Wolfe"),
    "Newton_BT": ("#6A1B9A", "-", 2.2, "Newton-LM"),
}
SCOPES = [("Global", None, "Global"), ("A", "A", "Cat. A"), ("B", "B", "Cat. B"), ("C", "C", "Cat. C")]


def compute(rows):
    """Retourne {scope: {...}} avec les echelons des deux profils, |P| et les resumes."""
    inst = {}
    for r in rows:
        inst.setdefault((r["problem"], r["n"]), {})[r["method"]] = r
    res = {}
    for scope, cat, _ in SCOPES:
        keys = sorted(k for k, d in inst.items() if cat is None or next(iter(d.values()))["category"] == cat)
        P = len(keys)
        ratios = {m: [] for m in METHODS}
        alphas = {m: [] for m in METHODS}
        unsolved_all = 0
        for k in keys:
            d = inst[k]
            ok = {m: int(d[m]["n_total"]) for m in METHODS if d[m]["status"] == "success"}
            if not ok:
                unsolved_all += 1
            best = min(ok.values()) if ok else None
            n = int(k[1])
            for m, nt in ok.items():
                ratios[m].append(nt / best)
                alphas[m].append(nt / (n + 1))
        res[scope] = {"P": P, "ratios": {m: sorted(v) for m, v in ratios.items()}, "alphas": {m: sorted(v) for m, v in alphas.items()},
                      "unsolved_by_all": unsolved_all}
    return res


def steps(xs, P):
    """points (x, y) de la fonction en echelons : y = (nombre de valeurs <= x) / P."""
    return [(x, (i + 1) / P) for i, x in enumerate(xs)]


def check_fonts(fig):
    fig.canvas.draw()
    small = [(t.get_text(), t.get_fontsize()) for t in fig.findobj(matplotlib.text.Text)
             if t.get_visible() and t.get_text().strip() and t.get_fontsize() < MIN_FONT - 1e-9]
    assert not small, small


def main():
    rows = load_distinct_rows()
    assert len({(r["problem"], r["n"]) for r in rows}) == 156
    res = compute(rows)
    all_r = [x for s in res.values() for m in METHODS for x in s["ratios"][m]]
    all_a = [x for s in res.values() for m in METHODS for x in s["alphas"][m]]
    tau_max, a_min, a_max = max(all_r), min(all_a), max(all_a)
    tau_hi = 10 ** math.ceil(math.log10(tau_max))
    a_lo, a_hi = 10 ** math.floor(math.log10(a_min)), 10 ** math.ceil(math.log10(a_max))

    fig, axes = plt.subplots(4, 2, figsize=(FIG_W, 8.3), sharey=True)
    curves, summary = [], []
    for i, (scope, cat, title) in enumerate(SCOPES):
        s = res[scope]
        P = s["P"]
        axP, axD = axes[i]
        for m in METHODS:
            col, ls, lw, lab = STYLE[m]
            rr, aa = s["ratios"][m], s["alphas"][m]
            # profil de performance : de tau = 1 (inclus) a tau_hi
            pts = steps(rr, P)
            n_win = sum(1 for x in rr if x <= 1.0 + 1e-12)
            xs = [1.0] + [x for x, _ in pts] + [tau_hi]
            ys = [n_win / P] + [y for _, y in pts] + [len(rr) / P]
            axP.step(xs, ys, where="post", color=col, ls=ls, lw=lw, label=lab)
            for x, y in pts:
                curves.append({"profil": "performance", "portee": scope, "methode": m, "x_tau": f"{x:.6g}", "y_fraction": f"{y:.6f}", "P": P})
            # profil de donnees
            ptsd = steps(aa, P)
            xs = [a_lo] + [x for x, _ in ptsd] + [a_hi]
            ys = [0.0] + [y for _, y in ptsd] + [len(aa) / P]
            axD.step(xs, ys, where="post", color=col, ls=ls, lw=lw, label=lab)
            for x, y in ptsd:
                curves.append({"profil": "donnees", "portee": scope, "methode": m, "x_alpha": f"{x:.6g}", "y_fraction": f"{y:.6f}", "P": P})
            summary.append({"portee": scope, "methode": m, "P": P, "n_resolues": len(rr), "plateau": f"{len(rr) / P:.6f}",
                            "frequence_de_victoire_tau1": f"{n_win / P:.6f}", "n_instances_non_resolues_par_aucun_solveur": s["unsolved_by_all"]})
        axP.set_xscale("log")
        axD.set_xscale("log")
        axP.set_xlim(1.0, tau_hi)
        axD.set_xlim(a_lo, a_hi)
        axP.set_ylim(-0.02, 1.03)
        axD.set_ylim(-0.02, 1.03)
        for ax in (axP, axD):
            ax.grid(True, which="major", ls=":", lw=0.5, alpha=0.5)
            ax.xaxis.set_major_locator(ticker.LogLocator(base=10, numticks=12))
            ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
        axP.set_title(f"{title} (|P| = {P}): performance profile", fontsize=8, loc="left", fontweight="bold")
        axD.set_title(f"{title} (|P| = {P}): data profile", fontsize=8, loc="left", fontweight="bold")
        axP.set_ylabel("Fraction of problems solved")
        if i == 3:
            axP.set_xlabel(r"$\tau$ (ratio of $N_{\mathrm{tot}}$ to the best solver)")
            axD.set_xlabel(r"$\alpha = N_{\mathrm{tot}}\,/\,(n+1)$")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.0), fontsize=8, columnspacing=1.6)
    fig.subplots_adjust(left=0.115, right=0.985, top=0.975, bottom=0.115, hspace=0.42, wspace=0.12)
    check_fonts(fig)
    os.makedirs(FIG_DIR, exist_ok=True)
    fig.savefig(os.path.join(FIG_DIR, "fig_profiles_v05.pdf"))
    fig.savefig(os.path.join(FIG_DIR, "fig_profiles_v05.png"), dpi=170)
    plt.close(fig)
    with open(os.path.join(FIG_DIR, "fig_profiles_v05_courbes.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["profil", "portee", "methode", "x_tau", "x_alpha", "y_fraction", "P"])
        w.writeheader()
        for c in curves:
            c.setdefault("x_tau", "")
            c.setdefault("x_alpha", "")
            w.writerow(c)
    with open(os.path.join(FIG_DIR, "fig_profiles_v05_resume.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)
    print(f"|P| : {[res[s]['P'] for s, _, _ in SCOPES]} ; instances non resolues par aucun solveur : "
          f"{[res[s]['unsolved_by_all'] for s, _, _ in SCOPES]} ; tau_max = {tau_max:.1f} (axe {tau_hi:g}) ; alpha dans [{a_min:.4g}, {a_max:.4g}] (axe {a_lo:g} a {a_hi:g})")
    for r in summary:
        if r["portee"] in ("Global",):
            print(r)


if __name__ == "__main__":
    main()
