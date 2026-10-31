# -*- coding: utf-8 -*-
"""
Consigne 010, priorite 2 : figure de scalabilite sur v0.5 (remplace fig_scalability.pdf).

DEFINITION D'ORIGINE (article_SciAfrican_R1_v05.tex, legende l. 830-835 ; texte l. 784-825) : « Median oracle cost N_tot (left) and
wall-clock CPU time (right) vs. dimension n, Category A, all six methods » ; axes : dimension n (echelle logarithmique, graduations
10, 50, 100, 500, 1000), mediane en echelle logarithmique ; une courbe par methode ; legende sous la figure ; titre integre
« Category A (ill-conditioned), all six methods » (supprime ici : la legende du manuscrit le dit).
Le manuscrit ne precise ni l'ensemble des problemes ni l'agregation : ils ont ete retrouves en recalculant la figure d'origine depuis
les donnees d'origine (revision/sources/donnees_code/APNL_3_Annexe_2026-04-12_resultats_benchmark.csv, colonnes n_total et cpu_s) :
POPULATION = instances de Categorie A a la dimension litterale n in {10, 50, 100, 500, 1000} (10 problemes a n = 10 et 50, 11 a n = 100,
500 et 1000 ; les problemes a dimension fixe A06, A08, A11, A13 et A09 (n = 8 et 48) n'y figurent donc pas), TOUTES les executions
(reussies ou non : une execution echouee compte son N_tot et son temps a K_max), AGREGATION = mediane sur les instances de la cellule.
Verification : les medianes d'origine (Cauchy Point 4002 ; BFGS-Armijo 169,5 / 777,5 / 1788 / 5589 / 6665 ; BFGS-Wolfe 93 / 444 / 849 /
4502 / 4502 ; Steihaug-CG 57 / 75 / 87 / 99 / 103 ; Dogleg 15 / 19 / 19 / 23 / 27 pour N_tot ; Cauchy Point 0,049 / 0,081 / 0,27 / 0,95 /
3,6 s en CPU) retrouvent les valeurs lues sur la figure d'origine.

REFAIT EN v0.5 : N_tot de resultats_benchmark_v0.5.csv (comptage avec memoire, 1080 lignes, premiere occurrence de chaque instance
distincte) ; temps = TEMPS REEL du passage A (mediane des repetitions de chaque tache, wall_perf_counter_s) au lieu du temps CPU du
benchmark ; mediane sur les instances de la cellule ; les cellules qui contiennent au moins une tache de temps reel median inferieur a 1 ms sont MARQUEES (marqueur creux ; aucune mediane n'est inferieure a 1 ms).
Ecart de definition (justifie) : le temps CPU du fichier de resultats n'est plus utilise (il provient de l'execution instrumentee du
benchmark) ; le temps reel du passage A est la mesure principale de 009e.

Sorties : figures/fig_scalability_v05.pdf (+ .png) et fig_scalability_v05_points.csv. Aucun nombre tape a la main.
"""
import csv
import os
import statistics as st

import matplotlib
matplotlib.use("Agg")
import matplotlib._mathtext as _mathtext
_mathtext.SHRINK_FACTOR = 1.0  # exposants et indices au meme corps que le texte (defaut 0,7 -> 5,6 pt) : aucun texte sous 8 pt
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D

from common_data import load_distinct_rows, METHODS
from common_timing import get_timing, LOW_PREC_S
from paths import FIG_DIR

TEXTWIDTH_PT = 466.62503
FIG_W = TEXTWIDTH_PT / 72.27
MIN_FONT = 8.0
DIMS = [10, 50, 100, 500, 1000]
matplotlib.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "DejaVu Sans", "font.size": 8.5, "axes.labelsize": 8.5,
    "axes.titlesize": 8.5, "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8, "mathtext.default": "regular",
    "axes.linewidth": 0.8,
})
STYLE = {  # couleur, trait, marqueur, libelle
    "Cauchy_Point": ("#6D4C41", "--", "x", "Cauchy Point"),
    "Dogleg": ("#2E7D32", "--", "D", "Dogleg"),
    "Steihaug_CG": ("#1565C0", "-", "o", "Steihaug-CG"),
    "BFGS_Armijo": ("#E65100", "-.", "v", "BFGS-Armijo"),
    "BFGS_Wolfe": ("#C62828", "-", "^", "BFGS-Wolfe"),
    "Newton_BT": ("#6A1B9A", "-", "s", "Newton-LM"),
}


def check_fonts(fig):
    fig.canvas.draw()
    small = [(t.get_text(), t.get_fontsize()) for t in fig.findobj(matplotlib.text.Text)
             if t.get_visible() and t.get_text().strip() and t.get_fontsize() < MIN_FONT - 1e-9]
    assert not small, small


def main():
    rows = load_distinct_rows()
    wall = get_timing("principale").wall
    pts = []
    for m in METHODS:
        for n in DIMS:
            cell = [r for r in rows if r["category"] == "A" and r["method"] == m and int(r["n"]) == n]
            ntot = [int(r["n_total"]) for r in cell]
            times = [wall[(r["problem"], r["n"], m)] for r in cell]  # toutes les taches de Categorie A sont mesurees (verifie)
            base = {"methode": m, "n": n, "n_instances": len(cell), "n_echecs": sum(r["status"] != "success" for r in cell),
                    "problemes": ";".join(sorted(r["problem"] for r in cell))}
            pts.append({**base, "panneau": "N_tot", "mediane": st.median(ntot), "n_taches_sous_1ms": "", "mediane_sous_1ms": "",
                        "faible_precision": ""})
            lowp = sum(t < LOW_PREC_S for t in times)
            med_t = st.median(times)
            pts.append({**base, "panneau": "temps_reel_passage_A_s", "mediane": med_t, "n_taches_sous_1ms": lowp,
                        "mediane_sous_1ms": med_t < LOW_PREC_S, "faible_precision": lowp > 0})
    fig, (axN, axT) = plt.subplots(1, 2, figsize=(FIG_W, 3.15))
    fig.subplots_adjust(left=0.105, right=0.985, top=0.965, bottom=0.30, wspace=0.27)
    for ax, panel, ylab in ((axN, "N_tot", r"Median $N_{\mathrm{tot}}$"), (axT, "temps_reel_passage_A_s", "Median wall time (s)")):
        for m in METHODS:
            col, ls, mk, lab = STYLE[m]
            sel = sorted((p for p in pts if p["methode"] == m and p["panneau"] == panel), key=lambda p: p["n"])
            xs = [p["n"] for p in sel]
            ys = [p["mediane"] for p in sel]
            ax.plot(xs, ys, color=col, ls=ls, lw=1.5, zorder=2)
            for p in sel:
                hollow = panel != "N_tot" and p["faible_precision"]
                ms = {"x": 6.5, "D": 8.0, "s": 4.5}.get(mk, 5.5)  # Dogleg plus grand, Newton-LM plus petit : visibles si superposes
                ax.plot([p["n"]], [p["mediane"]], mk, color=col, ms=ms, mfc="white" if hollow else col,
                        mec=col, mew=1.2, zorder=2.5 if mk == "D" else 3.5)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks(DIMS)
        ax.get_xaxis().set_major_formatter(ticker.FixedFormatter([str(d) for d in DIMS]))
        ax.xaxis.set_minor_formatter(ticker.NullFormatter())
        ax.set_xlim(8, 1250)
        ax.set_xlabel("Dimension $n$")
        ax.set_ylabel(ylab)
        ax.grid(True, which="major", ls=":", lw=0.5, alpha=0.6)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    handles = [Line2D([0], [0], color=STYLE[m][0], ls=STYLE[m][1], lw=1.5, marker=STYLE[m][2], ms=5.5, label=STYLE[m][3]) for m in METHODS]
    handles.append(Line2D([0], [0], color="#444444", ls="", marker="o", ms=5.5, mfc="white", mec="#444444", mew=1.2,
                          label="cell with a task < 1 ms (right panel)"))
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.0), fontsize=8, columnspacing=1.4)
    check_fonts(fig)
    os.makedirs(FIG_DIR, exist_ok=True)
    fig.savefig(os.path.join(FIG_DIR, "fig_scalability_v05.pdf"))
    fig.savefig(os.path.join(FIG_DIR, "fig_scalability_v05.png"), dpi=200)
    plt.close(fig)
    with open(os.path.join(FIG_DIR, "fig_scalability_v05_points.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["panneau", "methode", "n", "n_instances", "n_echecs", "mediane", "n_taches_sous_1ms",
                                           "mediane_sous_1ms", "faible_precision", "problemes"])
        w.writeheader()
        for p in pts:
            q = dict(p)
            q["mediane"] = f"{p['mediane']:.6g}"
            w.writerow(q)
    cnt = {}
    for p in pts:
        if p["panneau"] == "N_tot":
            cnt.setdefault(p["n"], set()).add(p["n_instances"])
    print("instances par dimension :", {n: sorted(v) for n, v in cnt.items()})
    for p in pts:
        if p["methode"] in ("Newton_BT", "Steihaug_CG") and p["panneau"] != "N_tot":
            print(p["methode"], p["n"], f"{p['mediane']:.5f}", p["n_taches_sous_1ms"], p["faible_precision"])


if __name__ == "__main__":
    main()
