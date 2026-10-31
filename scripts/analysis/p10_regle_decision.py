# -*- coding: utf-8 -*-
"""
Consigne 010, priorite 3 : nouvelle regle de decision (remplace fig_decision_tree_input.tex), conforme a la decision 2 de l'auteur :
un noeud (« Hessian available? »), deux feuilles (Newton-LM ; BFGS-Wolfe, alternative BFGS-Armijo) et un encadre
« Outside the tested range » (n > 1000, ou factorisation dense impossible en memoire ou en temps : Steihaug-CG, dont la pente du temps
par iteration est plus faible ; aucune donnee ne la soutient au-dela de n = 1000).
Les nombres ecrits dans le fichier TikZ sont lus (jamais tapes) dans : etape5_2_IC_taux_succes.csv (taux de succes moyen par probleme,
intervalle BCa a 95 %, portee globale, 36 problemes) et b4_pareto_front.csv (pente log-log du temps reel par iteration, n = 100-1000).
Meme equipement que l'original : tikz avec les bibliotheques shapes.geometric, arrows.meta, calc (et positioning, chargee par le
preambule du manuscrit), xcolor ; polices \\small et \\footnotesize (>= 10 pt en classe 12 pt) ; largeur < \\textwidth.
Sortie : figures/fig_decision_rule_v05.tex.
"""
import csv
import os

from paths import OUT_DIR, FIG_DIR


def rd(f):
    with open(os.path.join(OUT_DIR, f), encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def main():
    rate = {r["method"]: r for r in rd("etape5_2_IC_taux_succes.csv") if r["scope"] == "global"}
    par = {r["method"]: r for r in rd("b4_pareto_front.csv")}
    assert all(int(r["n_problemes"]) == 36 for r in rate.values())

    def ci(m):
        r = rate[m]
        return f"{float(r['taux_succes_moyen_par_probleme']):.3f} [{float(r['IC95_bas']):.3f},\\,{float(r['IC95_haut']):.3f}]"

    def slope(m):
        r = par[m]
        return f"{float(r['pente_temps_par_iteration']):.2f} [{float(r['IC95_bas_pente']):.2f},\\,{float(r['IC95_haut_pente']):.2f}]"

    tex = r"""% fig_decision_rule_v05.tex -- genere par p10_regle_decision.py (consigne 010, priorite 3) ; \input{} directement dans la section de la regle de decision.
% Requiert dans le preambule : \usepackage{tikz} + \usetikzlibrary{shapes.geometric,arrows.meta,calc} (le manuscrit charge aussi positioning).
% Sources des nombres : etape5_2_IC_taux_succes.csv (taux de succes moyen par probleme, IC BCa a 95 %) ; b4_pareto_front.csv (pentes du temps par iteration).
\begin{figure}[t]
\centering
\begin{tikzpicture}[scale=1.0]
\definecolor{cSt}{HTML}{1565C0}   % Steihaug-CG
\definecolor{cNL}{HTML}{6A1B9A}   % Newton-LM
\definecolor{cBW}{HTML}{C62828}   % BFGS-Wolfe
\definecolor{cBA}{HTML}{E65100}   % BFGS-Armijo
\definecolor{cQbg}{HTML}{ECEFF1}  % fond losange
\definecolor{cQbd}{HTML}{546E7A}  % bord losange
\definecolor{cWbg}{HTML}{FFF8E1}  % fond de la mise en garde
\definecolor{cWbd}{HTML}{8D6E63}  % bord de la mise en garde

% -- Styles -----------------------------------------------------------------
\tikzset{
  qnode/.style={
    diamond, aspect=2.5,
    draw=cQbd, fill=cQbg, line width=1pt,
    text width=2.4cm, align=center,
    inner sep=1pt, font=\small\sffamily
  },
  leafN/.style={
    rectangle, rounded corners=5pt,
    draw=cNL!80!black, fill=cNL!12, line width=1pt,
    text width=7.4cm, align=center, minimum height=1.3cm,
    inner sep=5pt, font=\small\sffamily
  },
  leafB/.style={
    rectangle, rounded corners=5pt,
    draw=cBW!80!black, fill=cBW!12, line width=1pt,
    text width=7.4cm, align=center, minimum height=1.3cm,
    inner sep=5pt, font=\small\sffamily
  },
  warn/.style={
    rectangle, rounded corners=5pt,
    draw=cWbd, fill=cWbg, line width=1pt, dashed,
    text width=11.4cm, align=center, minimum height=1.3cm,
    inner sep=5pt, font=\footnotesize\sffamily
  },
  arr/.style={
    -{Stealth[length=5.5pt,width=4pt]},
    line width=1pt, color=gray!65
  },
  darr/.style={
    -{Stealth[length=5.5pt,width=4pt]},
    line width=1pt, color=cWbd, dashed
  },
  ylbl/.style={
    font=\footnotesize\sffamily\bfseries,
    text=green!40!black, fill=white,
    inner sep=1pt
  },
  nlbl/.style={
    font=\footnotesize\sffamily\bfseries,
    text=red!60!black, fill=white,
    inner sep=1pt
  }
}

% -- Figure -----------------------------------------------------------------
\node[qnode] (Q) at (0, 0) {Hessian $\nabla^2\!f$\\available?};

\node[leafB] (LB) at (7.6, 1.9) {%
  \textbf{\textcolor{cBW!80!black}{BFGS--Wolfe}}\\[2pt]
  \footnotesize success rate """ + ci("BFGS_Wolfe") + r"""\\[1pt]
  \footnotesize alternative: \textcolor{cBA!90!black}{BFGS--Armijo}, """ + ci("BFGS_Armijo") + r"""
};

\node[leafN] (LN) at (7.6, -1.9) {%
  \textbf{\textcolor{cNL!80!black}{Newton-LM}}\\[2pt]
  \footnotesize success rate """ + ci("Newton_BT") + r"""
};

\node[warn] (W) at (6.4, -5.2) {%
  \textbf{Outside the tested range} (\textit{n}\,$>$\,1000, or dense factorization impossible in memory or in time):\\[2pt]
  \textcolor{cSt!80!black}{\textbf{Steihaug-CG}} (success rate """ + ci("Steihaug_CG") + r"""), whose time per iteration grows more slowly
  (log--log slope """ + slope("Steihaug_CG") + r""" against """ + slope("Newton_BT") + r""" for Newton-LM, \textit{n}\,=\,100--1000).\\[2pt]
  No data supports this advice beyond \textit{n}\,=\,1000.
};

%% Fleches
\draw[arr] (Q.north) |- node[nlbl, pos=0.25, left]{No} (LB.west);
\draw[arr] (Q.south) |- node[ylbl, pos=0.25, left]{Yes} (LN.west);
\draw[darr] (LN.south) -- (LN.south |- W.north);

\end{tikzpicture}
\caption{Decision rule for unconstrained optimization method selection: one node and two leaves. Success rate: mean over the 36 problems of the
per-problem success proportion, with its 95\,\% BCa cluster-bootstrap interval (benchmark of 156 distinct instances; $n\le 1000$, dense Hessians).
BFGS--Wolfe is the leaf retained by the learned rule (margin $\delta=0.05$); BFGS--Armijo is reported as an alternative.
The dashed box is a warning outside the tested range, not a result; the success rate and the slopes quoted in it were measured for $n\le 1000$.}
\label{fig:decision-rule}
\end{figure}
"""
    os.makedirs(FIG_DIR, exist_ok=True)
    with open(os.path.join(FIG_DIR, "fig_decision_rule_v05.tex"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write(tex)
    print("ecrit fig_decision_rule_v05.tex")


if __name__ == "__main__":
    main()
