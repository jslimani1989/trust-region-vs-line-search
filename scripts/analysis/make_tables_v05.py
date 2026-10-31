# -*- coding: utf-8 -*-
"""
Consigne 009e, priorite 7.1 : tableaux LaTeX (anglais, comme le manuscrit) et leur CSV source, dans tables/.
Ce script ne calcule AUCUN nombre nouveau (sauf des sommes de comptes entiers pour le tableau des departs multiples) : il lit les
sorties des scripts d'analyse du dossier et les met en forme. Un nombre d'un tableau est recalculable en relancant le script
indique dans la ligne "% source" du .tex et dans la colonne `source` du CSV.

  T-succes      : taux de succes, proportion des 156 instances distinctes, par solveur et categorie (ordre 1)
  T-seuil       : rapport de temps Newton-LM / Steihaug-CG par dimension (cellules >= 3 problemes), colonne "l'IC exclut 1", n*
  T-compromis   : quatre regles, perte de succes, regret (mesure seul ; mesure + estime ; sous-ensemble commun), IC95
  T-pareto      : succes (moyenne par probleme), pente du temps par iteration, pente du nombre d'iterations, front
  T-L1          : BFGS-Armijo et BFGS-Wolfe : succes, IC, test de permutation (p brut de la feuille ; p de Holm, 15 paires), N_tot
  S1 / S2       : Newton-LM par probleme et synthese ; Rosenbrock n = 10
  T-BFGS-sens   : sensibilite de BFGS (et Newton-LM, Dogleg) aux arrondis, v0.4 -> v0.5
  T-departs     : departs multiples (taux par rayon, tau de Kendall, departs indefinis de Categorie C)
"""
import csv
import json
import os

from paths import OUT_DIR, TABLES_DIR, ROB_DIR

NAME = {"Cauchy_Point": "Cauchy Point", "Dogleg": "Dogleg", "Steihaug_CG": r"\SteihaugCG", "BFGS_Armijo": r"\BFGSA",
        "BFGS_Wolfe": r"\BFGSW", "Newton_BT": r"\NewtonLM"}
ABBR = {"Cauchy_Point": "CP", "Dogleg": "DL", "Steihaug_CG": "SCG", "BFGS_Armijo": "BA", "BFGS_Wolfe": "BW", "Newton_BT": "NLM"}
ORDER = ["Cauchy_Point", "Dogleg", "Steihaug_CG", "BFGS_Armijo", "BFGS_Wolfe", "Newton_BT"]


def rd(f):
    with open(os.path.join(OUT_DIR, f), encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def rj(f):
    with open(os.path.join(OUT_DIR, f), encoding="utf-8") as fh:
        return json.load(fh)


def wcsv(name, rows):
    with open(os.path.join(TABLES_DIR, name + ".csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def wtex(name, text):
    with open(os.path.join(TABLES_DIR, name + ".tex"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)


def f(x, nd):
    return f"{float(x):.{nd}f}"


def ci(lo, hi, nd):
    return f"[{f(lo, nd)}, {f(hi, nd)}]"


def mm(txt):
    """signe moins typographique pour les nombres negatifs (mode maths)."""
    import re
    return re.sub(r"(?<![\w$])-(?=\d)", "$-$", txt)


# --------------------------------------------------------------------------------------------- T-succes
def t_succes():
    src = "p1_succes_156_et_comparaison_v04.py -> p1_table_succes_156_instances.csv"
    tab = {(r["solveur"], r["portee"]): r for r in rd("p1_table_succes_156_instances.csv")}
    scopes = ["A", "B", "C", "global"]
    n_of = {sc: int(tab[("Cauchy_Point", sc)]["n_instances"]) for sc in scopes}
    best = {sc: max(int(tab[(m, sc)]["n_succes_v0.5"]) / int(tab[(m, sc)]["n_instances"]) for m in ORDER) for sc in scopes}
    rows, body = [], []
    for m in ORDER:
        cells = []
        for sc in scopes:
            r = tab[(m, sc)]
            p = int(r["n_succes_v0.5"]) / int(r["n_instances"])
            txt = f"{100 * p:.1f} ({r['n_succes_v0.5']}/{r['n_instances']})"
            cells.append(rf"\textbf{{{txt}}}" if abs(p - best[sc]) < 1e-12 else txt)
            rows.append({"solveur": m, "portee": sc, "n_instances": r["n_instances"], "n_succes": r["n_succes_v0.5"],
                         "proportion": r["proportion_v0.5"], "pourcentage": f"{100 * p:.1f}", "n_succes_v0.4": r["n_succes_v0.4"],
                         "proportion_v0.4": r["proportion_v0.4"], "source": src})
        body.append(f"    {NAME[m]} & " + " & ".join(cells) + r" \\")
    tex = (f"% T-succes -- genere par make_tables_v05.py ; source : {src}\n"
           r"""\begin{table}[htbp]
  \centering
  \begin{threeparttable}
  \caption{Success rate (\%) by category and method over the """ + str(n_of["global"]) + r""" distinct (problem, dimension) instances,
    with Levenberg--Marquardt regularization for \NewtonLM and a unified $\Kmax=1000$ budget; in parentheses, number of
    successes over number of instances. Success means first-order stationarity ($\norm{\nabla f}\le 10^{-6}$) within the
    budget. Best rate per column in bold.}
  \label{tab:success}
  \small
  \begin{tabular}{@{}lrrrr@{}}
    \toprule
    Method & Cat.~A (""" + str(n_of["A"]) + r""") & Cat.~B (""" + str(n_of["B"]) + r""") & Cat.~C (""" + str(n_of["C"]) + r""") & Global (""" + str(n_of["global"]) + r""") \\
    \midrule
""" + "\n".join(body) + r"""
    \bottomrule
  \end{tabular}
  \begin{tablenotes}\footnotesize
    \item Instances are counted once each: the benchmark has 180 (problem, dimension) slots, of which some problems are run at a
    single fixed dimension and appear in several slots; the """ + str(n_of["global"]) + r""" distinct instances are the unit of this table. Statistical
    tests and intervals (Tables~\ref{tab:pareto} and~\ref{tab:L1}) use the mean over problems (36 clusters) instead.
  \end{tablenotes}
  \end{threeparttable}
\end{table}
""")
    wcsv("T-succes", rows)
    wtex("T-succes", tex)


# --------------------------------------------------------------------------------------------- T-seuil
def t_seuil():
    b1 = rj("b1_seuil_noeud2.json")["principale"]
    cells = [r for r in rd("b1_ratio_par_dimension.csv") if r["retenue"] == "1"]
    omitted = [(r["n"], r["n_problemes"]) for r in rd("b1_ratio_par_dimension.csv") if r["retenue"] != "1"]
    key = "ratio_geo_moyen_NewtonLM_sur_SteihaugCG"
    src = "b1_seuil_noeud2.py -> b1_ratio_par_dimension.csv, b1_seuil_noeud2.json"
    rows, body = [], []
    for r in cells:
        lo, hi = float(r["IC95_bas"]), float(r["IC95_haut"])
        excl = (hi < 1) or (lo > 1)
        side = "<1" if hi < 1 else (">1" if lo > 1 else "")
        rows.append({"type": "cellule", "n": r["n"], "n_problemes": r["n_problemes"], "ratio_NLM_sur_SCG": r[key],
                     "IC95_bas": r["IC95_bas"], "IC95_haut": r["IC95_haut"], "IC_exclut_1": excl,
                     "cote": side, "source": src})
        mark = r"yes ($<1$)" if hi < 1 else (r"yes ($>1$)" if lo > 1 else "no")
        body.append(rf"    {r['n']} & {r['n_problemes']} & {f(r[key], 2)} & {ci(lo, hi, 2)} & {mark} \\")
    part = b1["bootstrap_part_replicats_avec_croisement_montant"]
    q = b1["bootstrap_n_star_quantiles_2.5_50_97.5_conditionnels"]
    rows.append({"type": "n_star", "n": "non determinable", "n_problemes": "", "ratio_NLM_sur_SCG": "", "IC95_bas": "", "IC95_haut": "",
                 "IC_exclut_1": "", "cote": "",
                 "source": f"{src} ; croisement montant dans {100 * part:.1f} % des reechantillons bootstrap "
                           f"(quantiles conditionnels 2,5/50/97,5 % : {q[0]}, {q[1]}, {q[2]})"})
    omit_txt = ", ".join(f"$n={n}$ ({k} problem{'s' if k != '1' else ''})" for n, k in omitted)
    tex = (f"% T-seuil -- genere par make_tables_v05.py ; source : {src}\n"
           r"""\begin{table}[htbp]
  \centering
  \begin{threeparttable}
  \caption{Real-time ratio \NewtonLM\,/\,\SteihaugCG per dimension on Category~A instances solved by both solvers
    (median of the three repetitions of the wall-clock time; instances with a time below 1\,ms excluded).
    Ratio: geometric mean over the problems of the cell; brackets: 95\,\% BCa cluster-bootstrap interval (problems as
    clusters, 10\,000 resamples, seed 20260927). Only cells with at least three problems are retained.
    A ratio below~1 means \NewtonLM is faster.}
  \label{tab:threshold}
  \small
  \begin{tabular}{@{}rrrll@{}}
    \toprule
    $n$ & Problems & Ratio & 95\,\% CI & CI excludes 1 \\
    \midrule
""" + "\n".join(body) + rf"""
    \midrule
    \multicolumn{{5}}{{@{{}}l}}{{Upward crossing of 1 ($n^\ast$): \textbf{{not determinable}}}} \\
    \bottomrule
  \end{{tabular}}
  \begin{{tablenotes}}\footnotesize
    \item The ratio stays below~1 at every retained dimension up to $n=1000$: no upward crossing exists between consecutive
    retained dimensions, so the threshold $n^\ast$ (log--log interpolation of the upward crossing of 1) is not defined and the
    node ``dense factorization affordable?'' is a candidate for removal. The intervals at $n=500$ and $n=1000$ contain~1.
    An upward crossing appears in {100 * part:.1f}\,\% of the cluster-bootstrap resamples; this is descriptive and does not define an interval for $n^\ast$.
    Omitted cells (fewer than three problems): {omit_txt}.
    {b1['n_exploitables']} instances of Category~A enter the table (on the reference machine only).
  \end{{tablenotes}}
  \end{{threeparttable}}
\end{{table}}
""")
    wcsv("T-seuil", rows)
    wtex("T-seuil", tex)


# --------------------------------------------------------------------------------------------- T-compromis
def t_compromis():
    syn = {r["regle"]: r for r in rd("b3_synthese.csv") if r["portee"] == "toutes"}
    meta = rj("b3_meta.json")
    src = "b3_validation_LOO.py -> b3_synthese.csv (portee=toutes)"
    order = [("naive", r"Robustness first$^{a}$"), ("un_noeud", r"Time first$^{b}$"), ("arbre", r"Learned tree$^{c}$"),
             ("article", r"Article rule$^{d}$")]
    rows, body = [], []
    for key, label in order:
        r = syn[key]
        pl, ph = json.loads(r["perte_IC95"])
        ml, mh = json.loads(r["regret_mesure_seul_IC95"])
        el, eh = json.loads(r["regret_mesure_estime_IC95"])
        cl, ch = json.loads(r["regret_mesure_seul_commun_IC95"])
        rows.append({"regle": key, "perte_succes": r["perte_moyenne"], "perte_IC95_bas": pl, "perte_IC95_haut": ph,
                     "taux_accord": r["taux_accord"],
                     "regret_mesure_seul": r["regret_mesure_seul_geo"], "regret_mesure_seul_IC95_bas": ml, "regret_mesure_seul_IC95_haut": mh,
                     "n_problemes_mesure_seul": r["n_pb_regret_mesure_seul"], "n_instances_mesure_seul": r["n_inst_regret_mesure_seul"],
                     "regret_mesure_estime": r["regret_mesure_estime_geo"], "regret_mesure_estime_IC95_bas": el,
                     "regret_mesure_estime_IC95_haut": eh, "n_problemes_mesure_estime": r["n_pb_regret_mesure_estime"],
                     "n_instances_mesure_estime": r["n_inst_regret_mesure_estime"],
                     "regret_commun_mesure_seul": r["regret_mesure_seul_commun_geo"], "regret_commun_mesure_seul_IC95_bas": cl,
                     "regret_commun_mesure_seul_IC95_haut": ch, "n_problemes_commun": r["n_pb_commun"],
                     "n_instances_commun": r["n_inst_commun"],
                     "regret_commun_mesure_estime": r["regret_mesure_estime_commun_geo"],
                     "n_instances_commun_estime": r["n_inst_commun_estime"], "source": src})
        body.append(rf"    {label} & {f(r['perte_moyenne'], 3)} {ci(pl, ph, 3)} & {f(r['regret_mesure_seul_geo'], 2)} {ci(ml, mh, 2)} & "
                    rf"{f(r['regret_mesure_estime_geo'], 2)} {ci(el, eh, 2)} & {f(r['regret_mesure_seul_commun_geo'], 2)} {ci(cl, ch, 2)} \\")
    first = syn["naive"]
    commun = {r["regle"]: r for r in rows}
    ce = ", ".join(f(commun[k]["regret_commun_mesure_estime"], 2) for k in ("naive", "un_noeud", "arbre", "article"))
    tex = (f"% T-compromis -- genere par make_tables_v05.py ; source : {src}\n"
           r"""\begin{table}[htbp]
  \centering
  \begin{threeparttable}
  \caption{Trade-off between robustness and time for four selection rules (Hessian branch: Cauchy Point, Dogleg,
    \SteihaugCG, \NewtonLM), leave-one-problem-out validation over 36 folds; the threshold, the learned tree and the time model are
    relearned in every fold. Success loss: mean over problems of (best success observed among the four candidates) $-$ (success of
    the rule). Time regret: geometric mean over problems of the real time of the rule divided by that of the fastest candidate that
    succeeded (1 is best); ``measured only'' uses measured times of at least 1\,ms, ``measured $+$ estimated'' adds estimated times
    where no measurement exists, and the common subset restricts the measured-only regret to the instances where it is defined
    for all four rules. Brackets: 95\,\% percentile bootstrap intervals over problems (10\,000 resamples, seed 20260927), indicative.}
  \label{tab:tradeoff}
  \footnotesize
  \setlength{\tabcolsep}{4pt}
  \begin{tabular}{@{}lllll@{}}
    \toprule
    Rule & Success loss & \multicolumn{2}{c}{Time regret} & Common subset \\
    \cmidrule(lr){3-4}
     &  & measured only & measured $+$ estimated & (measured only) \\
    \midrule
""" + "\n".join(body) + rf"""
    \bottomrule
  \end{{tabular}}
  \begin{{tablenotes}}\footnotesize
    \item[a] \NewtonLM whenever the Hessian is available.
    \item[b] \NewtonLM if $n<n^\ast$, \SteihaugCG otherwise, with $n^\ast$ relearned in each fold. Here $n^\ast$ is not determinable in any of the
    36 folds (no upward crossing), so the rule always selects \NewtonLM and coincides with the robustness-first rule.
    \item[c] Rule learned with margin $\delta=0.05$ in every fold (all instances fall in the ``affordable'' branch).
    \item[d] \SteihaugCG if $n\ge 500$, and on C01$'$, C08$'$, C09$'$, C10$'$; \NewtonLM otherwise.
    \item Regret is defined on {first['n_pb_regret_mesure_seul']} of the 36 problems (measured only; {min(int(r['n_instances_mesure_seul']) for r in rows)}--{max(int(r['n_instances_mesure_seul']) for r in rows)} instances)
    and on {first['n_pb_regret_mesure_estime']} problems (measured $+$ estimated; {min(int(r['n_instances_mesure_estime']) for r in rows)}--{max(int(r['n_instances_mesure_estime']) for r in rows)} instances);
    the common subset has {first['n_inst_commun']} instances ({first['n_pb_commun']} problems; with estimated times added: {meta['n_instances_sous_ensemble_commun_mesure_estime']} instances,
    regret {ce} for the four rules in the order of the table). Times below 1\,ms are excluded; an estimated time is used only where no measurement
    exists (Cauchy Point and Dogleg at $n\ge 500$ in Categories~B and~C).
  \end{{tablenotes}}
  \end{{threeparttable}}
\end{{table}}
""")
    wcsv("T-compromis", rows)
    wtex("T-compromis", tex)


# --------------------------------------------------------------------------------------------- T-pareto
def t_pareto():
    rows_in = rd("b4_pareto_front.csv")
    by = {r["method"]: r for r in rows_in}
    src = "b4_pareto_temps.py -> b4_pareto_front.csv ; etape5_statistiques.py"
    rows, body = [], []
    for m in ORDER:
        r = by[m]
        dom = r["domine_par"]
        dom_tex = "---"
        if dom:
            parts = []
            for tok in dom.split(";"):
                name, status = tok.split("(")
                parts.append(ABBR[name] + (r"$^{\ast}$" if status.startswith("certaine") else ""))
            dom_tex = ", ".join(parts)
        it = "n/e" if r["pente_nombre_iterations_descriptif"] == "" else \
            f"{f(r['pente_nombre_iterations_descriptif'], 2)} {ci(r['IC95_bas_pente_iter'], r['IC95_haut_pente_iter'], 2)} ({r['n_problemes_pente_iter']})"
        if r["pente_nombre_iterations_descriptif"] == "":
            it = f"n/e ({r['n_problemes_pente_iter']})"
        rows.append({"method": m, "taux_succes": r["taux_succes"], "IC95_bas_succes": r["IC95_bas_succes"],
                     "IC95_haut_succes": r["IC95_haut_succes"], "pente_temps_par_iteration": r["pente_temps_par_iteration"],
                     "IC95_bas_pente": r["IC95_bas_pente"], "IC95_haut_pente": r["IC95_haut_pente"],
                     "n_problemes_pente": r["n_problemes_pente"],
                     "pente_nombre_iterations_descriptif": r["pente_nombre_iterations_descriptif"] or "non estimable",
                     "IC95_bas_pente_iter": r["IC95_bas_pente_iter"], "IC95_haut_pente_iter": r["IC95_haut_pente_iter"],
                     "n_problemes_pente_iter": r["n_problemes_pente_iter"], "sur_le_front": r["sur_le_front"],
                     "domine_par": dom, "source": src})
        front = r"\textbf{yes}" if r["sur_le_front"] == "True" else "no"
        body.append(rf"    {NAME[m]} & {f(r['taux_succes'], 3)} {ci(r['IC95_bas_succes'], r['IC95_haut_succes'], 3)} & "
                    rf"{f(r['pente_temps_par_iteration'], 2)} {ci(r['IC95_bas_pente'], r['IC95_haut_pente'], 2)} & {it} & {front} & {dom_tex} \\")
    k = {r["n_problemes_pente"] for r in rows_in}
    assert len(k) == 1
    tex = (f"% T-pareto -- genere par make_tables_v05.py ; source : {src}\n"
           r"""\begin{table}[htbp]
  \centering
  \begin{threeparttable}
  \caption{Robustness--scalability Pareto analysis. Robustness: success rate averaged over the 36 problems
    (BCa cluster-bootstrap interval). Scalability: log--log slope of the real time \emph{per iteration} against $n$
    ($n=100,500,1000$), all runs, median over the """ + next(iter(k)) + r""" Category~A problems available at the three dimensions
    (percentile cluster-bootstrap interval). The slope of the iteration count (successful runs at the three dimensions;
    number of problems in parentheses; n/e: not estimable, fewer than five problems) is descriptive and is not used for the front.}
  \label{tab:pareto}
  \footnotesize
  \setlength{\tabcolsep}{3pt}
  \begin{tabular}{@{}lllllc@{}}
    \toprule
    Solver & Success rate & \shortstack[l]{Slope of time\\per iteration} & \shortstack[l]{Slope of iteration\\count (descriptive)} & On front & Dominated by \\
    \midrule
""" + "\n".join(body) + r"""
    \bottomrule
  \end{tabular}
  \begin{tablenotes}\footnotesize
    \item A solver is on the front if no other solver is at least as robust and has at most the same slope with one strict inequality.
    Dominators: CP Cauchy Point, DL Dogleg, SCG \SteihaugCG, BA \BFGSA, NLM \NewtonLM;
    $^{\ast}$ certain (Holm-adjusted permutation test on robustness $<0.05$ and disjoint slope intervals), otherwise provisional.
  \end{tablenotes}
  \end{threeparttable}
\end{table}
""")
    wcsv("T-pareto", rows)
    wtex("T-pareto", tex)


# --------------------------------------------------------------------------------------------- T-L1
def t_l1():
    rates = {r["method"]: r for r in rd("etape5_2_IC_taux_succes.csv") if r["scope"] == "global"}
    diff = next(r for r in rd("etape5_2_IC_ecarts_paires.csv")
                if r["scope"] == "global" and r["solveur_a"] == "BFGS_Armijo" and r["solveur_b"] == "BFGS_Wolfe")
    glob = next(r for r in rd("etape5_1_tests_permutation.csv")
                if r["famille"] == "global" and r["solveur_a"] == "BFGS_Armijo" and r["solveur_b"] == "BFGS_Wolfe")
    leaf = next(r for r in rd("b2_feuilles.csv") if r["leaf"] == "L1_pas_hessienne")
    leaf_test = json.loads(leaf["tests_permutation_info"])[0]
    pair_t = next(r for r in rd("b2_paires_departage.csv")
                  if r["variante"] == "delta=0.05 (principale)" and r["leaf"] == "L1_pas_hessienne")
    cost = next(r for r in rd("tl1_ntot_15paires_recalcule.csv")
                if r["solveur_a"] == "BFGS_Armijo" and r["solveur_b"] == "BFGS_Wolfe")
    assert pair_t["a"] == "BFGS_Armijo" and pair_t["b"] == "BFGS_Wolfe" and pair_t["source"] == "temps"
    ra, rw = rates["BFGS_Armijo"], rates["BFGS_Wolfe"]
    sw = float(rw["taux_succes_moyen_par_probleme"]) - float(ra["taux_succes_moyen_par_probleme"])
    rows = [
        {"metrique": "success_rate_per_problem", "BFGS_Armijo": ra["taux_succes_moyen_par_probleme"], "BFGS_Armijo_IC95_bas": ra["IC95_bas"],
         "BFGS_Armijo_IC95_haut": ra["IC95_haut"], "BFGS_Wolfe": rw["taux_succes_moyen_par_probleme"], "BFGS_Wolfe_IC95_bas": rw["IC95_bas"],
         "BFGS_Wolfe_IC95_haut": rw["IC95_haut"], "paire_valeur": "", "paire_IC95_bas": "", "paire_IC95_haut": "", "n_problemes": 36,
         "source": "etape5_statistiques.py -> etape5_2_IC_taux_succes.csv (BCa, graine 20260927)"},
        {"metrique": "difference_Armijo_moins_Wolfe", "paire_valeur": diff["mean_d"], "paire_IC95_bas": diff["IC95_bas"],
         "paire_IC95_haut": diff["IC95_haut"], "n_problemes": 36, "source": "etape5_statistiques.py -> etape5_2_IC_ecarts_paires.csv"},
        {"metrique": "p_brut_test_de_permutation_de_la_feuille_L1", "paire_valeur": f"{leaf_test['p_brut']:.4f}", "n_problemes": leaf_test["n_problemes"],
         "source": "b2_arbre_decision_temps.py -> b2_feuilles.csv (feuille L1, une comparaison, Monte-Carlo 100000, graine 20260927)"},
        {"metrique": "p_de_Holm_famille_globale_15_paires", "paire_valeur": f"{float(glob['p_holm']):.4f}", "n_problemes": glob["n_problemes"],
         "source": f"etape5_statistiques.py -> etape5_1_tests_permutation.csv (famille globale ; p brut de cette paire {float(glob['p_brut']):.4f})"},
        {"metrique": "Ntot_ratio_Armijo_sur_Wolfe", "paire_valeur": cost["ratio_geo_moyen_v05"], "paire_IC95_bas": cost["IC95_bas_v05"],
         "paire_IC95_haut": cost["IC95_haut_v05"], "n_problemes": cost["n_problemes_utilises"],
         "source": f"tl1_ntot_15paires.py -> tl1_ntot_15paires_recalcule.csv ({cost['n_instances_utilisees']} instances ; v0.4 : {cost['ratio_geo_moyen_v04']} "
                   f"[{cost['IC95_bas_v04']}, {cost['IC95_haut_v04']}])"},
        {"metrique": "temps_reel_ratio_Armijo_sur_Wolfe_feuille_L1", "paire_valeur": pair_t["ratio_a_sur_b"], "n_problemes": pair_t["n_problemes"],
         "source": f"b2_arbre_decision_temps.py -> b2_paires_departage.csv ({pair_t['n_instances']} instances ; part de valeurs estimees {pair_t['part_valeurs_estimees']})"},
    ]
    for r in rows:
        for k in ("BFGS_Armijo", "BFGS_Armijo_IC95_bas", "BFGS_Armijo_IC95_haut", "BFGS_Wolfe", "BFGS_Wolfe_IC95_bas", "BFGS_Wolfe_IC95_haut",
                  "paire_valeur", "paire_IC95_bas", "paire_IC95_haut"):
            r.setdefault(k, "")
    wcsv("T-L1", rows)
    tex = (r"""% T-L1 -- genere par make_tables_v05.py ; source : etape5_statistiques.py, b2_arbre_decision_temps.py, tl1_ntot_15paires.py
\begin{table}[htbp]
  \centering
  \begin{threeparttable}
  \caption{Leaf~L1 (Hessian not available): \BFGSA versus \BFGSW over the 36 problems (156 distinct instances).
    Both solvers lie within the margin $\delta=0.05$ of the best success rate; the difference in robustness is not significant, and
    the tie-break on measured time and on oracle calls favors \BFGSW.}
  \label{tab:L1}
  \small
  \setlength{\tabcolsep}{4.5pt}
  \begin{tabular}{@{}lll@{}}
    \toprule
     & \BFGSA & \BFGSW \\
    \midrule
""" + rf"""    Success rate per problem [95\,\% CI]\tnote{{a}} & {f(ra['taux_succes_moyen_par_probleme'], 3)} {ci(ra['IC95_bas'], ra['IC95_haut'], 3)} & {f(rw['taux_succes_moyen_par_probleme'], 3)} {ci(rw['IC95_bas'], rw['IC95_haut'], 3)} \\
    \midrule
    \multicolumn{{3}}{{@{{}}l}}{{\textit{{Paired comparison (\BFGSA versus \BFGSW)}}}} \\
    Success difference [95\,\% CI]\tnote{{a}} & \multicolumn{{2}}{{l}}{{{mm(f(diff['mean_d'], 3))} {mm(ci(diff['IC95_bas'], diff['IC95_haut'], 3))}}} \\
    Permutation test of leaf L1, raw $p$\tnote{{b}} & \multicolumn{{2}}{{l}}{{{leaf_test['p_brut']:.3f}}} \\
    Permutation test, 15 pairs, Holm-adjusted $p$\tnote{{b}} & \multicolumn{{2}}{{l}}{{{float(glob['p_holm']):.3f}}} \\
    $\Ntot$ ratio \BFGSA\,/\,\BFGSW [95\,\% CI]\tnote{{c}} & \multicolumn{{2}}{{l}}{{{f(cost['ratio_geo_moyen_v05'], 2)} {ci(cost['IC95_bas_v05'], cost['IC95_haut_v05'], 2)}}} \\
    Real-time ratio \BFGSA\,/\,\BFGSW in leaf L1\tnote{{d}} & \multicolumn{{2}}{{l}}{{{f(pair_t['ratio_a_sur_b'], 2)}}} \\
    \bottomrule
  \end{{tabular}}
  \begin{{tablenotes}}\footnotesize
    \item[a] Mean over problems of the per-problem success proportion; BCa cluster bootstrap, 10\,000 resamples, seed 20260927.
    \item[b] Sign-flip permutation test on the per-problem differences (Monte-Carlo, 100\,000 draws, seed 20260927). Leaf test: one comparison,
    raw $p$. Global family: raw $p={float(glob['p_brut']):.3f}$ for this pair, adjusted by Holm over the 15 solver pairs.
    \item[c] Instances solved by both solvers ({cost['n_problemes_utilises']} problems, {cost['n_instances_utilisees']} instances), $\Ntot$ counted with the evaluation memory;
    geometric mean per problem then over problems; BCa cluster bootstrap. A ratio above~1 means \BFGSA uses more oracle calls.
    \item[d] Geometric mean over {pair_t['n_problemes']} problems ({pair_t['n_instances']} instances) of the ratio of real times; {100 * float(pair_t['part_valeurs_estimees']):.0f}\,\% of the time values are
    estimated (no measurement at $n\ge 500$ in Categories~B and~C). A ratio above~1 means \BFGSA is slower.
  \end{{tablenotes}}
  \end{{threeparttable}}
\end{{table}}
""")
    wtex("T-L1", tex)
    return sw


# --------------------------------------------------------------------------------------------- S1, S2
def ms(t):
    t = float(t) * 1000.0
    if t < 10:
        return f"{t:.2f}"
    if t < 100:
        return f"{t:.1f}"
    return f"{t:,.0f}".replace(",", r"\,")


def s1():
    cells = rd("b5_table_S1_avec_temps_reel.csv")
    synth = rd("b5_table_S1_synthese.csv")
    wcsv("S1", [{**c, "source": "b5_table_S1_temps.py -> b5_table_S1_avec_temps_reel.csv"} for c in cells])
    wcsv("S1_synthese", [{**s, "source": "b5_table_S1_temps.py -> b5_table_S1_synthese.csv"} for s in synth])
    problems = []
    for c in cells:
        if c["problem"] not in problems:
            problems.append(c["problem"])
    by = {(c["problem"], int(c["rang"])): c for c in cells}
    head_n = {1: "10", 2: "50", 3: "100", 4: "500", 5: "1000"}
    body = []
    for p in problems:
        parts = [p]
        for rk in range(1, 6):
            c = by[(p, rk)]
            it = c["n_iter"] if c["status"] == "success" else r"fail$^{\ddagger}$"
            lp = r"$^{\dagger}$" if c["faible_precision"] != "non" else ""
            parts.append(f"{it} & {ms(c['temps_reel_median_NewtonLM_s'])}{lp}")
        body.append("    " + " & ".join(parts) + r" \\")
    cols = "l" + "rr" * 5
    hdr1 = " & ".join(rf"\multicolumn{{2}}{{c}}{{$n={head_n[rk]}$}}" for rk in range(1, 6))
    sub = " & ".join("Iter. & ms" for _ in range(5))
    tex1 = (r"""% S1 (par probleme) -- genere par make_tables_v05.py ; source : b5_table_S1_temps.py -> b5_table_S1_avec_temps_reel.csv
\begin{table}[H]
  \centering
  \begin{threeparttable}
  \caption{\NewtonLM per problem, Category~A, five dimensions: number of iterations and median real time in milliseconds
    (median of 3 repetitions of the wall-clock time, single process pinned to one core). Referenced as Table~S1 in the main text.}
  \label{tab:S_newton_pb}
  \footnotesize
  \setlength{\tabcolsep}{3pt}
  \begin{tabular}{@{}l""" + cols + r"""@{}}
    \toprule
    Problem & """ + hdr1 + r""" \\
    \cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}\cmidrule(l){10-11}
     & """ + sub + r""" \\
    \midrule
""" + "\n".join(body) + r"""
    \bottomrule
  \end{tabular}
  \begin{tablenotes}\footnotesize
    \item $^{\dagger}$ ``low precision'': median real time below 1\,ms. $^{\ddagger}$ failure: the budget $\Kmax=1000$ is exhausted
    (the time is that of the 1000 iterations). Problem A09 is run at $n=8$ and $n=48$ in place of $n=10$ and $n=50$.
  \end{tablenotes}
  \end{threeparttable}
\end{table}
""")
    wtex("S1_par_probleme", tex1)
    body2 = []
    for s in synth:
        lp = s["cellules_faible_precision"]
        body2.append(rf"    {head_n[int(s['rang'])]} & {s['n_succes']}/{s['n_total']} & {float(s['mediane_n_iter']):g} & {float(s['mediane_N_tot_v05']):g} & "
                     rf"{ms(s['mediane_temps_reel_s'])} & {lp} \\")
    tex2 = (r"""% S1 (synthese par dimension) -- genere par make_tables_v05.py ; source : b5_table_S1_temps.py -> b5_table_S1_synthese.csv
\begin{table}[H]
  \centering
  \begin{threeparttable}
  \caption{\NewtonLM cost vs.\ dimension, Category~A (medians over the 11 problems; iterations and $\Ntot$ over converged instances;
    real time over all 11 instances). $\Ntot$ is counted with the evaluation memory (tag \texttt{v0.5}).}
  \label{tab:S_newton}
  \small
  \begin{tabular}{@{}rrrrrr@{}}
    \toprule
    $n$ & Converged & Iterations & $\Ntot$ & Real time (ms) & Low-precision cells\tnote{a} \\
    \midrule
""" + "\n".join(body2) + r"""
    \bottomrule
  \end{tabular}
  \begin{tablenotes}\footnotesize
    \item[a] Cells (out of 11) whose median real time is below 1\,ms.
  \end{tablenotes}
  \end{threeparttable}
\end{table}
""")
    wtex("S1_synthese", tex2)


def s2():
    rows = {r["method"]: r for r in rd("s2_rosenbrock_n10_v0.5.csv")}
    out, body = [], []
    for m in ("Steihaug_CG", "BFGS_Wolfe", "Newton_BT"):
        r = rows[m]
        g = float(r["grad_norm_final_v0.5"])
        mant, exp = f"{g:.2e}".split("e")
        out.append({"method": m, "n_iter": r["n_iter_v0.5"], "grad_norm_final": r["grad_norm_final_v0.5"], "n_f": r["n_f_evals_v0.5"],
                    "n_g": r["n_g_evals_v0.5"], "n_h": r["n_h_evals_v0.5"], "N_tot": r["N_tot_v0.5"], "status": r["status_v0.5"],
                    "n_iter_v0.4": r["n_iter_v0.4"], "N_tot_v0.4": r["N_tot_v0.4"],
                    "source": "s2_rosenbrock_n10.py -> s2_rosenbrock_n10_v0.5.csv"})
        body.append(rf"    {NAME[m]} & {r['n_iter_v0.5']} & ${mant}\times10^{{{int(exp)}}}$ & {r['n_f_evals_v0.5']} & {r['n_g_evals_v0.5']} & {r['N_tot_v0.5']} \\")
    wcsv("S2", out)
    tex = (r"""% S2 -- genere par make_tables_v05.py ; source : s2_rosenbrock_n10.py -> s2_rosenbrock_n10_v0.5.csv
\begin{table}[htbp]
  \centering
  \caption{Rosenbrock function~\cite{Rosenbrock60}, $n=10$, $x_0=(-1.2,1,-1.2,1,\ldots)^\top$, $\Kmax=1000$,
    $\varepsilon=10^{-6}$, with Levenberg--Marquardt regularization; counts of distinct oracle points (evaluation memory).
    Reproducible via the repository (tag~\texttt{v0.5}). Referenced as Table~S2 in the main text, Section~5.}
  \label{tab:S_rosen}
  \small
  \begin{tabular}{@{}lrrrrr@{}}
    \toprule
    Method & Iterations & $\norm{\nabla f}_{\mathrm{final}}$ & $\#f$ & $\#\nabla f$ & $\Ntot$ \\
    \midrule
""" + "\n".join(body) + r"""
    \bottomrule
  \end{tabular}
\end{table}
""")
    wtex("S2", tex)


# --------------------------------------------------------------------------------------------- T-BFGS-sens
def t_bfgs_sens():
    rows_in = rd("p6_sensibilite_BFGS.csv")
    by = {(r["ensemble"], r["solveur"]): r for r in rows_in}
    src = "p6_sensibilite_BFGS.py -> p6_sensibilite_BFGS.csv"
    solvers = ["BFGS-Armijo", "BFGS-Wolfe", "Newton-LM", "Dogleg"]
    tex_name = {"BFGS-Armijo": r"\BFGSA", "BFGS-Wolfe": r"\BFGSW", "Newton-LM": r"\NewtonLM", "Dogleg": "Dogleg"}
    rows, body = [], []
    for s in solvers:
        parts = [tex_name[s]]
        rec = {"solveur": s, "source": src}
        for ens, tag in (("benchmark", "b"), ("departs multiples", "m")):
            r = by[(ens, s)]
            n = r["unites"]
            parts += [f"{r['statut_change']} ({r['succes_vers_echec']}/{r['echec_vers_succes']})", r["n_iter_change"],
                      f"{100 * float(r['taux_v0.4']):.1f}$\\to${100 * float(r['taux_v0.5']):.1f}"]
            rec.update({f"{tag}_unites": n, f"{tag}_statut_change": r["statut_change"], f"{tag}_succes_vers_echec": r["succes_vers_echec"],
                        f"{tag}_echec_vers_succes": r["echec_vers_succes"], f"{tag}_n_iter_change": r["n_iter_change"],
                        f"{tag}_taux_v0.4": r["taux_v0.4"], f"{tag}_taux_v0.5": r["taux_v0.5"]})
        rows.append(rec)
        body.append("    " + " & ".join(str(x) for x in parts) + r" \\")
    tex = (f"% T-BFGS-sens -- genere par make_tables_v05.py ; source : {src}\n"
           r"""\begin{table}[htbp]
  \centering
  \begin{threeparttable}
  \caption{Sensitivity to rounding of the mathematically equivalent algebra introduced in tag \texttt{v0.5} (BFGS inverse-Hessian update in
    $O(n^2)$ for \BFGSA and \BFGSW; triangular substitutions in place of two dense solves for \NewtonLM and Dogleg), compared with tag
    \texttt{v0.4}. Status: number of runs whose success status changed (in parentheses: success $\to$ failure / failure $\to$ success);
    iter.: number of runs whose iteration count changed (status included); rate: success rate (\%) in \texttt{v0.4}\,$\to$\,\texttt{v0.5}.}
  \label{tab:bfgs-rounding}
  \small
  \setlength{\tabcolsep}{4pt}
  \begin{tabular}{@{}lcrrcrr@{}}
    \toprule
     & \multicolumn{3}{c}{Benchmark (156 instances)} & \multicolumn{3}{c}{Multi-start (462 starts)} \\
    \cmidrule(lr){2-4}\cmidrule(l){5-7}
    Solver & Status & Iter. & Rate & Status & Iter. & Rate \\
    \midrule
""" + "\n".join(body) + r"""
    \bottomrule
  \end{tabular}
  \begin{tablenotes}\footnotesize
    \item Cauchy Point and \SteihaugCG (code unchanged) show no difference. The changes are consistent with a sensitivity of BFGS to rounding near the
    optimum; only one case was diagnosed individually (A11 at $n=2$, \BFGSA: 12 iterations in \texttt{v0.4}, budget exhausted in \texttt{v0.5}, the Armijo test being decided by
    rounding noise of order $10^{-14}$ on $f(x+\alpha d)-f(x)$); the other changes were not diagnosed one by one.
  \end{tablenotes}
  \end{threeparttable}
\end{table}
""")
    wcsv("T-BFGS-sensibilite", rows)
    wtex("T-BFGS-sensibilite", tex)


# --------------------------------------------------------------------------------------------- T-departs
def t_departs():
    prop = rd("p5_departs_multiples_proportions.csv")
    kend = rd("p5_departs_multiples_kendall.csv")
    indef = rd("p5_departs_multiples_categorieC_indefinie.csv")
    src = "p5_multistart_G3.py -> p5_departs_multiples_*.csv"
    radii = ["nominal", "sigma=0.1", "sigma=1.0"]
    rows, body = [], []
    for m in ORDER:
        rec = {"solveur": m, "source": src}
        cells = []
        tot_s = tot_n = 0
        for rad in radii:
            s = sum(int(r["n_succes_v0.5"]) for r in prop if r["method"] == m and r["radius"] == rad)
            n = sum(int(r["n"]) for r in prop if r["method"] == m and r["radius"] == rad)
            tot_s, tot_n = tot_s + s, tot_n + n
            rec[f"succes_{rad}"], rec[f"n_{rad}"], rec[f"taux_{rad}"] = s, n, round(s / n, 4)
            cells.append(f"{100 * s / n:.1f}")
        rec["succes_tous"], rec["n_tous"], rec["taux_tous"] = tot_s, tot_n, round(tot_s / tot_n, 4)
        ind = next(r for r in indef if r["method"] == m and r["sous_ensemble"] == "departs_indefinis")
        rec["taux_C_departs_indefinis"] = ind["taux_v0.5"]
        s4 = sum(int(r["n_succes_v0.4"]) for r in prop if r["method"] == m)
        rec["taux_tous_v0.4"] = round(s4 / tot_n, 4)
        rows.append(rec)
        body.append(rf"    {NAME[m]} & " + " & ".join(cells) + rf" & {100 * tot_s / tot_n:.1f} & {100 * float(ind['taux_v0.5']):.1f} \\")
    krow = {(r["scope"], r["radius"]): r for r in kend}
    kbody = []
    for scope in ("global", "Cat.A", "Cat.B", "Cat.C"):
        a, b = krow[(scope, "sigma=0.1")], krow[(scope, "sigma=1.0")]
        kbody.append(f"    {scope.replace('Cat.', 'Cat.~')} & {float(a['kendall_tau_v0.5']):.2f} & {float(b['kendall_tau_v0.5']):.2f} \\\\")
        rows.append({"solveur": f"kendall_{scope}", "taux_sigma=0.1": a["kendall_tau_v0.5"], "taux_sigma=1.0": b["kendall_tau_v0.5"],
                     "tau_v0.4_sigma=0.1": a["kendall_tau_v0.4"], "tau_v0.4_sigma=1.0": b["kendall_tau_v0.4"], "source": src})
    ni = next(r for r in indef if r["method"] == "Cauchy_Point" and r["sous_ensemble"] == "tous_departs")["n"]
    nind = next(r for r in indef if r["method"] == "Cauchy_Point" and r["sous_ensemble"] == "departs_indefinis")["n"]
    tex = (f"% T-departs -- genere par make_tables_v05.py ; source : {src}\n"
           r"""\begin{table}[htbp]
  \centering
  \begin{threeparttable}
  \caption{Multi-start robustness (4 problems per category, $n=10$ and $n=100$ where the dimension is free; 462 starts per solver): success
    rate (\%) at the nominal start (22 starts) and at perturbed starts $x_0+\sigma\max(1,\norm{x_0}_\infty)\,\xi$, $\xi\sim\mathcal{N}(0,I_n)$,
    $\sigma\in\{0.1,1.0\}$ (10 draws per problem and dimension; 220 starts each), pooled over the three categories; all starts; and, on Category~C,
    the success rate over the starts whose Hessian is indefinite. Panel (b): Kendall's $\tau$ between the ranking of the six solvers by success
    rate at the nominal start and under each perturbation (no $p$-value reported).}
  \label{tab:multistart}
  \small
  \begin{tabular}{@{}lrrrrr@{}}
    \toprule
    \multicolumn{6}{@{}l}{\textit{(a) Success rate (\%)}} \\
    Solver & nominal & $\sigma=0.1$ & $\sigma=1.0$ & all starts & Cat.~C, indefinite starts \\
    \midrule
""" + "\n".join(body) + r"""
    \midrule
    \multicolumn{6}{@{}l}{\textit{(b) Kendall's $\tau$ (nominal vs perturbed ranking)}} \\
    Scope & $\sigma=0.1$ & $\sigma=1.0$ & & & \\
    \midrule
""" + "\n".join(kbody) + rf"""
    \bottomrule
  \end{{tabular}}
  \begin{{tablenotes}}\footnotesize
    \item Category~C: {nind} of the {ni} starts have an indefinite Hessian at $x_0$ ({100 * int(nind) / int(ni):.1f}\,\%; unchanged from \texttt{{v0.4}}).
    Rankings use the unweighted mean over categories of the category-level rates. Cauchy Point and \SteihaugCG: rows from \texttt{{v0.4}} (code unchanged);
    the other four solvers: \texttt{{v0.5}}.
  \end{{tablenotes}}
  \end{{threeparttable}}
\end{{table}}
""")
    # le CSV a des lignes heterogenes (taux et kendall) : fichier separe pour Kendall
    wcsv("T-departs", [r for r in rows if not r["solveur"].startswith("kendall")])
    wcsv("T-departs_kendall", [{"scope": r["solveur"].replace("kendall_", ""), "tau_sigma_0.1": r["taux_sigma=0.1"], "tau_sigma_1.0": r["taux_sigma=1.0"],
                                "tau_v0.4_sigma_0.1": r["tau_v0.4_sigma=0.1"], "tau_v0.4_sigma_1.0": r["tau_v0.4_sigma=1.0"], "source": src}
                               for r in rows if r["solveur"].startswith("kendall")])
    wtex("T-departs", tex)


def main():
    os.makedirs(TABLES_DIR, exist_ok=True)
    t_succes()
    t_seuil()
    t_compromis()
    t_pareto()
    t_l1()
    s1()
    s2()
    t_bfgs_sens()
    t_departs()
    print("tables ecrites dans", TABLES_DIR)


if __name__ == "__main__":
    main()
