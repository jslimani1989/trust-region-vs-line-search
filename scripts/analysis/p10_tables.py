# -*- coding: utf-8 -*-
"""
Consigne 010, priorite 4 : tables restantes (LaTeX anglais + CSV source, dans tables/).

  T-motifs-arret : motifs d'arret par solveur et par categorie, sur les 156 instances distinctes de resultats_benchmark_v0.5.csv
                   (convergence, K_max atteint, echec de la recherche lineaire, echec du rayon de confiance, exception, valeurs
                   non finies rejetees) ; reprise de analyse_v0.3/scripts/etape4_motifs_arret.py (R3.11), donnees v0.5.
  S-tests        : 15 paires, ecart moyen de succes par probleme, IC BCa, p brut, p de Holm (familles globale, A, B, C),
                   depuis etape5_1_tests_permutation.csv et etape5_2_IC_ecarts_paires.csv (009e) ; aucun nombre recalcule.
  S-couts        : rapports de cout N_tot des 15 paires (moyenne geometrique par probleme puis sur les problemes), IC BCa, SANS p-valeur,
                   depuis tl1_ntot_15paires_recalcule.csv (procedure de 009 ; meme source que T-L1) ; CSV par categorie depuis
                   etape5_3_couts.csv.
Aucun nombre n'est tape a la main ; les formats d'affichage sont ceux de make_tables_v05.py (f, ci, mm).
"""
import csv
import os
import re

from common_data import load_distinct_rows, METHODS
from make_tables_v05 import rd, wcsv, wtex, f, ci, mm, NAME
from paths import OUT_DIR, TABLES_DIR

ORDER = ["Cauchy_Point", "Dogleg", "Steihaug_CG", "BFGS_Armijo", "BFGS_Wolfe", "Newton_BT"]
CATS = ["A", "B", "C"]


def sgn(x, nd=3):
    """valeur signee avec signe moins typographique ; evite « -0.000 »."""
    v = float(x)
    if abs(v) < 0.5 * 10 ** (-nd):
        v = 0.0
    return mm(f"{v:.{nd}f}")


def sgn_ci(lo, hi, nd=3):
    return f"[{sgn(lo, nd)}, {sgn(hi, nd)}]"


def pfmt(p):
    p = float(p)
    if p >= 0.001:
        return f"{p:.3f}"
    mant, exp = f"{p:.1e}".split("e")
    return rf"${mant}\times10^{{{int(exp)}}}$"


# --------------------------------------------------------------------------------------------- motifs d'arret
def t_motifs():
    rows = load_distinct_rows()
    assert len(rows) == 936
    src = "p10_tables.py (motifs d'arret, resultats_benchmark_v0.5.csv, 156 instances distinctes)"
    out, body = [], []
    first_of_block = True
    for m in ORDER:
        for sc in CATS + ["Global"]:
            sub = [r for r in rows if r["method"] == m and (sc == "Global" or r["category"] == sc)]
            n = len(sub)
            conv = sum(r["stop_reason"] == "converged" for r in sub)
            kmax = sum(r["stop_reason"] == "max_iter" for r in sub)
            ls = sum(r["stop_reason"] == "line_search_failure" for r in sub)
            rad = sum(r["stop_reason"] == "radius_failure" for r in sub)
            exc = sum(r["status"] == "exception" or (r["error_msg"] or "").strip() != "" for r in sub)
            nonf = sum(r["n_nonfinite_rejected"] not in ("", "0") for r in sub)
            stag = sum(r["stagnation_num"] in ("True", "true", "1") for r in sub)
            succ = sum(r["status"] == "success" for r in sub)
            assert conv + kmax + ls + rad + exc == n, (m, sc)  # motifs mutuellement exclusifs
            assert conv == succ, (m, sc)  # convergence == succes (ordre 1)
            out.append({"solveur": m, "portee": sc, "n_instances": n, "convergence": conv, "Kmax_atteint": kmax,
                        "echec_recherche_lineaire": ls, "echec_rayon_confiance": rad, "exception": exc, "valeurs_non_finies_rejetees": nonf,
                        "stagnation_numerique_indicateur": stag, "source": src})
            body.append(f"    {NAME[m] if sc == 'A' else ''} & {'All' if sc == 'Global' else sc} & {n} & {conv} & {kmax} & {ls} & {rad} & {exc} & {nonf} \\\\")
        body.append(r"    \addlinespace[2pt]")
    fail_keys = {(r["problem"], r["n"], r["method"]) for r in rows if r["stop_reason"] in ("line_search_failure", "radius_failure")}
    stag_keys = {(r["problem"], r["n"], r["method"]) for r in rows if r["stagnation_num"] in ("True", "true", "1")}
    assert stag_keys <= fail_keys  # l'indicateur de stagnation est un sous-ensemble des echecs (controle)
    stag_total = sum(r["stagnation_numerique_indicateur"] for r in out if r["portee"] == "Global")
    assert stag_total == len(stag_keys)
    fail_total = sum(r["echec_recherche_lineaire"] + r["echec_rayon_confiance"] for r in out if r["portee"] == "Global")
    nonf_total = sum(r["valeurs_non_finies_rejetees"] for r in out if r["portee"] == "Global")
    exc_total = sum(r["exception"] for r in out if r["portee"] == "Global")
    wcsv("T-motifs-arret", out)
    tex = (f"% T-motifs-arret -- genere par p10_tables.py ; source : {src}\n"
           r"""\begin{table}[htbp]
  \centering
  \begin{threeparttable}
  \caption{Stopping reasons by solver and category over the 156 distinct instances (59, 55 and 42 in Categories~A, B and~C). The five stopping
    reasons are mutually exclusive and add up to the number of instances; convergence coincides with success (first-order stationarity,
    $\norm{\nabla f}\le 10^{-6}$).}
  \label{tab:stop-reasons}
  \footnotesize
  \setlength{\tabcolsep}{3pt}
  \begin{tabular}{@{}llrrrrrrr@{}}
    \toprule
    Solver & Cat. & Instances & Converged & $\Kmax$ reached & \shortstack{Line-search\\failure} & \shortstack{Trust-region\\failure} & Exception & \shortstack{Non-finite\\rejected} \\
    \midrule
""" + "\n".join(body[:-1]) + rf"""
    \bottomrule
  \end{{tabular}}
  \begin{{tablenotes}}\footnotesize
    \item $\Kmax=1000$ iterations. Line-search failure applies to \BFGSA, \BFGSW and \NewtonLM, trust-region failure to Cauchy Point, Dogleg and \SteihaugCG;
    the other cell of each solver is structurally zero. Exception: run aborted by an error ({exc_total} over the six solvers). Non-finite rejected: instances in which at least one
    function or gradient value was non-finite and rejected ({nonf_total} over the six solvers). Of the {fail_total} line-search and trust-region failures, {stag_total} are
    flagged as numerical stagnation (non-exclusive indicator, given in the CSV file).
  \end{{tablenotes}}
  \end{{threeparttable}}
\end{{table}}
""")
    wtex("T-motifs-arret", tex)
    return out


# --------------------------------------------------------------------------------------------- S-tests
def s_tests():
    perm = {(r["famille"], r["solveur_a"], r["solveur_b"]): r for r in rd("etape5_1_tests_permutation.csv")}
    diff = {(r["scope"], r["solveur_a"], r["solveur_b"]): r for r in rd("etape5_2_IC_ecarts_paires.csv")}
    assert set(perm) == set(diff) and len(perm) == 60
    src = "etape5_statistiques.py -> etape5_1_tests_permutation.csv, etape5_2_IC_ecarts_paires.csv"
    pairs = [(a, b) for i, a in enumerate(METHODS) for b in METHODS[i + 1:]]
    rows, blocks = [], {}
    for fam in ("global", "A", "B", "C"):
        body = []
        for a, b in pairs:
            p, d = perm[(fam, a, b)], diff[(fam, a, b)]
            assert abs(float(p["mean_d"]) - float(d["mean_d"])) < 1e-9 and p["n_problemes"] == d["n_problemes"]
            ph = float(p["p_holm"])
            cell_h = pfmt(ph)
            if ph < 0.05:
                cell_h = rf"\textbf{{{cell_h}}}"
            rows.append({"famille": fam, "solveur_a": a, "solveur_b": b, "n_problemes": p["n_problemes"], "ecart_moyen_a_moins_b": p["mean_d"],
                         "IC95_BCa_bas": d["IC95_bas"], "IC95_BCa_haut": d["IC95_haut"], "p_brut": p["p_brut"], "p_Holm": p["p_holm"],
                         "Holm_significatif_0.05": ph < 0.05, "methode_p": p["methode_p"], "source": src})
            body.append(rf"    {NAME[a]} vs {NAME[b]} & {sgn(p['mean_d'])} & {sgn_ci(d['IC95_bas'], d['IC95_haut'])} & {pfmt(p['p_brut'])} & {cell_h} \\")
        blocks[fam] = body
    head = {"global": "Global family (36 problems; $p$ by Monte-Carlo sign-flip, 100\\,000 draws)", "A": "Category~A (15 problems; exact $p$)",
            "B": "Category~B (11 problems; exact $p$)", "C": "Category~C (10 problems; exact $p$)"}

    def one(fams, label, cap_extra):
        inner = []
        for k, fam in enumerate(fams):
            if k:
                inner.append(r"    \midrule")
            inner.append(rf"    \multicolumn{{5}}{{@{{}}l}}{{\textit{{{head[fam]}}}}} \\")
            inner.append(r"    \addlinespace[1pt]")
            inner += blocks[fam]
        return (r"""\begin{table}[htbp]
  \centering
  \begin{threeparttable}
  \caption{Pairwise comparison of the six solvers on the success rate: mean per-problem difference (first solver minus second), 95\,\% BCa
    cluster-bootstrap interval, raw and Holm-adjusted permutation $p$-values. """ + cap_extra + r"""}
  \label{""" + label + r"""}
  \footnotesize
  \setlength{\tabcolsep}{4pt}
  \begin{tabular}{@{}lllrr@{}}
    \toprule
    Pair & Mean difference & 95\,\% CI & Raw $p$ & Holm $p$ \\
    \midrule
""" + "\n".join(inner) + r"""
    \bottomrule
  \end{tabular}
  \begin{tablenotes}\footnotesize
    \item The difference is the mean over problems of the per-problem success proportion of the first solver minus that of the second. Interval: BCa
    bootstrap with problems as clusters, 10\,000 resamples, seed 20260927 (approximate for the categories, which have 10 to 15 clusters).
    $p$: sign-flip permutation test on the per-problem differences (seed 20260927); Holm adjustment within each family of 15 pairs;
    bold: Holm $p<0.05$.
  \end{tablenotes}
  \end{threeparttable}
\end{table}
""")

    tex = (f"% S-tests -- genere par p10_tables.py ; source : {src}\n"
           + one(("global", "A"), "tab:S_tests_a", "Panel~(a): global family and Category~A.") + "\n"
           + one(("B", "C"), "tab:S_tests_b", "Panel~(b): Categories~B and~C."))
    wcsv("S-tests", rows)
    wtex("S-tests", tex)
    return rows


# --------------------------------------------------------------------------------------------- S-couts
def s_couts():
    cost = rd("tl1_ntot_15paires_recalcule.csv")
    assert len(cost) == 15
    src = "tl1_ntot_15paires.py -> tl1_ntot_15paires_recalcule.csv (procedure de 009 ; graine 20260927 consommee sur les 15 paires)"
    rows, body = [], []
    for r in cost:
        lo, hi = float(r["IC95_bas_v05"]), float(r["IC95_haut_v05"])
        excl = "yes ($<1$)" if hi < 1 else ("yes ($>1$)" if lo > 1 else "no")
        rows.append({"solveur_a": r["solveur_a"], "solveur_b": r["solveur_b"], "n_problemes": r["n_problemes_utilises"],
                     "n_instances": r["n_instances_utilisees"], "rapport_geometrique_Ntot_a_sur_b": r["ratio_geo_moyen_v05"],
                     "IC95_BCa_bas": r["IC95_bas_v05"], "IC95_BCa_haut": r["IC95_haut_v05"], "IC_exclut_1": excl,
                     "rapport_v0.4": r["ratio_geo_moyen_v04"], "IC95_bas_v0.4": r["IC95_bas_v04"], "IC95_haut_v0.4": r["IC95_haut_v04"], "source": src})
        body.append(rf"    {{{NAME[r['solveur_a']]}}} / {{{NAME[r['solveur_b']]}}} & {r['n_problemes_utilises']} & {r['n_instances_utilisees']} & "
                    rf"{f(r['ratio_geo_moyen_v05'], 2)} & {ci(r['IC95_bas_v05'], r['IC95_haut_v05'], 2)} & {excl} \\")
    wcsv("S-couts", rows)
    cat = [r for r in rd("etape5_3_couts.csv")]
    wcsv("S-couts_par_portee", [{**r, "source": "etape5_statistiques.py -> etape5_3_couts.csv (flux aleatoire de l'etape 5.3 ; peut differer de S-couts au 3e chiffre)"}
                                for r in cat])
    tex = (f"% S-couts -- genere par p10_tables.py ; source : {src}\n"
           r"""\begin{table}[htbp]
  \centering
  \begin{threeparttable}
  \caption{Oracle-cost ratios $\Ntot(a)/\Ntot(b)$ for the 15 solver pairs, over the instances solved by both solvers: geometric mean per problem,
    then geometric mean over problems, with a 95\,\% BCa cluster-bootstrap interval (problems as clusters, 10\,000 resamples, seed 20260927).
    A ratio above~1 means that the first solver uses more oracle calls. No $p$-value is attached to a cost ratio.}
  \label{tab:S_costs}
  \footnotesize
  \setlength{\tabcolsep}{4pt}
  \begin{tabular}{@{}lrrlll@{}}
    \toprule
    Pair $(a/b)$ & Problems & Instances & Ratio & 95\,\% CI & CI excludes 1 \\
    \midrule
""" + "\n".join(body) + r"""
    \bottomrule
  \end{tabular}
  \begin{tablenotes}\footnotesize
    \item $\Ntot=N_f+N_g+N_h$ counted with the evaluation memory (tag \texttt{v0.5}). The ratio is computed on the 156 distinct instances; the pair
    \BFGSA\,/\,\BFGSW is the one reported in Table~\ref{tab:L1}. The ratios by category are in the CSV file accompanying this table.
  \end{tablenotes}
  \end{threeparttable}
\end{table}
""")
    wtex("S-couts", tex)
    return rows


def main():
    os.makedirs(TABLES_DIR, exist_ok=True)
    t_motifs()
    s_tests()
    s_couts()
    print("tables ecrites dans", TABLES_DIR)


if __name__ == "__main__":
    main()
