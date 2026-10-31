# -*- coding: utf-8 -*-
"""
Consigne 007, Etape 5 (D6) : statistiques. Unite = le probleme (36 grappes :
15 A, 11 B, 10 C). Aucun test fonde sur les rangs.
AUCUNE p-valeur sur les couts (5.3). Lecture seule, aucune execution.

5.1 Tests par paires (15 paires) sur d(p)=s(p,a)-s(p,b) : permutation par
    inversion de signe au niveau du probleme ; enumeration exacte par
    categorie (2^n_problemes) ; Monte-Carlo (100000, graine notee) au global.
    Holm dans chaque famille (global ; A ; B ; C).
5.2 IC des taux de succes (par solveur) et des ecarts entre paires :
    bootstrap par grappes (probleme = grappe), B=10000, BCa, graine notee.
5.3 Couts : ratio N_tot(a)/N_tot(b) sur les instances resolues par les deux,
    moyenne geometrique par probleme puis sur les problemes ; IC bootstrap
    par grappes. Aucune p-valeur.
5.4 Sensibilite (supplement) : McNemar exact sur les 156 instances distinctes,
    Clopper-Pearson (independance supposee, a signaler).
"""
import csv
import itertools
import json
import math

import numpy as np
from scipy import stats as sps

from common_data import load_distinct_rows, problems_by_category, success_matrix, ntot_by_instance, METHODS, CATEGORIES
from paths import OUT_DIR  # consigne 009e : sorties sous analyse_v0.5/
SEED_MC = 20260927
SEED_BOOT = 20260927
B_BOOT = 10000
N_MC = 100000

PAIRS = list(itertools.combinations(METHODS, 2))  # 15 paires


# ───────────────────────── 5.1 : permutation par inversion de signe ─────────

def sign_flip_pvalue_exact(d):
    """d : array des ecarts par probleme. Enumeration exacte des 2^n configurations."""
    n = len(d)
    obs = abs(np.mean(d))
    count_ge = 0
    total = 2 ** n
    for bits in range(total):
        signs = np.array([1 if (bits >> i) & 1 else -1 for i in range(n)])
        m = abs(np.mean(d * signs))
        if m >= obs - 1e-15:
            count_ge += 1
    return count_ge / total


def sign_flip_pvalue_mc(d, n_draws, rng):
    n = len(d)
    obs = abs(np.mean(d))
    signs = rng.choice([-1, 1], size=(n_draws, n))
    means = np.abs(signs @ d / n)
    count_ge = np.sum(means >= obs - 1e-15) + 1  # +1 pour la configuration observee elle-meme
    return count_ge / (n_draws + 1)


def holm(pvals):
    idx = np.argsort(pvals)
    n = len(pvals)
    adj = np.empty(n)
    running_max = 0.0
    for rank, i in enumerate(idx):
        factor = n - rank
        val = min(1.0, factor * pvals[i])
        running_max = max(running_max, val)
        adj[i] = running_max
    return adj


def etape5_1(s_matrix, probs_by_cat, all_probs):
    rows_out = []
    rng = np.random.default_rng(SEED_MC)

    families = {"global": all_probs}
    for c in CATEGORIES:
        families[c] = probs_by_cat[c]

    for fam_name, fam_probs in families.items():
        fam_rows = []
        for a, b in PAIRS:
            d = np.array([s_matrix[p][a] - s_matrix[p][b] for p in fam_probs])
            mean_d = float(np.mean(d))
            if fam_name == "global":
                p_raw = sign_flip_pvalue_mc(d, N_MC, rng)
                methode_p = f"Monte-Carlo, {N_MC} tirages, graine={SEED_MC}"
            else:
                p_raw = sign_flip_pvalue_exact(d)
                methode_p = f"enumeration exacte (2^{len(fam_probs)}={2**len(fam_probs)} configurations)"
            fam_rows.append({"famille": fam_name, "solveur_a": a, "solveur_b": b,
                              "n_problemes": len(fam_probs), "mean_d": round(mean_d, 4),
                              "p_brut": p_raw, "methode_p": methode_p})
        p_holm = holm(np.array([r["p_brut"] for r in fam_rows]))
        for r, ph in zip(fam_rows, p_holm):
            r["p_holm"] = float(ph)
        rows_out.extend(fam_rows)
        print(f"--- Famille {fam_name} ({len(fam_probs)} problemes) : Holm sur {len(fam_rows)} paires ---")
        for r in sorted(fam_rows, key=lambda r: r["p_holm"])[:5]:
            print(f"  {r['solveur_a']:13s} vs {r['solveur_b']:13s}  mean_d={r['mean_d']:+.4f}  "
                  f"p_brut={r['p_brut']:.5f}  p_holm={r['p_holm']:.5f}")

    with open(OUT_DIR + r"\etape5_1_tests_permutation.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["famille", "solveur_a", "solveur_b", "n_problemes",
                                            "mean_d", "p_brut", "p_holm", "methode_p"])
        w.writeheader()
        w.writerows(rows_out)
    print(f"\nEcrit : {OUT_DIR}\\etape5_1_tests_permutation.csv ({len(rows_out)} lignes)")
    return rows_out


# ───────────────────────── 5.2 : bootstrap par grappes (BCa) ─────────────────

def bca_ci(values, rng, B=B_BOOT, alpha=0.05):
    """values : array 1D, une valeur par grappe (probleme). Statistique = moyenne."""
    values = np.asarray(values, dtype=float)
    n = len(values)
    theta_hat = float(np.mean(values))

    idx_boot = rng.integers(0, n, size=(B, n))
    boot_means = values[idx_boot].mean(axis=1)

    # Biais : z0
    prop_less = np.mean(boot_means < theta_hat)
    prop_less = min(max(prop_less, 1.0 / (B + 1)), 1 - 1.0 / (B + 1))
    z0 = sps.norm.ppf(prop_less)

    # Acceleration : jackknife
    jack_means = np.array([np.mean(np.delete(values, i)) for i in range(n)])
    jack_mean_of_means = np.mean(jack_means)
    num = np.sum((jack_mean_of_means - jack_means) ** 3)
    den = 6.0 * (np.sum((jack_mean_of_means - jack_means) ** 2) ** 1.5)
    a_hat = num / den if den != 0 else 0.0

    z_lo = sps.norm.ppf(alpha / 2)
    z_hi = sps.norm.ppf(1 - alpha / 2)

    def adj_percentile(z):
        num_ = z0 + z
        denom_ = 1 - a_hat * num_
        pct = sps.norm.cdf(z0 + num_ / denom_)
        return min(max(pct, 0.0001), 0.9999)

    lo = np.percentile(boot_means, 100 * adj_percentile(z_lo))
    hi = np.percentile(boot_means, 100 * adj_percentile(z_hi))
    return theta_hat, lo, hi


def etape5_2(s_matrix, probs_by_cat, all_probs):
    rng = np.random.default_rng(SEED_BOOT)
    rows_rate = []
    scopes = {"global": all_probs, **{c: probs_by_cat[c] for c in CATEGORIES}}

    for method in METHODS:
        for scope_name, probs in scopes.items():
            vals = np.array([s_matrix[p][method] for p in probs])
            theta, lo, hi = bca_ci(vals, rng)
            approx = "approximatif (10 a 15 grappes)" if scope_name != "global" else ""
            rows_rate.append({"method": method, "scope": scope_name, "n_problemes": len(probs),
                               "taux_succes_moyen_par_probleme": round(theta, 4),
                               "IC95_bas": round(lo, 4), "IC95_haut": round(hi, 4), "note": approx})
    with open(OUT_DIR + r"\etape5_2_IC_taux_succes.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows_rate[0].keys()))
        w.writeheader()
        w.writerows(rows_rate)
    print(f"Ecrit : {OUT_DIR}\\etape5_2_IC_taux_succes.csv ({len(rows_rate)} lignes)")

    rows_diff = []
    for scope_name, probs in scopes.items():
        for a, b in PAIRS:
            d = np.array([s_matrix[p][a] - s_matrix[p][b] for p in probs])
            theta, lo, hi = bca_ci(d, rng)
            approx = "approximatif (10 a 15 grappes)" if scope_name != "global" else ""
            rows_diff.append({"scope": scope_name, "solveur_a": a, "solveur_b": b,
                               "n_problemes": len(probs), "mean_d": round(theta, 4),
                               "IC95_bas": round(lo, 4), "IC95_haut": round(hi, 4), "note": approx})
    with open(OUT_DIR + r"\etape5_2_IC_ecarts_paires.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows_diff[0].keys()))
        w.writeheader()
        w.writerows(rows_diff)
    print(f"Ecrit : {OUT_DIR}\\etape5_2_IC_ecarts_paires.csv ({len(rows_diff)} lignes)")
    return rows_rate, rows_diff


# ───────────────────────── 5.3 : couts (aucune p-valeur) ─────────────────────

def etape5_3(ntot, succ, probs_by_cat, all_probs):
    rng = np.random.default_rng(SEED_BOOT + 1)
    rows_out = []
    scopes = {"global": all_probs, **{c: probs_by_cat[c] for c in CATEGORIES}}

    for scope_name, probs in scopes.items():
        for a, b in PAIRS:
            per_problem_gm = []
            n_inst_used = 0
            n_prob_used = 0
            for p in probs:
                ratios = []
                for n_key in ntot.get(p, {}):
                    if a in succ[p][n_key] and b in succ[p][n_key] and succ[p][n_key][a] and succ[p][n_key][b]:
                        ratios.append(ntot[p][n_key][a] / ntot[p][n_key][b])
                if ratios:
                    per_problem_gm.append(sps.gmean(ratios))
                    n_inst_used += len(ratios)
                    n_prob_used += 1
            if not per_problem_gm:
                rows_out.append({"scope": scope_name, "solveur_a": a, "solveur_b": b,
                                  "n_problemes_utilises": 0, "n_instances_utilisees": 0,
                                  "ratio_geo_moyen": None, "IC95_bas": None, "IC95_haut": None,
                                  "note": "aucune instance resolue par les deux solveurs"})
                continue
            per_problem_gm = np.array(per_problem_gm)
            log_gm = np.log(per_problem_gm)
            theta_log, lo_log, hi_log = bca_ci(log_gm, rng)
            rows_out.append({"scope": scope_name, "solveur_a": a, "solveur_b": b,
                              "n_problemes_utilises": n_prob_used, "n_instances_utilisees": n_inst_used,
                              "ratio_geo_moyen": round(math.exp(theta_log), 4),
                              "IC95_bas": round(math.exp(lo_log), 4), "IC95_haut": round(math.exp(hi_log), 4),
                              "note": "approximatif (10 a 15 grappes)" if scope_name != "global" else ""})
    with open(OUT_DIR + r"\etape5_3_couts.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows_out[0].keys()))
        w.writeheader()
        w.writerows(rows_out)
    print(f"Ecrit : {OUT_DIR}\\etape5_3_couts.csv ({len(rows_out)} lignes) -- aucune p-valeur (regle 007)")
    return rows_out


# ───────────────────────── 5.4 : McNemar exact + Clopper-Pearson ────────────

def clopper_pearson(n_success, n_total, alpha=0.05):
    if n_total == 0:
        return None, None
    lo = 0.0 if n_success == 0 else sps.beta.ppf(alpha / 2, n_success, n_total - n_success + 1)
    hi = 1.0 if n_success == n_total else sps.beta.ppf(1 - alpha / 2, n_success + 1, n_total - n_success)
    return lo, hi


def etape5_4(rows):
    inst_success = {}
    for r in rows:
        inst_success.setdefault((r["problem"], r["n"]), {})[r["method"]] = (r["status"] == "success")
    instances = list(inst_success.keys())
    n_inst = len(instances)
    assert n_inst == 156, n_inst

    cp_rows = []
    for method in METHODS:
        n_succ = sum(1 for k in instances if inst_success[k][method])
        lo, hi = clopper_pearson(n_succ, n_inst)
        cp_rows.append({"method": method, "n_succes": n_succ, "n_instances": n_inst,
                         "taux": round(n_succ / n_inst, 4), "CP95_bas": round(lo, 4), "CP95_haut": round(hi, 4),
                         "note": "suppose les 156 instances independantes (approximation, cf. R3.11/D6)"})
    with open(OUT_DIR + r"\etape5_4_clopper_pearson.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(cp_rows[0].keys()))
        w.writeheader()
        w.writerows(cp_rows)
    print(f"Ecrit : {OUT_DIR}\\etape5_4_clopper_pearson.csv")

    mcnemar_rows = []
    for a, b in PAIRS:
        n01 = sum(1 for k in instances if (not inst_success[k][a]) and inst_success[k][b])
        n10 = sum(1 for k in instances if inst_success[k][a] and (not inst_success[k][b]))
        n_discordant = n01 + n10
        if n_discordant == 0:
            p_exact = 1.0
        else:
            p_exact = sps.binomtest(min(n01, n10), n_discordant, 0.5, alternative="two-sided").pvalue
        mcnemar_rows.append({"solveur_a": a, "solveur_b": b, "n01_b_seul": n01, "n10_a_seul": n10,
                              "n_discordant": n_discordant, "p_exact_McNemar": p_exact,
                              "note": "suppose les 156 instances independantes (approximation, cf. R3.11/D6)"})
    p_holm_mc = holm(np.array([r["p_exact_McNemar"] for r in mcnemar_rows]))
    for r, ph in zip(mcnemar_rows, p_holm_mc):
        r["p_holm"] = float(ph)
    with open(OUT_DIR + r"\etape5_4_mcnemar.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(mcnemar_rows[0].keys()))
        w.writeheader()
        w.writerows(mcnemar_rows)
    print(f"Ecrit : {OUT_DIR}\\etape5_4_mcnemar.csv")
    return cp_rows, mcnemar_rows


def main():
    rows = load_distinct_rows()
    probs_by_cat = problems_by_category(rows)
    all_probs = sorted({p for c in CATEGORIES for p in probs_by_cat[c]})
    assert len(all_probs) == 36 and [len(probs_by_cat[c]) for c in CATEGORIES] == [15, 11, 10]
    s_matrix, n_inst = success_matrix(rows)
    ntot, succ = ntot_by_instance(rows)

    print("=== 5.1 Tests par paires (permutation, inversion de signe) ===")
    etape5_1(s_matrix, probs_by_cat, all_probs)

    print("\n=== 5.2 Intervalles de confiance (bootstrap par grappes, BCa) ===")
    etape5_2(s_matrix, probs_by_cat, all_probs)

    print("\n=== 5.3 Couts (aucune p-valeur) ===")
    etape5_3(ntot, succ, probs_by_cat, all_probs)

    print("\n=== 5.4 Sensibilite (McNemar exact, Clopper-Pearson) ===")
    etape5_4(rows)

    meta = {"seed_monte_carlo_5_1": SEED_MC, "n_mc_5_1": N_MC, "seed_bootstrap_5_2_5_3": SEED_BOOT,
            "B_bootstrap": B_BOOT}
    with open(OUT_DIR + r"\etape5_meta_graines.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    print(f"\nGraines : {meta}")


if __name__ == "__main__":
    main()
