# -*- coding: utf-8 -*-
"""
Consigne 009e (reprise de 008d-bis), etape 2 (reprise de 008d) : temps estimes.

Modele (a UN terme : la colonne cg_inner_total n'existe pas dans
resultats_benchmark_v0.4.csv) :
    temps_estime(p, n, m) = n_iter(p, n, m) x t_iter(n, m)
ou t_iter(n, m) = mediane, sur les problemes de Categorie A a la dimension n, du temps
reel par iteration du solveur m (TOUTES les executions, reussies ou non ; iterations >= 1).

DECISION 5 (008d-bis) : un temps estime n'est utilise QUE lorsqu'aucune mesure n'existe pour
la tache. Une tache mesuree mais "faible precision" (< 1 ms) n'est donc PAS remplacee par
un temps estime : elle est exclue des rapports de temps (analyse principale) ; elle n'est
utilisee que dans la sensibilite "faible precision incluse" (allow_low_precision=True).
Avec la nuit v0.5 (comme en 009c), les seules taches sans mesure sont les solveurs autres que Newton_BT et
Steihaug_CG en Categories B et C a n >= 500 (152 taches) : verifie dans main().

Validations (mediane du rapport estime / mesure dans [0,5 ; 2] ; sinon le solveur n'utilise
que des temps mesures) :
  - A3bis (76 taches : Newton_BT et Steihaug_CG, Cat. B/C, n >= 500)  [consigne 008d]
  - complement 008d : "un probleme exclu a la fois" sur la Categorie A, six solveurs
  - NOUVEAU (008d-bis) : A3ter (Cat. B/C, n < 500, six solveurs ; seulement si la nuit
    mesuree contient ce bloc, i.e. v0.5) : validation hors categorie de tous les solveurs.
Un solveur n'utilise ses temps estimes que si TOUTES les validations disponibles pour lui
passent le critere.
"""
import csv
import json
import os
import statistics as st

import numpy as np

from common_data import load_distinct_rows, METHODS
from common_timing import get_timing, LOW_PREC_S
from paths import OUT_DIR

VALID_LO, VALID_HI = 0.5, 2.0
FOUR_OTHERS = ["Cauchy_Point", "Dogleg", "BFGS_Armijo", "BFGS_Wolfe"]


def per_iter_table(rows, wall, exclude_problem=None):
    """{(n, m): (t_iter mediane, n_problemes)} sur la Categorie A (toutes executions)."""
    vals = {}
    for r in rows:
        if r["category"] != "A" or r["problem"] == exclude_problem:
            continue
        k = (r["problem"], r["n"], r["method"])
        it = int(r["n_iter"])
        t = wall.get(k)
        if t is None or it < 1:
            continue
        vals.setdefault((int(r["n"]), r["method"]), {}).setdefault(r["problem"], []).append(t / it)
    out = {}
    for key, byp in vals.items():
        per_problem = [st.median(v) for v in byp.values()]
        out[key] = (st.median(per_problem), len(per_problem))
    return out


def build(timing, exclude_problem=None):
    """exclude_problem : t_iter calcule SANS ce probleme (validation par plis : pas de fuite de temps)."""
    rows = load_distinct_rows()
    wall = timing.wall
    t_iter = per_iter_table(rows, wall, exclude_problem=exclude_problem)
    recs = []
    for r in rows:
        k = (r["problem"], r["n"], r["method"])
        n_iter = int(r["n_iter"])
        meas = wall.get(k)
        ti = t_iter.get((int(r["n"]), r["method"]))
        est = n_iter * ti[0] if ti is not None else None
        recs.append({"problem": r["problem"], "n": int(r["n"]), "category": r["category"], "method": r["method"],
                     "status": r["status"], "n_iter": n_iter, "n_total": int(r["n_total"]),
                     "bloc": timing.bloc.get(k, ""), "mesure_s": meas,
                     "faible_precision": (meas is not None and meas < LOW_PREC_S),
                     "estime_s": est, "n_pb_ref_t_iter": ti[1] if ti is not None else 0})
    return rows, wall, t_iter, recs


def _summ(rs):
    med, p5, p95 = (float(x) for x in np.percentile(rs, [50, 5, 95]))
    return {"n_taches": len(rs), "mediane": round(med, 4), "p5": round(p5, 4), "p95": round(p95, 4),
            "mediane_dans_[0.5;2]": VALID_LO <= med <= VALID_HI}


def validation_bloc(recs, bloc, methods):
    """Validation sur un bloc de taches mesurees (>= 1 ms, estimation disponible)."""
    out = {}
    for m in methods:
        rs = [r["estime_s"] / r["mesure_s"] for r in recs
              if r["bloc"] == bloc and r["method"] == m and r["mesure_s"] and r["mesure_s"] >= LOW_PREC_S
              and r["estime_s"] is not None and r["n_iter"] >= 1]
        if rs:
            out[m] = _summ(rs)
    return out


def validation_a3bis(recs):
    """Demandee par 008d : tous les A3bis (Newton_BT, Steihaug_CG), meme < 1 ms s'il y en avait (mesure > 0)."""
    out = {}
    for m in ("Newton_BT", "Steihaug_CG"):
        rs = [r["estime_s"] / r["mesure_s"] for r in recs
              if r["bloc"] == "A3bis" and r["method"] == m and r["mesure_s"] and r["mesure_s"] > 0
              and r["estime_s"] is not None]
        if rs:
            out[m] = _summ(rs)
    return out


def validation_loo_catA(rows, wall, recs):
    out = {}
    catA_problems = sorted({r["problem"] for r in rows if r["category"] == "A"})
    ratios = {m: [] for m in METHODS}
    cache = {p: per_iter_table(rows, wall, exclude_problem=p) for p in catA_problems}
    for r in recs:
        if r["category"] != "A" or r["mesure_s"] is None or r["mesure_s"] < LOW_PREC_S or r["n_iter"] < 1:
            continue
        ti = cache[r["problem"]].get((r["n"], r["method"]))
        if ti is None:
            continue
        ratios[r["method"]].append((r["n"], (r["n_iter"] * ti[0]) / r["mesure_s"]))
    for m in METHODS:
        vals = [v for _, v in ratios[m]]
        out[m] = _summ(vals) if vals else {"n_taches": 0}
    return out, ratios


def usable_estimates(val_sets):
    """val_sets : liste de dicts {methode: resume}. Un solveur utilise ses estimes si TOUTES les validations
    disponibles pour lui passent le critere (et s'il en existe au moins une)."""
    use = {}
    for m in METHODS:
        vs = [v[m] for v in val_sets if m in v and v[m].get("n_taches", 0) > 0]
        use[m] = bool(vs) and all(x["mediane_dans_[0.5;2]"] for x in vs)
    return use


class TimeBook:
    """Acces aux temps utilises : (valeur, source) ; source in {'mesure', 'estime'} ou (None, None).

    Regle (decision 5) : temps mesure (>= 1 ms) s'il existe ; tache mesuree mais < 1 ms ("faible precision") :
    exclue (None) sauf allow_low_precision=True (sensibilite) ; temps estime seulement si AUCUNE mesure n'existe
    pour la tache (et si le solveur passe la validation du modele)."""

    def __init__(self, recs, use_estimate):
        self.rec = {(r["problem"], r["n"], r["method"]): r for r in recs}
        self.use_estimate = use_estimate

    def get(self, p, n, m, allow_estimate=True, allow_low_precision=False):
        r = self.rec.get((p, int(n), m))
        if r is None:
            return None, None
        if r["mesure_s"] is not None:
            if (not r["faible_precision"]) or allow_low_precision:
                return r["mesure_s"], "mesure"
            return None, None
        if allow_estimate and self.use_estimate.get(m) and r["estime_s"] is not None:
            return r["estime_s"], "estime"
        return None, None


def validations(timing, rows, wall, recs):
    va = validation_a3bis(recs)
    vl, loo_ratios = validation_loo_catA(rows, wall, recs)
    v3t = validation_bloc(recs, "A3ter", METHODS)
    sets = [va, vl] + ([v3t] if v3t else [])
    use = usable_estimates(sets)
    return va, vl, v3t, use, loo_ratios


def load_timebook(timing):
    rows, wall, t_iter, recs = build(timing)
    va, vl, v3t, use, _ = validations(timing, rows, wall, recs)
    return TimeBook(recs, use), recs, va, vl, use


def timebook_excluding(problem, use_estimate, timing):
    """TimeBook dont le modele t_iter ignore `problem` (plis de la validation hors echantillon)."""
    _, _, _, recs = build(timing, exclude_problem=problem)
    return TimeBook(recs, use_estimate)


def main(variant="principale"):
    timing = get_timing(variant)
    rows, wall, t_iter, recs = build(timing)
    va, vl, v3t, use, loo_ratios = validations(timing, rows, wall, recs)

    with open(os.path.join(OUT_DIR, "etape2_t_iter.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["n", "method", "t_iter_median_s", "n_problemes_CatA"])
        for (n, m), (t, k) in sorted(t_iter.items()):
            w.writerow([n, m, f"{t:.6e}", k])

    tb = TimeBook(recs, use)
    cov = {}
    with open(os.path.join(OUT_DIR, "etape2_temps_par_instance.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["problem", "n", "category", "method", "status", "n_iter", "n_total", "bloc", "mesure_s",
                    "faible_precision", "estime_s", "n_pb_ref_t_iter", "temps_utilise_s", "source"])
        for r in sorted(recs, key=lambda r: (r["category"], r["problem"], r["n"], r["method"])):
            v, src = tb.get(r["problem"], r["n"], r["method"])
            if src is None:
                src_out = "faible_precision" if r["faible_precision"] else "aucun"
            else:
                src_out = src
            c = cov.setdefault(r["method"], {"mesure": 0, "estime": 0, "faible_precision": 0, "aucun": 0})
            c[src_out] += 1
            w.writerow([r["problem"], r["n"], r["category"], r["method"], r["status"], r["n_iter"], r["n_total"],
                        r["bloc"], "" if r["mesure_s"] is None else f"{r['mesure_s']:.6e}",
                        int(r["faible_precision"]), "" if r["estime_s"] is None else f"{r['estime_s']:.6e}",
                        r["n_pb_ref_t_iter"], "" if v is None else f"{v:.6e}", src_out])

    # Decision 5 : verification des taches sans aucune mesure
    unmeasured = [r for r in recs if r["mesure_s"] is None]
    shape_ok = all(r["category"] in ("B", "C") and r["n"] >= 500 and r["method"] in FOUR_OTHERS for r in unmeasured)
    decision5 = {"n_taches_sans_mesure": len(unmeasured), "toutes_B_ou_C_n_ge_500_solveurs_autres_que_NLM_SCG": shape_ok,
                 "par_solveur": {m: sum(1 for r in unmeasured if r["method"] == m) for m in METHODS}}

    loo_by_n = {}
    for m, lst in loo_ratios.items():
        for n, v in lst:
            loo_by_n.setdefault((m, n), []).append(v)
    with open(os.path.join(OUT_DIR, "etape2_validation_LOO_CatA_par_dimension.csv"), "w", newline="",
              encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["method", "n", "n_taches", "mediane_estime_sur_mesure"])
        for (m, n), v in sorted(loo_by_n.items()):
            w.writerow([m, n, len(v), round(float(np.median(v)), 4)])

    res = {"variante": variant, "modele": "un terme : n_iter x t_iter(n, m) ; colonne cg_inner_total absente de resultats_benchmark_v0.5.csv",
           "decision_5_taches_sans_mesure": decision5,
           "validation_A3bis_76_taches": va, "validation_A3ter_B_C_n_inf_500_six_solveurs": v3t,
           "complement_validation_LOO_CatA": vl, "estimations_utilisables": use,
           "couverture_temps_par_solveur_936_instances": cov}
    with open(os.path.join(OUT_DIR, "etape2_validation.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2, ensure_ascii=False)
    print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
