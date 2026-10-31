# -*- coding: utf-8 -*-
"""
data_profiles.py — Gap G8 / Bloc B09
------------------------------------------------------------
Data profiles (Moré & Wild 2009) pour le benchmark Phase 0.
Trois sous-graphes : Cat. A (mal conditionné), B (non-convexe), C (H indéfinie).

Formule :  alpha_{p,s} = N_tot / (n+1)   si convergé
                        = inf              sinon

d_s(alpha) = (1/N_p) * |{p : alpha_{p,s} <= alpha}|

Usage :
  python data_profiles.py <resultats.csv> [dossier_sortie]

Exemple :
  python data_profiles.py resultats_benchmark_v2.csv .
"""
import sys
import os
import csv
import numpy as np

# ── Import matplotlib avec message clair si absent ────────────────────────────
try:
    import matplotlib
    matplotlib.use("Agg")          # pas besoin d'affichage interactif
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
except ImportError:
    sys.exit(
        "matplotlib est absent. Installez-le avec :\n"
        "  pip install matplotlib\n"
        "puis relancez ce script."
    )

# ─────────────────────────────────────────────────────────────────────────────
# Paramètres graphiques
# ─────────────────────────────────────────────────────────────────────────────

# Couleurs et styles cohérents avec les profils Dolan-Moré du rapport
STYLE = {
    # (couleur, style de ligne, épaisseur)
    "Cauchy Point":          ("#1f77b4", "-",  1.8),
    "Dogleg":                ("#ff7f0e", "-",  1.8),
    "Steihaug-CG":           ("#2ca02c", "-",  2.2),   # trait plus épais = résultat central
    "BFGS + Armijo":         ("#d62728", "--", 1.6),
    "BFGS + Wolfe":          ("#9467bd", "--", 1.6),
    "Newton + Backtracking": ("#8c564b", "--", 1.6),
}

# Catégories attendues dans le CSV (valeurs de la colonne 'category')
CAT_LABELS = {
    "A": "Cat. A — Mal conditionnés",
    "B": "Cat. B — Non-convexes",
    "C": "Cat. C — Hessienne indéfinie",
}


# ─────────────────────────────────────────────────────────────────────────────
# Chargement du CSV
# ─────────────────────────────────────────────────────────────────────────────

def _parse_bool(val: str) -> bool:
    return val.strip().lower() in ("true", "1", "yes", "oui")


def load_csv(path: str) -> list[dict]:
    """
    Charge le CSV produit par framework_united_final.py.
    Colonnes requises : category, method, n, converged, n_total.
    Compatible v1 (sans budget_ratio) et v2 (avec budget_ratio).
    """
    rows = []
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            try:
                rows.append({
                    "category":  row["category"].strip(),
                    "method":    row["method"].strip(),
                    "n":         int(row["n"]),
                    "converged": _parse_bool(row["converged"]),
                    "n_total":   int(row["n_total"]),
                })
            except (KeyError, ValueError):
                continue   # ligne malformée ignorée
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Calcul du data profile
# ─────────────────────────────────────────────────────────────────────────────

def compute_profile(rows: list[dict], method: str, category: str):
    """
    Retourne (alphas, fracs) pour la fonction en escalier du data profile.

    alphas : abscisses triées (alpha = n_total / (n+1))
    fracs  : ordonnées correspondantes (fraction de problèmes résolus)
    """
    subset = [r for r in rows
              if r["category"] == category and r["method"] == method]
    if not subset:
        return None

    n_probs = len(subset)
    finite_alphas = sorted(
        r["n_total"] / (r["n"] + 1)
        for r in subset if r["converged"]
    )

    if not finite_alphas:
        # Solveur qui ne converge jamais sur cette catégorie
        return np.array([0.0]), np.array([0.0])

    xs = np.array(finite_alphas)
    ys = np.arange(1, len(xs) + 1) / n_probs

    # On préfixe avec un point en (0, 0) pour avoir la fonction complète
    xs = np.concatenate([[xs[0] * 0.9], xs])
    ys = np.concatenate([[0.0],         ys])

    return xs, ys


# ─────────────────────────────────────────────────────────────────────────────
# Tracé
# ─────────────────────────────────────────────────────────────────────────────

def plot_data_profiles(rows: list[dict], output_dir: str) -> None:

    # Détecter les méthodes présentes dans le CSV
    methods_in_csv = sorted({r["method"] for r in rows})

    # Correspondance robuste avec STYLE : exact d'abord, puis sous-chaîne
    # Correspondance explicite noms CSV → clés STYLE
    CSV_ALIASES = {
        "Cauchy_Point":  "Cauchy Point",
        "Dogleg":        "Dogleg",
        "Steihaug_CG":   "Steihaug-CG",
        "BFGS_Armijo":   "BFGS + Armijo",
        "BFGS_Wolfe":    "BFGS + Wolfe",
        "Newton_BT":     "Newton + Backtracking",
    }

    def match_style(method: str):
        # 1. Exact
        if method in STYLE:
            return method, STYLE[method]
        # 2. Alias CSV explicite
        if method in CSV_ALIASES:
            key = CSV_ALIASES[method]
            if key in STYLE:
                return key, STYLE[key]
        # 3. Sous-chaîne normalisée (fallback souple)
        m_lower = method.lower().replace("_", " ").replace("-", " ")
        for key in STYLE:
            k_lower = key.lower().replace("_", " ").replace("-", " ")
            if k_lower in m_lower or m_lower in k_lower:
                return key, STYLE[key]
        # 4. Gris par défaut
        return method, ("#888888", ":", 1.2)

    # Catégories présentes
    cats_in_csv = sorted({r["category"] for r in rows})
    cats_to_plot = [c for c in ["A", "B", "C"] if c in cats_in_csv]
    if not cats_to_plot:
        cats_to_plot = cats_in_csv[:3]

    ncats = len(cats_to_plot)
    fig, axes = plt.subplots(1, ncats, figsize=(5.5 * ncats, 5), sharey=True)
    if ncats == 1:
        axes = [axes]

    fig.suptitle(
        "Data profiles — Moré & Wild (2009)\n"
        r"$\alpha = N_{\mathrm{tot}}\,/\,(n+1)$   "
        r"|   critère : $\|\nabla f(x_k)\| < 10^{-6}$",
        fontsize=12, y=1.02
    )

    for ax, cat in zip(axes, cats_to_plot):

        # Plage x : 98e percentile des alphas finis, toutes méthodes
        all_finite = [
            r["n_total"] / (r["n"] + 1)
            for r in rows
            if r["category"] == cat and r["converged"]
        ]
        if all_finite:
            alpha_max = float(np.percentile(all_finite, 98)) * 1.15
        else:
            alpha_max = 1000.0

        for method in methods_in_csv:
            result = compute_profile(rows, method, cat)
            if result is None:
                continue
            xs, ys = result

            label, (color, ls, lw) = match_style(method)

            # Prolonger jusqu'à alpha_max
            xs_ext = np.append(xs, alpha_max)
            ys_ext = np.append(ys, ys[-1])

            ax.step(xs_ext, ys_ext,
                    where="post",
                    color=color, linestyle=ls, linewidth=lw,
                    label=label)

        # Mise en forme
        ax.set_xscale("log")
        ax.set_xlim(left=max(0.3, min(all_finite) * 0.5) if all_finite else 0.3,
                    right=alpha_max)
        ax.set_ylim(-0.02, 1.05)
        ax.set_xlabel(r"$\alpha$  (évaluations / $(n+1)$)", fontsize=10)
        ax.set_title(CAT_LABELS.get(cat, f"Cat. {cat}"), fontsize=11)
        ax.grid(True, which="both", linestyle=":", alpha=0.35)
        ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))
        ax.tick_params(axis="both", labelsize=9)

    axes[0].set_ylabel("Proportion de problèmes résolus", fontsize=10)
    axes[-1].legend(loc="lower right", fontsize=8.5, framealpha=0.92)

    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    for ext in ("png", "pdf"):
        out = os.path.join(output_dir, f"data_profiles_phase0.{ext}")
        plt.savefig(out, dpi=150 if ext == "png" else None, bbox_inches="tight")
        print(f"  Sauvegardé : {out}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Point d'entrée
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    csv_path   = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else (
        os.path.dirname(os.path.abspath(csv_path))
    )

    if not os.path.isfile(csv_path):
        sys.exit(f"Fichier introuvable : {csv_path}")

    print(f"Chargement : {csv_path}")
    rows = load_csv(csv_path)
    print(f"  {len(rows)} lignes — "
          f"{len({r['method'] for r in rows})} méthodes — "
          f"catégories : {sorted({r['category'] for r in rows})}")

    print("Génération des data profiles…")
    plot_data_profiles(rows, output_dir)
    print("Terminé.")


if __name__ == "__main__":
    main()
