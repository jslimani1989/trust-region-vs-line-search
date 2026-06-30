#!/usr/bin/env python3
"""
sparse_example.py — Illustration §9 (G5, sparse Hessians)
============================================================
Démontre numériquement la Proposition de §9.1 (PLAN_S9_SPARSE.md, B38/B39) :
sous sparsité, Steihaug-CG (matvec-only, O(nnz)/iter) ne subit aucun fill-in,
tandis que les méthodes à factorisation (Newton-LM, Dogleg) en subissent un,
borné uniquement dans le pire cas par O(n^3).

Problème : quadratique f(x) = 1/2 x^T A x sur Hessienne A pentadiagonale SPD
(bande b=2, nnz(A) = O(n)), construite diagonalement dominante (Gershgorin).

Équivalence exploitée (déjà validée Phase 1, test_steihaug_newton.py) :
sur f exactement quadratique et A SPD globalement, Steihaug-CG avec Delta->inf
se réduit à CG standard résolvant A p = -g exactement — aucune troncature de
courbure négative, aucune frontière. Un seul "pas Newton" suffit donc à
illustrer le mécanisme, sans boucle d'optimisation complète (cohérent avec
"illustration ciblée" du plan, pas une nouvelle campagne benchmark).

Comparaison à 3 voies par dimension n :
  1. Steihaug-CG (= CG, matvec creux uniquement)         -> jamais de fill-in
  2. Newton-LM, factorisation DENSE (numpy Cholesky)      -> fill total (O(n^2) stockage)
  3. Newton-LM, factorisation CREUSE (scipy splu, LU)     -> fill partiel mais réel

Sorties :
  sparse_example.csv         — résultats bruts (n, méthode, cpu_s, nnz, ratio)
  sparse_fillin_S9.pdf/.png  — figure double panneau (CPU vs n ; fill-ratio vs n)
  table_sparse.tex           — table LaTeX résumée pour §9.2

Bloc : B40 (sous-plan B38, PLAN_S9_SPARSE.md)
"""

import time
import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Constantes ────────────────────────────────────────────────────────────────
N_GRID    = [500, 1000, 2000, 5000]
BANDWIDTH = 2          # demi-largeur de bande -> pentadiagonale (5 diagonales)
N_REPEAT  = 3          # répétitions pour médiane robuste du temps CPU
SEED      = 990000     # graine dédiée §9 (distincte des graines G3/G4)
TOL_CG    = 1e-10

OUT_CSV  = "sparse_example.csv"
OUT_PDF  = "sparse_fillin_S9.pdf"
OUT_PNG  = "sparse_fillin_S9.png"
OUT_TEX  = "table_sparse.tex"


# ══════════════════════════════════════════════════════════════════════════════
# CONSTRUCTION DE LA HESSIENNE PENTADIAGONALE SPD
# ══════════════════════════════════════════════════════════════════════════════

def build_banded_spd(n: int, bandwidth: int = BANDWIDTH,
                     seed: int = SEED) -> sp.csr_matrix:
    """Hessienne pentadiagonale SPD par dominance diagonale stricte (Gershgorin).

    Diagonales off-diag tirées aléatoirement dans [-1, 1] ; diagonale principale
    fixée à 1.1 * somme des |off-diag| de la ligne + 1.0 -> dominance diagonale
    stricte -> SPD garanti (symétrique + dominance diagonale à diagonale positive).

    nnz(A) = O(bandwidth * n) = O(n) pour bandwidth fixe.
    """
    rng = np.random.RandomState(seed + n)   # graine dépendante de n pour variété

    diagonals = []
    offsets = []
    # Off-diagonales (symétriques : même valeur de part et d'autre)
    off_vals = {}
    for k in range(1, bandwidth + 1):
        vals = rng.uniform(-1.0, 1.0, size=n - k)
        off_vals[k] = vals
        diagonals.append(vals)
        offsets.append(k)
        diagonals.append(vals)
        offsets.append(-k)

    # Somme des |off-diag| par ligne pour la dominance diagonale
    row_abs_sum = np.zeros(n)
    for k in range(1, bandwidth + 1):
        v = np.abs(off_vals[k])
        row_abs_sum[:n - k] += v
        row_abs_sum[k:]     += v

    diag_vals = 1.1 * row_abs_sum + 1.0
    diagonals.insert(0, diag_vals)
    offsets.insert(0, 0)

    A = sp.diags(diagonals, offsets, shape=(n, n), format="csr")
    A.eliminate_zeros()
    return A


# ══════════════════════════════════════════════════════════════════════════════
# STEIHAUG-CG (= CG STANDARD ICI, Delta -> inf, A SPD GLOBALE)
# ══════════════════════════════════════════════════════════════════════════════

def steihaug_cg_sparse(A: sp.csr_matrix, b: np.ndarray,
                       tol: float = TOL_CG, max_iter: int | None = None):
    """CG résolvant A p = b, matvec creux uniquement — aucune factorisation.

    Équivalent à Steihaug-CG avec Delta->inf sur A SPD globale (pas de courbure
    négative possible, pas de troncature de frontière) : cf. test unitaire
    Phase 1 'Steihaug -> Newton (Delta->inf)'.

    Retourne (p, n_iter, n_matvec, cpu_s). n_matvec = n_iter (1 produit Hv/iter).
    """
    n = A.shape[0]
    max_iter = max_iter or n

    t0 = time.perf_counter()
    p = np.zeros(n)
    r = b.copy()
    d = r.copy()
    rs_old = r @ r
    b_norm = np.linalg.norm(b)
    n_iter = 0

    for k in range(max_iter):
        Ad = A @ d                      # ── seul accès à A : produit creux O(nnz) ──
        denom = d @ Ad
        if denom <= 0:                  # garde-fou (non atteint ici, A SPD globale)
            break
        alpha = rs_old / denom
        p += alpha * d
        r -= alpha * Ad
        n_iter = k + 1
        if np.linalg.norm(r) < tol * b_norm:
            break
        rs_new = r @ r
        beta = rs_new / rs_old
        d = r + beta * d
        rs_old = rs_new

    cpu_s = time.perf_counter() - t0
    return p, n_iter, n_iter, cpu_s   # n_matvec == n_iter (1 produit Hv par itération)


# ══════════════════════════════════════════════════════════════════════════════
# NEWTON-LM — FACTORISATION DENSE (numpy Cholesky)
# ══════════════════════════════════════════════════════════════════════════════

def newton_dense_factor(A: sp.csr_matrix, b: np.ndarray):
    """Facto + résolution via Cholesky dense (numpy). Mesure le coût de fill total :
    le facteur dense occupe n(n+1)/2 flottants, quelle que soit la sparsité de A.
    """
    n = A.shape[0]
    A_dense = A.toarray()

    t0 = time.perf_counter()
    L = np.linalg.cholesky(A_dense)
    y = np.linalg.solve(L, b)
    p = np.linalg.solve(L.T, y)
    cpu_s = time.perf_counter() - t0

    nnz_factor = n * (n + 1) // 2   # stockage structurel du triangle dense
    return p, cpu_s, nnz_factor


# ══════════════════════════════════════════════════════════════════════════════
# NEWTON-LM — FACTORISATION CREUSE (scipy splu, LU avec réordonnancement)
# ══════════════════════════════════════════════════════════════════════════════

def newton_sparse_factor(A: sp.csr_matrix, b: np.ndarray):
    """Facto + résolution via LU creuse (scipy.sparse.linalg.splu, SuperLU).

    splu effectue un réordonnancement (COLAMD par défaut) pour limiter le
    fill-in, mais ne l'élimine pas : c'est exactement le phénomène que la
    Proposition §9.1 qualifie de 'non bornable a priori'.
    """
    n = A.shape[0]
    A_csc = A.tocsc()

    t0 = time.perf_counter()
    lu = spla.splu(A_csc)
    p = lu.solve(b)
    cpu_s = time.perf_counter() - t0

    nnz_factor = lu.L.nnz + lu.U.nnz
    return p, cpu_s, nnz_factor


# ══════════════════════════════════════════════════════════════════════════════
# RUNNER — UNE CELLULE n
# ══════════════════════════════════════════════════════════════════════════════

def run_cell(n: int, verbose: bool = True) -> list:
    """Exécute les 3 méthodes sur une instance n, avec N_REPEAT répétitions
    (médiane retenue pour le temps CPU). Vérifie l'exactitude via ||A p - b||.
    """
    rng = np.random.RandomState(SEED + 7 * n)
    A = build_banded_spd(n)
    x0 = rng.randn(n)
    g0 = A @ x0              # gradient exact en x0 pour f(x)=1/2 x^T A x
    b = -g0                  # système Newton : A p = -g0  =>  solution exacte p* = -x0

    nnz_A = A.nnz

    rows = []

    # ── Steihaug-CG ──────────────────────────────────────────────────────────
    cpu_times, iters = [], []
    for _ in range(N_REPEAT):
        p_cg, n_iter, n_matvec, cpu_s = steihaug_cg_sparse(A, b)
        cpu_times.append(cpu_s)
        iters.append(n_iter)
    err_cg = np.linalg.norm(p_cg - (-x0)) / np.linalg.norm(x0)
    rows.append({
        "n": n, "method": "Steihaug-CG", "cpu_s": float(np.median(cpu_times)),
        "n_iter": int(np.median(iters)), "nnz_A": nnz_A,
        "nnz_factor": nnz_A,        # convention : matvec touche exactement nnz(A)
        "fill_ratio": 1.0,          # AUCUN fill-in par construction
        "rel_error": err_cg,
    })

    # ── Newton-LM dense ──────────────────────────────────────────────────────
    cpu_times = []
    for _ in range(N_REPEAT):
        p_dense, cpu_s, nnz_dense = newton_dense_factor(A, b)
        cpu_times.append(cpu_s)
    err_dense = np.linalg.norm(p_dense - (-x0)) / np.linalg.norm(x0)
    rows.append({
        "n": n, "method": "Newton-LM (dense)", "cpu_s": float(np.median(cpu_times)),
        "n_iter": 1, "nnz_A": nnz_A,
        "nnz_factor": nnz_dense,
        "fill_ratio": nnz_dense / nnz_A,
        "rel_error": err_dense,
    })

    # ── Newton-LM sparse (splu) ──────────────────────────────────────────────
    cpu_times = []
    for _ in range(N_REPEAT):
        p_sparse, cpu_s, nnz_sparse = newton_sparse_factor(A, b)
        cpu_times.append(cpu_s)
    err_sparse = np.linalg.norm(p_sparse - (-x0)) / np.linalg.norm(x0)
    rows.append({
        "n": n, "method": "Newton-LM (sparse LU)", "cpu_s": float(np.median(cpu_times)),
        "n_iter": 1, "nnz_A": nnz_A,
        "nnz_factor": nnz_sparse,
        "fill_ratio": nnz_sparse / nnz_A,
        "rel_error": err_sparse,
    })

    if verbose:
        print(f"\n  n={n:6d}  nnz(A)={nnz_A:8d}  (density={100*nnz_A/n**2:.3f}%)")
        for r in rows:
            print(f"    {r['method']:24s} cpu={r['cpu_s']:9.5f}s  "
                  f"nnz_factor={r['nnz_factor']:10d}  "
                  f"fill_ratio={r['fill_ratio']:8.2f}  "
                  f"rel_err={r['rel_error']:.2e}")

    return rows


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE — DOUBLE PANNEAU (S3=B)
# ══════════════════════════════════════════════════════════════════════════════

COLORS = {
    "Steihaug-CG":            "#1F77B4",
    "Newton-LM (dense)":      "#D62728",
    "Newton-LM (sparse LU)":  "#FF7F0E",
}
MARKERS = {
    "Steihaug-CG":            "s",
    "Newton-LM (dense)":      "D",
    "Newton-LM (sparse LU)":  "^",
}

plt.rcParams.update({
    "font.family":       "serif",
    "font.size":         10,
    "axes.titlesize":    11,
    "axes.labelsize":    10,
    "legend.fontsize":   8.5,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "lines.linewidth":   1.8,
    "lines.markersize":  7,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.dpi":        150,
})


def make_figure(df: pd.DataFrame, out_pdf: str = OUT_PDF, out_png: str = OUT_PNG) -> None:
    fig, (ax_cpu, ax_fill) = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.subplots_adjust(wspace=0.32, left=0.08, right=0.97, top=0.88, bottom=0.13)

    # ── Panneau gauche : CPU vs n ────────────────────────────────────────────
    for method in ["Steihaug-CG", "Newton-LM (sparse LU)", "Newton-LM (dense)"]:
        sub = df[df["method"] == method].sort_values("n")
        ax_cpu.plot(sub["n"], sub["cpu_s"],
                   color=COLORS[method], marker=MARKERS[method],
                   linestyle="-", label=method)

    ax_cpu.set_xscale("log")
    ax_cpu.set_yscale("log")
    ax_cpu.set_xlabel("Dimension $n$")
    ax_cpu.set_ylabel("CPU time (s)")
    ax_cpu.set_title("(a) Solve cost vs $n$")
    ax_cpu.set_xticks(N_GRID)
    ax_cpu.set_xticklabels([str(n) for n in N_GRID])
    ax_cpu.legend(loc="upper left", framealpha=0.92, edgecolor="0.8", fontsize=8)
    ax_cpu.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.5)
    ax_cpu.minorticks_off()

    # ── Panneau droit : fill ratio vs n ──────────────────────────────────────
    for method in ["Steihaug-CG", "Newton-LM (sparse LU)", "Newton-LM (dense)"]:
        sub = df[df["method"] == method].sort_values("n")
        ax_fill.plot(sub["n"], sub["fill_ratio"],
                    color=COLORS[method], marker=MARKERS[method],
                    linestyle="-", label=method)

    ax_fill.axhline(y=1.0, color=COLORS["Steihaug-CG"], linestyle=":",
                    linewidth=1.0, alpha=0.5)
    ax_fill.set_xscale("log")
    ax_fill.set_yscale("log")
    ax_fill.set_xlabel("Dimension $n$")
    ax_fill.set_ylabel(r"Fill ratio  $\mathrm{nnz(factor)} / \mathrm{nnz}(A)$")
    ax_fill.set_title("(b) Fill-in vs $n$")
    ax_fill.set_xticks(N_GRID)
    ax_fill.set_xticklabels([str(n) for n in N_GRID])
    ax_fill.legend(loc="upper left", framealpha=0.92, edgecolor="0.8", fontsize=8)
    ax_fill.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.5)
    ax_fill.minorticks_off()
    ax_fill.annotate("Steihaug-CG: no factorization,\nfill ratio = 1 by construction",
                     xy=(N_GRID[0], 1.0), xytext=(N_GRID[0]*1.3, 1.6),
                     fontsize=7.5, color=COLORS["Steihaug-CG"],
                     arrowprops=dict(arrowstyle="->", color=COLORS["Steihaug-CG"],
                                     lw=0.8))

    fig.suptitle("Sparse Hessian illustration: pentadiagonal SPD quadratic "
                 f"(bandwidth $b={BANDWIDTH}$)",
                 fontsize=10.5, fontweight="bold", y=0.98)

    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight", dpi=200)
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# TABLE LATEX
# ══════════════════════════════════════════════════════════════════════════════

def make_table(df: pd.DataFrame, out_tex: str = OUT_TEX) -> None:
    lines = []
    lines.append("% Table §9.2 — Sparse Hessian illustration (B40, PLAN_S9_SPARSE.md)")
    lines.append(r"\begin{table}[ht]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{l r r r r}")
    lines.append(r"\toprule")
    lines.append(r"Method & $n$ & CPU (s) & nnz(factor) & Fill ratio \\")
    lines.append(r"\midrule")

    show_n = [N_GRID[0], N_GRID[-1]]   # extrêmes de la grille pour la table
    method_order = ["Steihaug-CG", "Newton-LM (sparse LU)", "Newton-LM (dense)"]

    for n in show_n:
        for method in method_order:
            row = df[(df["n"] == n) & (df["method"] == method)].iloc[0]
            lines.append(
                f"  {method} & {n} & {row['cpu_s']:.4f} & "
                f"{int(row['nnz_factor'])} & {row['fill_ratio']:.2f} \\\\"
            )
        if n != show_n[-1]:
            lines.append(r"\addlinespace")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\caption{Sparse Hessian illustration: pentadiagonal SPD quadratic, "
                 f"bandwidth $b={BANDWIDTH}$, $\\mathrm{{nnz}}(A) = O(n)$. "
                 r"Steihaug--CG accesses $A$ only through matrix--vector products "
                 r"and never factorizes it (fill ratio $=1$ by construction); "
                 r"Newton-type methods incur fill-in that grows with $n$ regardless "
                 r"of fill-reducing ordering.}")
    lines.append(r"\label{tab:sparse}")
    lines.append(r"\end{table}")

    with open(out_tex, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# ══════════════════════════════════════════════════════════════════════════════
# GO / NO-GO (PLAN_S9_SPARSE.md §5)
# ══════════════════════════════════════════════════════════════════════════════

def print_go_nogo(df: pd.DataFrame) -> None:
    print("\n" + "=" * 65)
    print("GO / NO-GO — PLAN_S9_SPARSE.md §5")
    print("=" * 65)

    # Critère 1 : séparation CPU divergente (ratio croissant avec n) ?
    cg   = df[df["method"] == "Steihaug-CG"].sort_values("n")
    dns  = df[df["method"] == "Newton-LM (dense)"].sort_values("n")
    ratio_first = dns["cpu_s"].values[0] / cg["cpu_s"].values[0]
    ratio_last  = dns["cpu_s"].values[-1] / cg["cpu_s"].values[-1]
    diverging = ratio_last > 1.5 * ratio_first

    print(f"\nC1. Séparation CPU Steihaug-CG vs Newton-LM(dense) :")
    print(f"    n={N_GRID[0]:5d} : ratio = {ratio_first:8.1f}x")
    print(f"    n={N_GRID[-1]:5d} : ratio = {ratio_last:8.1f}x")
    print(f"    -> {'DIVERGENTE (GO)' if diverging else 'CONSTANTE (reformuler en NO-GO)'}")

    # Critère 2 : fill-in mesurable et croissant ?
    fill_first = dns["fill_ratio"].values[0]
    fill_last  = dns["fill_ratio"].values[-1]
    fill_growing = fill_last > fill_first

    print(f"\nC2. Fill ratio Newton-LM(dense) :")
    print(f"    n={N_GRID[0]:5d} : {fill_first:8.1f}")
    print(f"    n={N_GRID[-1]:5d} : {fill_last:8.1f}")
    print(f"    -> {'CROISSANT (GO)' if fill_growing else 'PLAT (NO-GO)'}")

    # Critère 3 : convergence/exactitude
    max_err = df["rel_error"].max()
    print(f"\nC3. Erreur relative max (toutes méthodes) : {max_err:.2e}")
    print(f"    -> {'OK (< 1e-6)' if max_err < 1e-6 else 'ATTENTION — vérifier'}")

    verdict = "GO" if (diverging and fill_growing and max_err < 1e-6) else "NO-GO / reformulation requise"
    print(f"\n{'='*65}")
    print(f"VERDICT : {verdict}")
    print(f"{'='*65}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 65)
    print("B40 — Sparse Hessian illustration (§9, G5)")
    print(f"Grille n : {N_GRID}  |  bandwidth={BANDWIDTH}  |  repeat={N_REPEAT}")
    print("=" * 65)

    all_rows = []
    for n in N_GRID:
        all_rows.extend(run_cell(n))

    df = pd.DataFrame(all_rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nRésultats écrits : {OUT_CSV}  ({len(df)} lignes)")

    make_figure(df)
    print(f"Figure écrite     : {OUT_PDF}  |  {OUT_PNG}")

    make_table(df)
    print(f"Table écrite      : {OUT_TEX}")

    print_go_nogo(df)


if __name__ == "__main__":
    main()
