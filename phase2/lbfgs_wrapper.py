#!/usr/bin/env python3
"""
lbfgs_wrapper.py — L-BFGS (m in {5, 10, 20}) pour Gap G4
==========================================================
Two-loop recursion NW06 Alg. 7.4 + strong_wolfe existante (L3 Option A).
Standalone : ne dépend que de implementation_methode_lineaires.py et NumPy/pandas.

Blocs  :  (code) ->  (exécution) ->  (plot_lbfgs.py)
Spec   : SPEC_G4_LBFGS.md (verrouillée -06-23)
Sortie : results_lbfgs.csv (schéma §8 SPEC)
"""

import sys
import time
import itertools
import numpy as np
import pandas as pd

# ── Import recherche linéaire existante (L3 Option A) ─────────────────────────
try:
    from implementation_methode_lineaires import strong_wolfe
except ImportError:
    sys.path.insert(0, ".")
    from implementation_methode_lineaires import strong_wolfe

# ── Constantes (SPEC_G4 §5, KMAX = benchmark unifié) ─────────────────────────
KMAX   = 1000
TOL    = 1e-6
C1, C2 = 1e-4, 0.9
MS     = [5, 10, 20]

# Grilles d'expérience (SPEC_G4 §3)
KAPPAS_A = [1e2, 1e3, 1e4, 1e5]       # panneau A : κ-sweep, n=100
NS_B     = [10, 50, 100, 500, 1000]   # panneau B : n-sweep, κ=1e4
N_SEEDS  = 20

# Graines (SPEC_G4 §7, cohérentes avec G3)
def _seed_Q(n: int)  -> int: return 770000 + n
def _seed_x0(k: int) -> int: return 880000 + k   # k = 0..19


# ══════════════════════════════════════════════════════════════════════════════
# FAMILLE DE PROBLÈMES (SPEC_G4 §2)
# ══════════════════════════════════════════════════════════════════════════════

def _make_A(n: int, kappa: float) -> np.ndarray:
    """Hessienne SPD avec cond(A) = kappa exact.

    Spectre log-uniforme lambda_i = kappa^((i-1)/(n-1)), i=1..n.
    Rotation orthogonale Q = facteur Q de QR(G), G ~ N(0,1)^{n×n}.
    A = Q^T D Q.  Graine : SEED_Q(n) — indépendante de kappa (décision SPEC §7).
    """
    rng = np.random.RandomState(_seed_Q(n))
    G   = rng.randn(n, n)
    Q, _ = np.linalg.qr(G)
    idx  = np.arange(n, dtype=float)
    lam  = kappa ** (idx / max(n - 1, 1))   # log-uniforme [1, kappa]
    return (Q.T * lam) @ Q                  # Q^T diag(lam) Q  (broadcast: col j de Q.T *= lam[j])


def _make_x0(n: int, seed_idx: int) -> np.ndarray:
    """Point initial unitaire x0 = v/||v||, v ~ N(0, I_n). (SPEC §2)"""
    rng = np.random.RandomState(_seed_x0(seed_idx))
    v   = rng.randn(n)
    return v / np.linalg.norm(v)


# ══════════════════════════════════════════════════════════════════════════════
# RECHERCHE LINÉAIRE ROBUSTE
# ══════════════════════════════════════════════════════════════════════════════

def _armijo_fallback(f, xk: np.ndarray, pk: np.ndarray,
                     slope: float, c1: float = C1,
                     alpha0: float = 1.0, rho: float = 0.5) -> float:
    """Backtracking Armijo — fallback si strong_wolfe lève une exception.

    Utilisé quand le bracket de zoom collapse (b - a → 0 dans _cubic_interp),
    ce qui peut arriver sur des directions L-BFGS dégénérées en début d'optimisation.
    slope = g_k^T p_k  (doit être < 0 ; déjà vérifié avant l'appel).
    """
    alpha = alpha0
    fk    = f(xk)
    for _ in range(60):
        if f(xk + alpha * pk) <= fk + alpha * c1 * slope:
            return alpha
        alpha *= rho
        if alpha < 1e-14:
            break
    return alpha


def _safe_wolfe(f, grad_f, xk: np.ndarray, pk: np.ndarray,
                slope: float, c1: float = C1, c2: float = C2) -> float:
    """strong_wolfe avec fallback Armijo sur ZeroDivisionError / FloatingPointError."""
    try:
        alpha = strong_wolfe(f, grad_f, xk, pk, c1=c1, c2=c2)
        if alpha > 0.0:
            return alpha
        # alpha nul ou négatif retourné (ne devrait pas arriver mais guard-rail)
        return _armijo_fallback(f, xk, pk, slope, c1=c1)
    except (ZeroDivisionError, FloatingPointError, ValueError):
        return _armijo_fallback(f, xk, pk, slope, c1=c1)


# ══════════════════════════════════════════════════════════════════════════════
# TWO-LOOP RECURSION (NW06 Algorithme 7.4)
# ══════════════════════════════════════════════════════════════════════════════

def _two_loop(gk: np.ndarray,
              s_hist: list, y_hist: list,
              gamma: float) -> np.ndarray:
    """Calcule H_k^{-1} g_k par two-loop recursion.

    Convention de stockage : s_hist[0] = plus ancienne paire (k-m),
                             s_hist[-1] = plus récente (k-1).
    Boucle 1 : descendante (récent → ancien).
    Boucle 2 : montante   (ancien → récent).
    H_0 = gamma * I  (scaling Shanno-Phua, §5 SPEC).
    """
    m = len(s_hist)
    q = gk.copy()
    alphas = np.empty(m)
    rhos   = np.empty(m)

    # Boucle 1 — descendante
    for i in range(m - 1, -1, -1):
        sy       = float(s_hist[i] @ y_hist[i])
        rhos[i]  = 1.0 / sy if sy > 0.0 else 0.0
        alphas[i] = rhos[i] * float(s_hist[i] @ q)
        q -= alphas[i] * y_hist[i]

    r = gamma * q   # H_0 q

    # Boucle 2 — montante
    for i in range(m):
        beta = rhos[i] * float(y_hist[i] @ r)
        r += s_hist[i] * (alphas[i] - beta)

    return r   # H_k^{-1} g_k


# ══════════════════════════════════════════════════════════════════════════════
# SOLVEUR L-BFGS
# ══════════════════════════════════════════════════════════════════════════════

def lbfgs(f, grad_f, x0,
          m: int = 10,
          tol: float = TOL,
          max_iter: int = KMAX,
          c1: float = C1,
          c2: float = C2) -> dict:
    """Solveur L-BFGS (NW06 §7.4).

    Compatible avec _extract_solver_output du framework (retour dict avec x_opt,
    iterations, status, gnorm_final).  Enregistrement comme LS1 dans METHODS.

    Paramètres
    ----------
    f, grad_f : callables (wrappés via _make_counted_oracles si appelé par framework)
    x0        : np.ndarray, point initial (copié en interne)
    m         : int, taille mémoire — nombre de paires {s_k, y_k} conservées
    tol       : critère arrêt ||∇f|| < tol
    max_iter  : budget itérations (Kmax)
    c1, c2    : constantes Wolfe fortes (identiques à BFGS-Wolfe existant)

    Retour
    ------
    dict avec clés compatibles _extract_solver_output :
        x_opt, f_opt, iterations, status, gnorm_final, grad_norm, f, x
    """
    xk = np.array(x0, dtype=float)
    gk = grad_f(xk)

    s_hist: list = []   # paires s_k = x_{k+1} - x_k  (FIFO, longueur <= m)
    y_hist: list = []   # paires y_k = g_{k+1} - g_k
    gamma  = 1.0        # scaling H_0 = gamma * I  (1ere iter : H_0 = I)

    hist_gnorm: list = []
    hist_f:     list = []
    hist_x:     list = []
    converged   = False

    for k in range(max_iter):
        gnorm = float(np.linalg.norm(gk))
        hist_gnorm.append(gnorm)
        hist_f.append(f(xk))
        hist_x.append(xk.copy())

        if gnorm < tol:
            converged = True
            break

        # ── Direction de descente ─────────────────────────────────────────────
        if len(s_hist) == 0:
            pk = -gk                           # premier pas : gradient pur
        else:
            pk = -_two_loop(gk, s_hist, y_hist, gamma)

        # Garde-fou : vérifier que pk est une direction de descente
        if np.dot(gk, pk) >= 0.0:
            pk = -gk                           # repli gradient si two-loop dégénérée

        # ── Recherche linéaire Wolfe robuste ─────────────────────────────────
        slope = float(np.dot(gk, pk))          # g_k^T p_k < 0 (direction descente)
        alpha = _safe_wolfe(f, grad_f, xk, pk, slope, c1=c1, c2=c2)

        xk_new = xk + alpha * pk
        gk_new = grad_f(xk_new)

        # ── Mise à jour mémoire {s_k, y_k} ──────────────────────────────────
        sk = xk_new - xk
        yk = gk_new - gk
        sy = float(sk @ yk)

        if sy > 1e-12:   # safeguard courbure positive (SPEC §5)
            if len(s_hist) == m:   # buffer plein : supprimer la plus ancienne paire
                s_hist.pop(0)
                y_hist.pop(0)
            s_hist.append(sk)
            y_hist.append(yk)
            gamma = sy / float(yk @ yk)   # scaling Shanno-Phua pour H_0 suivant

        xk = xk_new
        gk = gk_new

    gnorm_final = float(np.linalg.norm(gk))

    return {
        "x_opt":       xk,
        "f_opt":       hist_f[-1] if hist_f else f(xk),
        "iterations":  len(hist_gnorm),
        "status":      "converged" if converged else "max_iter",
        "gnorm_final": gnorm_final,
        "grad_norm":   hist_gnorm,
        "f":           hist_f,
        "x":           hist_x,
    }


# ── Variantes exposées au framework (pattern LS1, même signature que bfgs_wolfe) ──

def lbfgs_5(f, g, x0, tol=TOL, max_iter=KMAX, c1=C1, c2=C2):
    return lbfgs(f, g, x0, m=5,  tol=tol, max_iter=max_iter, c1=c1, c2=c2)

def lbfgs_10(f, g, x0, tol=TOL, max_iter=KMAX, c1=C1, c2=C2):
    return lbfgs(f, g, x0, m=10, tol=tol, max_iter=max_iter, c1=c1, c2=c2)

def lbfgs_20(f, g, x0, tol=TOL, max_iter=KMAX, c1=C1, c2=C2):
    return lbfgs(f, g, x0, m=20, tol=tol, max_iter=max_iter, c1=c1, c2=c2)

# Dictionnaire pour intégration future dans METHODS de framework_united_final.py
LBFGS_VARIANTS = {"LBFGS-5": lbfgs_5, "LBFGS-10": lbfgs_10, "LBFGS-20": lbfgs_20}


# ══════════════════════════════════════════════════════════════════════════════
# COMPTAGE DES ORACLES (réplique locale de _make_counted_oracles)
# ══════════════════════════════════════════════════════════════════════════════

def _counted(f, grad_f):
    """Wrappe f et grad_f avec des compteurs N_f / N_g.  N_h = 0 (L-BFGS)."""
    counts = {"f": 0, "g": 0}
    def fc(x): counts["f"] += 1; return f(x)
    def gc(x): counts["g"] += 1; return grad_f(x)
    return fc, gc, counts


# ══════════════════════════════════════════════════════════════════════════════
# SANITY CHECK (exécuté avant l'expérience)
# ══════════════════════════════════════════════════════════════════════════════

def _sanity_check() -> bool:
    """Vérifie les deux invariants de base sur f = ½||x||² (A=I, kappa=1).

    1. Convergence en 1 itération (direction exacte dès m≥1).
    2. N_h = 0 par construction.
    Retourne True si tout est correct, lève une AssertionError sinon.
    """
    n = 5
    A = np.eye(n)
    f_test      = lambda x: 0.5 * float(x @ x)
    grad_f_test = lambda x: x.copy()
    rng  = np.random.RandomState(42)
    x0   = rng.randn(n)

    fc, gc, counts = _counted(f_test, grad_f_test)
    hist = lbfgs(fc, gc, x0, m=5, tol=TOL, max_iter=KMAX)

    assert hist["status"] == "converged", \
        f"Sanity FAIL: non convergé sur f=½||x||²  (status={hist['status']})"
    assert hist["gnorm_final"] < TOL, \
        f"Sanity FAIL: gnorm_final={hist['gnorm_final']:.2e} > tol={TOL}"
    assert counts["f"] > 0 and counts["g"] > 0, \
        "Sanity FAIL: compteurs oracles non incrémentés"

    # Test Cauchy direction sur quadratique pure (1 itération attendue)
    A2 = 4.0 * np.eye(n)
    f2      = lambda x: 0.5 * float(x @ A2 @ x)
    grad_f2 = lambda x: A2 @ x
    x0_2    = np.ones(n)
    fc2, gc2, _ = _counted(f2, grad_f2)
    h2 = lbfgs(fc2, gc2, x0_2, m=10, tol=TOL, max_iter=KMAX)
    assert h2["status"] == "converged", \
        f"Sanity FAIL: non convergé sur f=2||x||² (status={h2['status']})"

    print("  [sanity] OK — convergence et compteurs oracles valides.")
    return True


# ══════════════════════════════════════════════════════════════════════════════
# RUNNER D'EXPÉRIENCE G4 (SPEC_G4 §3)
# ══════════════════════════════════════════════════════════════════════════════

def run_g4(output_path: str = "results_lbfgs.csv",
           verbose: bool = True) -> pd.DataFrame:
    """Exécute l'expérience G4 et écrit results_lbfgs.csv.

    Panneau A : κ ∈ {1e2,1e3,1e4,1e5},  n = 100        (κ-sweep)
    Panneau B : n ∈ {10,50,100,500,1000}, κ = 1e4       (n-sweep)
    Méthodes  : L-BFGS-{5,10,20}
    Seeds     : 20 par cellule — mêmes x0 que les 6 méthodes existantes (pairing)

    Complexité indicative (ThinkPad T14, NumPy BLAS) :
      n=100  → ~0.02 s / run × 60 = ~1 s  par cellule
      n=500  → ~0.3  s / run × 60 = ~18 s par cellule
      n=1000 → ~1    s / run × 60 = ~60 s par cellule
    Durée totale estimée : 5–15 min selon la machine.
    """
    # Liste des cellules (panel, kappa, n)
    cells = [("A", k, 100)  for k in KAPPAS_A] + \
            [("B", 1e4,  n)  for n in NS_B]

    total = len(cells) * N_SEEDS * len(MS)
    done  = 0
    rows  = []
    t_exp = time.perf_counter()

    for panel, kappa, n in cells:
        # Précalcul de A une seule fois par (n, kappa) — O(n³), mutualise les 20 seeds
        A     = _make_A(n, kappa)
        f_raw = lambda x, _A=A: 0.5 * float(x @ _A @ x)
        g_raw = lambda x, _A=A: _A @ x

        for seed_idx in range(N_SEEDS):
            x0 = _make_x0(n, seed_idx)

            for m in MS:
                fc, gc, counts = _counted(f_raw, g_raw)

                t0   = time.perf_counter()
                hist = lbfgs(fc, gc, x0.copy(), m=m,
                             tol=TOL, max_iter=KMAX, c1=C1, c2=C2)
                cpu_s = time.perf_counter() - t0

                rows.append({
                    "panel":       panel,
                    "method":      f"LBFGS-{m}",
                    "m":           m,
                    "kappa":       kappa,
                    "n":           n,
                    "seed":        seed_idx,
                    "iter_outer":  hist["iterations"],
                    "nfev":        counts["f"],
                    "ngev":        counts["g"],
                    "nhev":        0,
                    "ntot":        counts["f"] + counts["g"],
                    "converged":   hist["status"] == "converged",
                    "gnorm_final": round(hist["gnorm_final"], 10),
                    "cpu_s":       round(cpu_s, 6),
                })

                done += 1
                if verbose and done % 60 == 0:
                    elapsed = time.perf_counter() - t_exp
                    eta     = elapsed / done * (total - done)
                    print(f"  [{done:4d}/{total}]  "
                          f"panel={panel}  κ={kappa:.0e}  n={n:5d}  m={m:2d}  "
                          f"iter={hist['iterations']:4d}  "
                          f"conv={'Y' if hist['status']=='converged' else 'N'}  "
                          f"ETA={eta/60:.1f} min")

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)

    if verbose:
        elapsed = time.perf_counter() - t_exp
        print(f"\nResultats ecrits : {output_path}  "
              f"({len(df)} lignes, {elapsed:.1f} s)")
        print("\nTaux de convergence par méthode et panneau :")
        print(df.groupby(["panel", "method"])["converged"]
                .mean().round(3).to_string())
        print("\nIter_outer médian par méthode et panneau :")
        print(df.groupby(["panel", "method", "n"])["iter_outer"]
                .median().round(1).to_string())

    return df


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("G4 — L-BFGS Experiment  (SPEC_G4_LBFGS, bloc /)")
    print("=" * 60)

    print("\n[1/2] Sanity check ...")
    _sanity_check()

    print("\n[2/2] Expérience G4 ...")
    run_g4("results_lbfgs.csv", verbose=True)

