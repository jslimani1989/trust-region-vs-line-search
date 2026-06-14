# -*- coding: utf-8 -*-
"""
test_steihaug_newton.py
------------------------------------------------------------
TEST UNITAIRE — Steihaug-CG → pas Newton  (Gap G6 / Bloc B06)

Propriété testée (Steihaug 1983, Théorème 3.1) :
  Si B ≻ 0 et Δ → ∞, le pas Steihaug converge vers −B⁻¹g.

Stratégie : quadratique SPD à minimiseur connu.
  A  = diag(2, 3)
  f(x) = ½ xᵀAx        →  x* = [0, 0]
  g(x) = Ax
  h(x) = A  (constante)
  x₀   = [1.0, 1.0]

Pas Newton depuis x₀ :
  p_N = −A⁻¹ g(x₀) = −A⁻¹(A x₀) = −x₀ = [−1, −1]
  → x₀ + p_N = [0, 0] = x*   (convergence en 1 itération externe)

Appel : trust_region_steihaug(..., delta0=1e8, tol=1e-10)

Vérifications :
  1. result.converged == True
  2. ‖result.x‖₂ < 1e-8          (atterrissage sur x*)
  3. result.iterations ≤ 2        (pas Newton pris dès la 1ʳᵉ itération)
"""

import sys
import os
import numpy as np

# ── chemin vers trust_region.py ───────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from trust_region import trust_region_steihaug


# ─── Problème test ────────────────────────────────────────────────────────────

def _make_quadratic(A: np.ndarray):
    """Génère (f, g, h) pour f(x) = ½ xᵀAx."""
    def f(x): return 0.5 * float(x @ A @ x)
    def g(x): return A @ x
    def h(x): return A.copy()
    return f, g, h


# ─── Cas de test ──────────────────────────────────────────────────────────────

def test_steihaug_newton_step():
    """
    Δ₀ = 1e8 (≫ ‖p_N‖ = √2 ≈ 1.41) :
    le pas de Steihaug doit coïncider avec le pas Newton,
    amenant x₀ = [1, 1] en x* = [0, 0] en ≤ 2 itérations.
    """
    A  = np.diag([2.0, 3.0])
    x0 = np.array([1.0, 1.0])
    f, g, h = _make_quadratic(A)

    # cg_tol_factor=0.0 : désactive Eisenstat-Walker → CG interne tourne
    # jusqu'à résidu nul → converge vers le pas Newton exact en n=2 pas
    # internes → 1 itération externe suffit.
    # (avec les paramètres par défaut cg_tol_factor=0.5, le CG s'arrête
    # tôt par Eisenstat-Walker et il faut ~4 outer iterations — comportement
    # correct en pratique, mais hors scope de ce test théorique.)
    result = trust_region_steihaug(
        f, g, h, x0,
        delta0        = 1e8,
        tol           = 1e-10,
        max_iter      = 100,
        cg_tol_factor = 0.0,   # CG exact → pas Newton en 1 outer iteration
    )

    # 1 — convergence déclarée
    assert result.converged, (
        f"Steihaug non convergé (grad_norm={result.grad_norm:.2e})"
    )

    # 2 — solution correcte
    err_x = np.linalg.norm(result.x)
    assert err_x < 1e-8, (
        f"‖x_final‖ = {err_x:.2e} ≥ 1e-8  (x* = [0, 0] non atteint)"
    )

    # 3 — pas Newton en ≤ 2 outer iterations (CG exact, n=2 pas internes)
    assert result.iterations <= 2, (
        f"iterations = {result.iterations} > 2  "
        f"(CG exact sur quadratique 2D doit converger en 1 outer iteration)"
    )

    return err_x, result.iterations, result.grad_norm


# ─── Runner ───────────────────────────────────────────────────────────────────

TESTS = [
    ("Steihaug → Newton  (Δ=1e8, A=diag(2,3), x₀=[1,1])",
     test_steihaug_newton_step),
]


def run() -> bool:
    print("=" * 60)
    print("TEST UNITAIRE — Steihaug-CG → pas Newton  (G6 / B06)")
    print("  f(x) = ½ xᵀAx  |  A = diag(2,3)  |  x₀ = [1, 1]")
    print("  Δ₀ = 1e8  →  p_Newton = [−1, −1]  →  x* = [0, 0]")
    print("=" * 60)

    passed = 0
    for name, fn in TESTS:
        try:
            err_x, iters, gnorm = fn()
            print(f"  ✓  {name}")
            print(f"       ‖x_final‖ = {err_x:.2e}"
                  f"  |  ‖∇f‖_final = {gnorm:.2e}"
                  f"  |  itérations = {iters}")
            passed += 1
        except AssertionError as e:
            print(f"  ✗  {name}")
            print(f"       {e}")
        except Exception as e:
            print(f"  ✗  {name}  [ERREUR : {type(e).__name__}: {e}]")

    print("-" * 60)
    print(f"  Résultat : {passed}/{len(TESTS)} tests passés.")
    print("=" * 60)
    return passed == len(TESTS)


if __name__ == "__main__":
    ok = run()
    sys.exit(0 if ok else 1)
