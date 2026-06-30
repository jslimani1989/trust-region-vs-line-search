# -*- coding: utf-8 -*-
"""
test_cauchy_point.py
------------------------------------------------------------
TEST UNITAIRE — Cauchy Point  (Gap G6 / Bloc B05)

Propriété testée (Nocedal & Wright, Algorithme 4.2) :
  f(x) = ½‖x‖²,  g = x,  B = I
  x₀ = [3, 4],  ‖g‖ = 5

  Sous-cas A (‖g‖ < Δ) : Δ = 10  →  p = −g        (pas Newton exact)
  Sous-cas B (‖g‖ > Δ) : Δ =  2  →  p = −(Δ/‖g‖)g  (frontière)

Tolérance : ‖p − p_exact‖₂ < 1e-10
"""

import sys
import os
import numpy as np

# ── chemin vers trust_region.py ───────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from trust_region import _cauchy_point


# ─── Cas de test ──────────────────────────────────────────────────────────────

def test_cauchy_newton_step() -> float:
    """
    Sous-cas A : ‖g‖ = 5 < Δ = 10
    τ* = min(1, ‖g‖/Δ) = min(1, 0.5) = 0.5
    p  = −τ* · (Δ/‖g‖) · g = −0.5 · 2 · [3,4] = −[3, 4]
    → contrainte inactive, pas de Newton exact.
    """
    g       = np.array([3.0, 4.0])
    B       = np.eye(2)
    delta   = 10.0
    p_exact = np.array([-3.0, -4.0])

    p   = _cauchy_point(g, B, delta)
    err = np.linalg.norm(p - p_exact)
    assert err < 1e-10, f"Sous-cas A : err = {err:.2e} ≥ 1e-10"
    return err


def test_cauchy_boundary() -> float:
    """
    Sous-cas B : ‖g‖ = 5 > Δ = 2
    τ* = min(1, ‖g‖/Δ) = min(1, 2.5) = 1
    p  = −1 · (Δ/‖g‖) · g = −(2/5)·[3, 4] = [−1.2, −1.6]
    → contrainte active, p sur la frontière de la région.
    """
    g       = np.array([3.0, 4.0])
    B       = np.eye(2)
    delta   = 2.0
    gnorm   = np.linalg.norm(g)
    p_exact = -(delta / gnorm) * g   # = [−1.2, −1.6]

    p   = _cauchy_point(g, B, delta)
    err = np.linalg.norm(p - p_exact)
    assert err < 1e-10, f"Sous-cas B : err = {err:.2e} ≥ 1e-10"
    return err


# ─── Runner ───────────────────────────────────────────────────────────────────

TESTS = [
    ("A — Pas Newton (Δ=10, ‖g‖<Δ)",    test_cauchy_newton_step),
    ("B — Frontière  (Δ= 2, ‖g‖>Δ)",    test_cauchy_boundary),
]


def run() -> bool:
    print("=" * 60)
    print("TEST UNITAIRE — Cauchy Point  (G6 / B05)")
    print("  f(x) = ½‖x‖²  |  B = I  |  x₀ = [3, 4]  |  ‖g‖ = 5")
    print("=" * 60)

    passed = 0
    for name, fn in TESTS:
        try:
            err = fn()
            print(f"  ✓  {name}  [err = {err:.2e}]")
            passed += 1
        except AssertionError as e:
            print(f"  ✗  {name}  [{e}]")
        except Exception as e:
            print(f"  ✗  {name}  [ERREUR : {type(e).__name__}: {e}]")

    print("-" * 60)
    print(f"  Résultat : {passed}/{len(TESTS)} tests passés.")
    print("=" * 60)
    return passed == len(TESTS)


if __name__ == "__main__":
    ok = run()
    sys.exit(0 if ok else 1)
