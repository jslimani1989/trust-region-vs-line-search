
# -*- coding: utf-8 -*-
import numpy as np
from scipy.linalg import solve_triangular  # 009d : substitutions triangulaires (Newton-LM)

# ─────────────────────────────────────────────
# Recherche linéaire : Backtracking Armijo
# ─────────────────────────────────────────────
def backtracking_armijo(f, grad_f, xk, dk, alpha0=1.0, rho=0.5, c=1e-4):
    """Retourne (alpha, n_eval, ls_failed, n_nonfinite).

    ls_failed : True si le plancher alpha<1e-14 a ete atteint sans satisfaire
    la condition d'Armijo (B1, consigne 003 S1e).
    n_nonfinite : nombre d'essais dont f(x+alpha*d) n'est pas fini ; un tel
    essai compte comme un rejet et force la poursuite du backtracking (B3).
    Sequence d'alpha et nombre d'evaluations : strictement identiques a la
    version v0.1 pour tout essai fini (aucune modification de la logique).
    """
    alpha = alpha0
    fk = f(xk)
    slope = c * np.dot(grad_f(xk), dk)
    n_eval = 0
    n_nonfinite = 0
    ls_failed = False
    while True:
        ftrial = f(xk + alpha * dk)
        if np.isfinite(ftrial) and ftrial <= fk + alpha * slope:
            break
        if not np.isfinite(ftrial):
            n_nonfinite += 1
        alpha *= rho
        n_eval += 1
        if alpha < 1e-14:
            ls_failed = True
            break
    return alpha, n_eval, ls_failed, n_nonfinite


def bfgs(f, grad_f, x0, tol=1e-6, max_iter=500, alpha0=1.0, rho=0.5, c=1e-4):
    n = len(x0)
    xk = np.array(x0, dtype=float)
    Hk = np.eye(n)
    gk = grad_f(xk)
    history = {'x': [], 'f': [], 'grad_norm': [], 'alpha': []}
    last_ls_failed = False
    n_nonfinite_total = 0

    for _ in range(max_iter):
        history['x'].append(xk.copy())
        history['f'].append(f(xk))
        history['grad_norm'].append(np.linalg.norm(gk))
        if np.linalg.norm(gk) < tol:
            history['status'] = 'converged'
            break
        dk = -Hk @ gk
        alpha, _, last_ls_failed, n_nf = backtracking_armijo(f, grad_f, xk, dk, alpha0, rho, c)
        n_nonfinite_total += n_nf
        history['alpha'].append(alpha)
        xk_new = xk + alpha * dk
        gk_new = grad_f(xk_new)
        sk = xk_new - xk
        yk = gk_new - gk
        sy = sk @ yk
        if sy > 1e-12:
            rho_k = 1.0 / sy
            # 009d : developpement exact de (I - rho s y^T) H (I - rho y s^T) + rho s s^T pour H symetrique,
            # en O(n^2) (un produit matrice-vecteur et des produits exterieurs) au lieu de deux produits n x n.
            Hy = Hk @ yk
            yHy = float(yk @ Hy)
            Hk = Hk - rho_k * (np.outer(sk, Hy) + np.outer(Hy, sk)) + (rho_k**2 * yHy + rho_k) * np.outer(sk, sk)
        xk = xk_new
        gk = gk_new
    else:
        history['status'] = 'max_iter'

    # ── B1 : motif d'arret precis, sans toucher au critere d'arret ─────────
    if history['status'] == 'converged':
        history['stop_reason'] = 'converged'
        history['stagnation_num'] = False
    else:
        history['stop_reason'] = 'line_search_failure' if last_ls_failed else 'max_iter'
        history['stagnation_num'] = last_ls_failed and (history['grad_norm'][-1] <= 1e-4)
    history['n_nonfinite_rejected'] = n_nonfinite_total

    history['x_opt'] = xk
    history['f_opt'] = f(xk)
    history['iterations'] = len(history['f'])
    return history


def strong_wolfe(f, grad_f, xk, dk, alpha_max=10.0, c1=1e-4, c2=0.9, counters=None):
    """Retourne (alpha, wolfe_ok, hit_nonfinite).

    wolfe_ok : True si les conditions de Wolfe fortes sont effectivement
    satisfaites (B1). hit_nonfinite : True si un essai non fini a ete
    rencontre et rejete au cours de la recherche (B3). Choix d'alpha
    inchange pour tout essai fini (memes branches, meme ordre), SAUF le cas
    degenere corrige en 008b/1a (alpha deja a alpha_max sans encadrement,
    Nocedal & Wright, algorithme 3.5) : ce cas ne produisait auparavant
    qu'une exception (ZeroDivisionError, cf. rapport 008a), jamais un pas
    fini -- aucun essai auparavant reussi n'est donc affecte.

    counters (optionnel) : dict mutable {'alpha_max_reached', 'interp_guard_used'}
    incremente par cet appel et par _zoom/_cubic_interp (008b, 1a.3).
    """
    if counters is None:
        counters = {"alpha_max_reached": 0, "interp_guard_used": 0}
    phi = lambda a: f(xk + a * dk)
    dphi = lambda a: np.dot(grad_f(xk + a * dk), dk)
    phi0 = phi(0.0)
    dphi0 = dphi(0.0)
    alpha_prev, alpha_i = 0.0, 1.0
    phi_prev = phi0
    for i in range(20):
        if i > 0 and alpha_i == alpha_prev:
            # 008b, 1a.1 : alpha_i est deja au plafond alpha_max (le doublement
            # ci-dessous sature) et AUCUN encadrement n'a ete forme -- ne plus
            # jamais reevaluer phi au meme point ni transmettre a _zoom un
            # intervalle [alpha_prev, alpha_i] deja reduit a alpha_max (c'etait
            # la cause exacte des 12 exceptions ZeroDivisionError de v0.2,
            # rapport 008a). phi_prev vaut deja phi(alpha_max) (calcule a
            # l'iteration precedente) : pas de nouvelle evaluation de f.
            counters["alpha_max_reached"] += 1
            armijo_ok = np.isfinite(phi_prev) and phi_prev <= phi0 + c1 * alpha_i * dphi0
            if armijo_ok:
                return alpha_i, False, not np.isfinite(phi_prev)
            if alpha_prev < alpha_max:  # ne peut pas se produire ici (garde defensive, cf. rapport)
                return _zoom(phi, dphi, alpha_prev, alpha_max, phi_prev, phi_prev, phi0, dphi0, c1, c2, counters)
            return alpha_i, False, not np.isfinite(phi_prev)
        phi_i = phi(alpha_i)
        nf_i = not np.isfinite(phi_i)
        if nf_i or phi_i > phi0 + c1 * alpha_i * dphi0 or (phi_i >= phi_prev and i > 0):
            return _zoom(phi, dphi, alpha_prev, alpha_i, phi_prev, phi_i, phi0, dphi0, c1, c2, counters)
        dphi_i = dphi(alpha_i)
        if abs(dphi_i) <= -c2 * dphi0:
            return alpha_i, True, False
        if dphi_i >= 0:
            return _zoom(phi, dphi, alpha_i, alpha_prev, phi_i, phi_prev, phi0, dphi0, c1, c2, counters)
        alpha_prev = alpha_i
        phi_prev = phi_i
        alpha_i = min(2.0 * alpha_i, alpha_max)
    return alpha_i, False, False


def _zoom(phi, dphi, alo, ahi, phi_lo, phi_hi, phi0, dphi0, c1, c2, counters=None):
    if counters is None:
        counters = {"alpha_max_reached": 0, "interp_guard_used": 0}
    hit_nonfinite = (not np.isfinite(phi_lo)) or (not np.isfinite(phi_hi))
    for _ in range(30):
        alpha_j = _cubic_interp(alo, ahi, phi_lo, phi_hi, dphi(alo), dphi(ahi), counters)
        phi_j = phi(alpha_j)
        nf_j = not np.isfinite(phi_j)
        if nf_j:
            hit_nonfinite = True
        if nf_j or phi_j > phi0 + c1 * alpha_j * dphi0 or phi_j >= phi_lo:
            ahi = alpha_j
            phi_hi = phi_j
        else:
            dphi_j = dphi(alpha_j)
            if abs(dphi_j) <= -c2 * dphi0:
                return alpha_j, True, hit_nonfinite
            if dphi_j * (ahi - alo) >= 0:
                ahi = alo
                phi_hi = phi_lo
            alo = alpha_j
            phi_lo = phi_j
        if abs(ahi - alo) < 1e-14:
            break
    return alo, False, hit_nonfinite


def _cubic_interp(a, b, fa, fb, dfa, dfb, counters=None):
    # 008b, 1a.2 : garde-fou avant toute division -- intervalle deja reduit a
    # un point (ou numeriquement indistinguable). N'affecte aucun cas ou
    # |b-a| est significatif (meme resultat qu'avant, au bit pres).
    if counters is None:
        counters = {"alpha_max_reached": 0, "interp_guard_used": 0}
    if abs(b - a) <= 1e-14 * max(1.0, abs(a), abs(b)):
        counters["interp_guard_used"] += 1
        return (a + b) / 2.0
    d1 = dfa + dfb - 3.0 * (fb - fa) / (b - a)
    disc = d1**2 - dfa * dfb
    if disc < 0:
        return (a + b) / 2.0
    d2 = np.sqrt(disc)
    alpha = b - (b - a) * (dfb + d2 - d1) / (dfb - dfa + 2.0 * d2)
    return float(np.clip(alpha, min(a, b), max(a, b)))


def bfgs_wolfe(f, grad_f, x0, tol=1e-6, max_iter=500, c1=1e-4, c2=0.9):
    n = len(x0)
    xk = np.array(x0, dtype=float)
    Hk = np.eye(n)
    gk = grad_f(xk)
    history = {'x': [], 'f': [], 'grad_norm': [], 'alpha': []}
    last_ls_failed = False
    n_nonfinite_total = 0
    counters = {"alpha_max_reached": 0, "interp_guard_used": 0}  # 008b, 1a.3

    for _ in range(max_iter):
        history['x'].append(xk.copy())
        history['f'].append(f(xk))
        history['grad_norm'].append(np.linalg.norm(gk))
        if np.linalg.norm(gk) < tol:
            history['status'] = 'converged'
            break
        dk = -Hk @ gk
        alpha, wolfe_ok, hit_nf = strong_wolfe(f, grad_f, xk, dk, c1=c1, c2=c2, counters=counters)
        last_ls_failed = not wolfe_ok
        n_nonfinite_total += int(hit_nf)
        history['alpha'].append(alpha)
        xk_new = xk + alpha * dk
        gk_new = grad_f(xk_new)
        sk = xk_new - xk
        yk = gk_new - gk
        sy = sk @ yk
        if sy > 1e-12:
            rho_k = 1.0 / sy
            # 009d : meme mise a jour en O(n^2) que dans bfgs (developpement exact, H symetrique).
            Hy = Hk @ yk
            yHy = float(yk @ Hy)
            Hk = Hk - rho_k * (np.outer(sk, Hy) + np.outer(Hy, sk)) + (rho_k**2 * yHy + rho_k) * np.outer(sk, sk)
        xk = xk_new
        gk = gk_new
    else:
        history['status'] = 'max_iter'

    if history['status'] == 'converged':
        history['stop_reason'] = 'converged'
        history['stagnation_num'] = False
    else:
        history['stop_reason'] = 'line_search_failure' if last_ls_failed else 'max_iter'
        history['stagnation_num'] = last_ls_failed and (history['grad_norm'][-1] <= 1e-4)
    history['n_nonfinite_rejected'] = n_nonfinite_total
    history['alpha_max_reached'] = counters["alpha_max_reached"]      # 008b, 1a.3
    history['interp_guard_used'] = counters["interp_guard_used"]      # 008b, 1a.3

    history['x_opt'] = xk
    history['f_opt'] = f(xk)
    history['iterations'] = len(history['f'])
    return history


def backtracking_armijo_newton(f, grad_f, xk, dk, alpha0=1.0, rho=0.5, c=1e-4):
    """Retourne (alpha, ls_failed, n_nonfinite) — memes garanties que
    backtracking_armijo (B1/B3), meme sequence d'alpha pour tout essai fini."""
    alpha = alpha0
    fk = f(xk)
    slope = c * np.dot(grad_f(xk), dk)
    n_nonfinite = 0
    ls_failed = False
    while True:
        ftrial = f(xk + alpha * dk)
        if np.isfinite(ftrial) and ftrial <= fk + alpha * slope:
            break
        if not np.isfinite(ftrial):
            n_nonfinite += 1
        alpha *= rho
        if alpha < 1e-14:
            ls_failed = True
            break
    return alpha, ls_failed, n_nonfinite


def newton_backtracking(f, grad_f, hess_f, x0, tol=1e-6, max_iter=100,
                         alpha0=1.0, rho=0.5, c=1e-4, lm_base=1e-3, instrument=True):
    xk = np.array(x0, dtype=float)
    n  = len(xk)
    In = np.eye(n)                       # identite, reutilisee a chaque iteration (patch G1)
    history = {'x': [], 'f': [], 'grad_norm': [], 'alpha': []}
    lm_trace = []                        # B2 : trace separee par execution (ecrite par le driver)
    last_ls_failed = False
    n_nonfinite_total = 0
    for it in range(max_iter):
        gk = grad_f(xk)
        Hk = hess_f(xk)
        gnorm = np.linalg.norm(gk)       # calcule une fois, reutilise pour tol et lambda0
        history['x'].append(xk.copy())
        history['f'].append(f(xk))
        history['grad_norm'].append(gnorm)
        if gnorm < tol:
            history['status'] = 'converged'
            break
        # ── Regularisation Levenberg-Marquardt (patch G1) ────────────────────
        # Remplace : solve + fallback -gk
        # Garantit une direction de descente meme si Hk est indefinie (Cat. C)
        lam = gnorm * lm_base            # lambda0 adaptatif : scale au gradient
        dk  = None
        doubling_reasons = []            # B2 : motif de chaque doublement
        for _attempt in range(30):       # max 2^30 ~ 1e9 x lambda0
            try:
                L     = np.linalg.cholesky(Hk + lam * In)
                # 009d : deux substitutions triangulaires O(n^2) (L est triangulaire inferieure) au lieu de
                # deux resolutions par factorisation LU generale ; Cholesky inchangee, LinAlgError inchangee.
                dk_try = solve_triangular(L.T, solve_triangular(L, -gk, lower=True, check_finite=False),
                                          lower=False, check_finite=False)
                if np.dot(gk, dk_try) < 0:   # direction de descente verifiee
                    dk = dk_try
                    break
                else:
                    doubling_reasons.append('non_descent')
            except np.linalg.LinAlgError:
                doubling_reasons.append('cholesky_fail')  # Hk + lam*I non SPD → doubler lam
            lam *= 2.0
        if dk is None:                   # garde-fou ultime (quasi-impossible en pratique)
            dk = -gk
        # B2 : lambda_min(H) a cette iteration (calcule sur le Hk deja evalue, aucun cout f/g/h supplementaire).
        # 008e : decomposition spectrale dense, coute plusieurs fois une factorisation de Cholesky -- desactivee
        # si instrument=False (aucun iteree n'en depend, cf. rapport 008e).
        if instrument:
            lam_min_H = float(np.linalg.eigvalsh(Hk).min())
            lm_trace.append({
                'iter': it, 'lambda_final': lam, 'n_doublings': len(doubling_reasons),
                'reasons': ';'.join(doubling_reasons), 'lambda_min_H': lam_min_H,
            })
        alpha, last_ls_failed, n_nf = backtracking_armijo_newton(f, grad_f, xk, dk, alpha0, rho, c)
        n_nonfinite_total += n_nf
        history['alpha'].append(alpha)
        xk = xk + alpha * dk
    else:
        history['status'] = 'max_iter'

    if history['status'] == 'converged':
        history['stop_reason'] = 'converged'
        history['stagnation_num'] = False
    else:
        history['stop_reason'] = 'line_search_failure' if last_ls_failed else 'max_iter'
        history['stagnation_num'] = last_ls_failed and (history['grad_norm'][-1] <= 1e-4)
    history['n_nonfinite_rejected'] = n_nonfinite_total
    history['lm_trace'] = lm_trace

    history['x_opt'] = xk
    history['f_opt'] = f(xk)
    history['iterations'] = len(history['f'])
    return history
