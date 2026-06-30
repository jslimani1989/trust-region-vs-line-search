
# -*- coding: utf-8 -*-
import numpy as np

# ─────────────────────────────────────────────
# Recherche linéaire : Backtracking Armijo
# ─────────────────────────────────────────────
def backtracking_armijo(f, grad_f, xk, dk, alpha0=1.0, rho=0.5, c=1e-4):
    alpha = alpha0
    fk = f(xk)
    slope = c * np.dot(grad_f(xk), dk)
    n_eval = 0
    while f(xk + alpha * dk) > fk + alpha * slope:
        alpha *= rho
        n_eval += 1
        if alpha < 1e-14:
            break
    return alpha, n_eval


def bfgs(f, grad_f, x0, tol=1e-6, max_iter=500, alpha0=1.0, rho=0.5, c=1e-4):
    n = len(x0)
    xk = np.array(x0, dtype=float)
    Hk = np.eye(n)
    gk = grad_f(xk)
    history = {'x': [], 'f': [], 'grad_norm': [], 'alpha': []}

    for _ in range(max_iter):
        history['x'].append(xk.copy())
        history['f'].append(f(xk))
        history['grad_norm'].append(np.linalg.norm(gk))
        if np.linalg.norm(gk) < tol:
            history['status'] = 'converged'
            break
        dk = -Hk @ gk
        alpha, _ = backtracking_armijo(f, grad_f, xk, dk, alpha0, rho, c)
        history['alpha'].append(alpha)
        xk_new = xk + alpha * dk
        gk_new = grad_f(xk_new)
        sk = xk_new - xk
        yk = gk_new - gk
        sy = sk @ yk
        if sy > 1e-12:
            rho_k = 1.0 / sy
            I = np.eye(n)
            A = I - rho_k * np.outer(sk, yk)
            B = I - rho_k * np.outer(yk, sk)
            Hk = A @ Hk @ B + rho_k * np.outer(sk, sk)
        xk = xk_new
        gk = gk_new
    else:
        history['status'] = 'max_iter'

    history['x_opt'] = xk
    history['f_opt'] = f(xk)
    history['iterations'] = len(history['f'])
    return history


def strong_wolfe(f, grad_f, xk, dk, alpha_max=10.0, c1=1e-4, c2=0.9):
    phi = lambda a: f(xk + a * dk)
    dphi = lambda a: np.dot(grad_f(xk + a * dk), dk)
    phi0 = phi(0.0)
    dphi0 = dphi(0.0)
    alpha_prev, alpha_i = 0.0, 1.0
    phi_prev = phi0
    for i in range(20):
        phi_i = phi(alpha_i)
        if phi_i > phi0 + c1 * alpha_i * dphi0 or (phi_i >= phi_prev and i > 0):
            return _zoom(phi, dphi, alpha_prev, alpha_i, phi_prev, phi_i, phi0, dphi0, c1, c2)
        dphi_i = dphi(alpha_i)
        if abs(dphi_i) <= -c2 * dphi0:
            return alpha_i
        if dphi_i >= 0:
            return _zoom(phi, dphi, alpha_i, alpha_prev, phi_i, phi_prev, phi0, dphi0, c1, c2)
        alpha_prev = alpha_i
        phi_prev = phi_i
        alpha_i = min(2.0 * alpha_i, alpha_max)
    return alpha_i


def _zoom(phi, dphi, alo, ahi, phi_lo, phi_hi, phi0, dphi0, c1, c2):
    for _ in range(30):
        alpha_j = _cubic_interp(alo, ahi, phi_lo, phi_hi, dphi(alo), dphi(ahi))
        phi_j = phi(alpha_j)
        if phi_j > phi0 + c1 * alpha_j * dphi0 or phi_j >= phi_lo:
            ahi = alpha_j
            phi_hi = phi_j
        else:
            dphi_j = dphi(alpha_j)
            if abs(dphi_j) <= -c2 * dphi0:
                return alpha_j
            if dphi_j * (ahi - alo) >= 0:
                ahi = alo
                phi_hi = phi_lo
            alo = alpha_j
            phi_lo = phi_j
        if abs(ahi - alo) < 1e-14:
            break
    return alo


def _cubic_interp(a, b, fa, fb, dfa, dfb):
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

    for _ in range(max_iter):
        history['x'].append(xk.copy())
        history['f'].append(f(xk))
        history['grad_norm'].append(np.linalg.norm(gk))
        if np.linalg.norm(gk) < tol:
            history['status'] = 'converged'
            break
        dk = -Hk @ gk
        alpha = strong_wolfe(f, grad_f, xk, dk, c1=c1, c2=c2)
        history['alpha'].append(alpha)
        xk_new = xk + alpha * dk
        gk_new = grad_f(xk_new)
        sk = xk_new - xk
        yk = gk_new - gk
        sy = sk @ yk
        if sy > 1e-12:
            rho_k = 1.0 / sy
            I = np.eye(n)
            A = I - rho_k * np.outer(sk, yk)
            B = I - rho_k * np.outer(yk, sk)
            Hk = A @ Hk @ B + rho_k * np.outer(sk, sk)
        xk = xk_new
        gk = gk_new
    else:
        history['status'] = 'max_iter'

    history['x_opt'] = xk
    history['f_opt'] = f(xk)
    history['iterations'] = len(history['f'])
    return history


def backtracking_armijo_newton(f, grad_f, xk, dk, alpha0=1.0, rho=0.5, c=1e-4):
    alpha = alpha0
    fk = f(xk)
    slope = c * np.dot(grad_f(xk), dk)
    while f(xk + alpha * dk) > fk + alpha * slope:
        alpha *= rho
        if alpha < 1e-14:
            break
    return alpha


def newton_backtracking(f, grad_f, hess_f, x0, tol=1e-6, max_iter=100, alpha0=1.0, rho=0.5, c=1e-4):
    xk = np.array(x0, dtype=float)
    history = {'x': [], 'f': [], 'grad_norm': [], 'alpha': []}
    for _ in range(max_iter):
        gk = grad_f(xk)
        Hk = hess_f(xk)
        history['x'].append(xk.copy())
        history['f'].append(f(xk))
        history['grad_norm'].append(np.linalg.norm(gk))
        if np.linalg.norm(gk) < tol:
            history['status'] = 'converged'
            break
        try:
            dk = np.linalg.solve(Hk, -gk)
        except np.linalg.LinAlgError:
            dk = -gk
        if np.dot(gk, dk) >= 0:
            dk = -gk
        alpha = backtracking_armijo_newton(f, grad_f, xk, dk, alpha0, rho, c)
        history['alpha'].append(alpha)
        xk = xk + alpha * dk
    else:
        history['status'] = 'max_iter'

    history['x_opt'] = xk
    history['f_opt'] = f(xk)
    history['iterations'] = len(history['f'])
    return history
