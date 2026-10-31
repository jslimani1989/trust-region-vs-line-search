import numpy as np
from scipy.linalg import solve_triangular  # 009d : substitutions triangulaires (Dogleg)
from dataclasses import dataclass, field
from typing import Callable, Tuple, Optional


DELTA_RADIUS_FLOOR = 1e-10  # seuil de detection d'echec de rayon (B1/journalisation, consigne 003 S1e)


@dataclass
class OptResult:
    x: np.ndarray
    f: float
    grad_norm: float
    iterations: int
    f_evals: int
    g_evals: int
    cpu_time: float
    converged: bool
    method: str
    history: list = field(default_factory=list)
    stop_reason: str = ""          # B1 : 'converged' / 'max_iter' / 'radius_failure'
    stagnation_num: bool = False   # B1 : radius_failure avec grad_norm final <= 1e-4
    n_nonfinite_rejected: int = 0  # B3 : nb d'iterations candidates rejetees (rho non fini)


def _cauchy_point(g: np.ndarray, B: np.ndarray, delta: float) -> np.ndarray:
    Bg = B @ g
    gBg = g @ Bg
    if gBg <= 0:
        tau = 1.0
    else:
        tau = min(np.linalg.norm(g)**3 / (gBg * delta), 1.0)
    return -tau * delta / np.linalg.norm(g) * g


def _update_radius(delta: float, rho: float, p_norm: float,
                   eta1: float = 0.1, eta2: float = 0.75,
                   gamma1: float = 0.25, gamma2: float = 2.0,
                   delta_max: float = 1e4) -> float:
    if rho < eta1:
        return gamma1 * delta
    elif rho >= eta2 and abs(p_norm - delta) < 1e-10:
        return min(gamma2 * delta, delta_max)
    return delta


def trust_region_cauchy(
    f: Callable, grad: Callable, hess: Callable,
    x0: np.ndarray,
    tol: float = 1e-6, max_iter: int = 1000,
    delta0: float = 1.0, delta_max: float = 1e4
) -> OptResult:
    import time
    x = x0.copy().astype(float)
    delta = delta0
    f_evals = g_evals = 0
    history = []
    t0 = time.time()
    last_rho, last_delta, n_nonfinite = 0.0, delta, 0

    for k in range(max_iter):
        fk = f(x); f_evals += 1
        gk = grad(x); g_evals += 1
        Bk = hess(x)
        gnorm = np.linalg.norm(gk)
        history.append(gnorm)

        if gnorm < tol:
            return OptResult(x, fk, gnorm, k, f_evals, g_evals,
                             time.time()-t0, True, "Cauchy Point", history,
                             stop_reason="converged", stagnation_num=False,
                             n_nonfinite_rejected=n_nonfinite)

        p = _cauchy_point(gk, Bk, delta)
        pred = -(gk @ p + 0.5 * p @ Bk @ p)
        f_new = f(x + p); f_evals += 1
        if not np.isfinite(f_new):
            n_nonfinite += 1
        ared = fk - f_new
        rho = ared / pred if abs(pred) > 1e-15 else 0.0
        if not np.isfinite(rho):
            rho = 0.0  # B3 : un candidat non fini ne peut jamais etre accepte (rho>0.1 reste faux)

        if rho > 0.1:
            x = x + p
        delta = _update_radius(delta, rho, np.linalg.norm(p), delta_max=delta_max)
        last_rho, last_delta = rho, delta

    fk = f(x); gk = grad(x)
    gnorm_final = np.linalg.norm(gk)
    radius_failure = (last_rho <= 0.1) and (last_delta < DELTA_RADIUS_FLOOR)
    stop_reason = "radius_failure" if radius_failure else "max_iter"
    stagnation_num = radius_failure and (gnorm_final <= 1e-4)
    return OptResult(x, fk, gnorm_final, max_iter,
                     f_evals, g_evals, time.time()-t0, False, "Cauchy Point", history,
                     stop_reason=stop_reason, stagnation_num=stagnation_num,
                     n_nonfinite_rejected=n_nonfinite)


def trust_region_dogleg(
    f: Callable, grad: Callable, hess: Callable,
    x0: np.ndarray,
    tol: float = 1e-6, max_iter: int = 1000,
    delta0: float = 1.0, delta_max: float = 1e4
) -> OptResult:
    import time
    x = x0.copy().astype(float)
    delta = delta0
    f_evals = g_evals = 0
    history = []
    t0 = time.time()
    last_rho, last_delta, n_nonfinite = 0.0, delta, 0

    for k in range(max_iter):
        fk = f(x); f_evals += 1
        gk = grad(x); g_evals += 1
        Bk = hess(x)
        gnorm = np.linalg.norm(gk)
        history.append(gnorm)

        if gnorm < tol:
            return OptResult(x, fk, gnorm, k, f_evals, g_evals,
                             time.time()-t0, True, "Dogleg", history,
                             stop_reason="converged", stagnation_num=False,
                             n_nonfinite_rejected=n_nonfinite)

        try:
            L = np.linalg.cholesky(Bk + 1e-12 * np.eye(len(gk)))
            # 009d : deux substitutions triangulaires O(n^2) au lieu de deux resolutions par LU generale.
            pN = -solve_triangular(L.T, solve_triangular(L, gk, lower=True, check_finite=False),
                                   lower=False, check_finite=False)
        except np.linalg.LinAlgError:
            pN = _cauchy_point(gk, Bk, delta)

        pN_norm = np.linalg.norm(pN)

        if pN_norm <= delta:
            p = pN
        else:
            Bg = Bk @ gk
            gBg = gk @ Bg
            pU = -(gnorm**2 / gBg) * gk if gBg > 0 else -delta / gnorm * gk
            pU_norm = np.linalg.norm(pU)

            if pU_norm >= delta:
                p = -(delta / gnorm) * gk
            else:
                d = pN - pU
                a_ = d @ d
                b_ = 2 * pU @ d
                c_ = pU @ pU - delta**2
                disc = b_**2 - 4*a_*c_
                tau = (-b_ + np.sqrt(max(disc, 0))) / (2*a_) if a_ > 1e-15 else 1.0
                tau = np.clip(tau, 0, 1)
                p = pU + tau * d

        pred = -(gk @ p + 0.5 * p @ Bk @ p)
        f_new = f(x + p); f_evals += 1
        if not np.isfinite(f_new):
            n_nonfinite += 1
        ared = fk - f_new
        rho = ared / pred if abs(pred) > 1e-15 else 0.0
        if not np.isfinite(rho):
            rho = 0.0

        if rho > 0.1:
            x = x + p
        delta = _update_radius(delta, rho, np.linalg.norm(p), delta_max=delta_max)
        last_rho, last_delta = rho, delta

    fk = f(x); gk = grad(x)
    gnorm_final = np.linalg.norm(gk)
    radius_failure = (last_rho <= 0.1) and (last_delta < DELTA_RADIUS_FLOOR)
    stop_reason = "radius_failure" if radius_failure else "max_iter"
    stagnation_num = radius_failure and (gnorm_final <= 1e-4)
    return OptResult(x, fk, gnorm_final, max_iter,
                     f_evals, g_evals, time.time()-t0, False, "Dogleg", history,
                     stop_reason=stop_reason, stagnation_num=stagnation_num,
                     n_nonfinite_rejected=n_nonfinite)


def trust_region_steihaug(
    f: Callable, grad: Callable, hess: Callable,
    x0: np.ndarray,
    tol: float = 1e-6, max_iter: int = 1000,
    delta0: float = 1.0, delta_max: float = 1e4,
    cg_tol_factor: float = 0.5
) -> OptResult:
    import time
    x = x0.copy().astype(float)
    delta = delta0
    f_evals = g_evals = 0
    history = []
    t0 = time.time()
    last_rho, last_delta, n_nonfinite = 0.0, delta, 0

    for k in range(max_iter):
        fk = f(x); f_evals += 1
        gk = grad(x); g_evals += 1
        Bk = hess(x)
        gnorm = np.linalg.norm(gk)
        history.append(gnorm)

        if gnorm < tol:
            return OptResult(x, fk, gnorm, k, f_evals, g_evals,
                             time.time()-t0, True, "Steihaug-CG", history,
                             stop_reason="converged", stagnation_num=False,
                             n_nonfinite_rejected=n_nonfinite)

        cg_tol = min(cg_tol_factor, np.sqrt(gnorm)) * gnorm
        p = np.zeros_like(gk)
        r = gk.copy()
        d = -gk.copy()

        for _ in range(len(gk)):
            Bd = Bk @ d
            dBd = d @ Bd
            if dBd <= 0:
                a_ = d @ d
                b_ = 2 * p @ d
                c_ = p @ p - delta**2
                disc = b_**2 - 4*a_*c_
                sigma = (-b_ + np.sqrt(max(disc, 0))) / (2*a_) if a_ > 1e-15 else 0.0
                p = p + sigma * d
                break

            alpha_cg = (r @ r) / dBd
            p_new = p + alpha_cg * d

            if np.linalg.norm(p_new) >= delta:
                a_ = d @ d
                b_ = 2 * p @ d
                c_ = p @ p - delta**2
                disc = b_**2 - 4*a_*c_
                sigma = (-b_ + np.sqrt(max(disc, 0))) / (2*a_) if a_ > 1e-15 else 0.0
                p = p + sigma * d
                break

            p = p_new
            r_new = r + alpha_cg * Bd

            if np.linalg.norm(r_new) < cg_tol:
                break

            beta = (r_new @ r_new) / (r @ r)
            d = -r_new + beta * d
            r = r_new

        pred = -(gk @ p + 0.5 * p @ Bk @ p)
        f_new = f(x + p); f_evals += 1
        if not np.isfinite(f_new):
            n_nonfinite += 1
        ared = fk - f_new
        rho = ared / pred if abs(pred) > 1e-15 else 0.0
        if not np.isfinite(rho):
            rho = 0.0

        if rho > 0.1:
            x = x + p
        delta = _update_radius(delta, rho, np.linalg.norm(p), delta_max=delta_max)
        last_rho, last_delta = rho, delta

    fk = f(x); gk = grad(x)
    gnorm_final = np.linalg.norm(gk)
    radius_failure = (last_rho <= 0.1) and (last_delta < DELTA_RADIUS_FLOOR)
    stop_reason = "radius_failure" if radius_failure else "max_iter"
    stagnation_num = radius_failure and (gnorm_final <= 1e-4)
    return OptResult(x, fk, gnorm_final, max_iter,
                     f_evals, g_evals, time.time()-t0, False, "Steihaug-CG", history,
                     stop_reason=stop_reason, stagnation_num=stagnation_num,
                     n_nonfinite_rejected=n_nonfinite)
