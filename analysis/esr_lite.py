"""Dependency-light re-scoring helpers for the CLASH sqrt-grid convergence checks.

Deliberately does NOT import esr.fitting.WL_likelihood_CLASH.WLLikelihood or
esr.generation.simplifier: WLLikelihood pulls in matplotlib + pandas (broken in
this environment) and simplifier requires mpi4py (only needed for the cluster
fitting loop, not for re-scoring already-fitted params). Everything here
reproduces exactly the piece of each that re-scoring needs -- see
check_resolution_1000_vs_5000.py and triage_tail_probe.py for the callers.

Verified against results_sqrt_grid/pretty_all_clusters_comp8_1k_funcs.txt: this
loader + esd_fixed.ExcessSurfaceDensity reproduces the reported DL exactly
(e.g. pow(Abs(a0),(1/(x+pow(Abs(a1),x)))) -> 87.9918, reported 87.99).
"""
import sys

import jax
jax.config.update("jax_enable_x64", True)  # matches test_all-asmaclap80.py -- JAX
# defaults to float32 otherwise, silently, which this module ran under until this
# fix (2026-08-03). See MEMORY.md for what changed / didn't as a result.

import numpy as np
import jax.numpy as jnp
import sympy
from jax.scipy.linalg import solve_triangular
from scipy.optimize import minimize

sys.path.insert(0, '..')
from esr.fitting.sympy_symbols import *  # noqa: E402,F403 (x, a0, a1, a2, inv, square, cube, sqrt, log, pow)

import esd_fixed

DATA = '../data/CLASH_fixed_cosmo'

SYMPIFY_LOCALS = {
    "inv": inv, "square": square, "cube": cube, "sqrt": sqrt, "log": log, "pow": pow,
    "x": x, "a0": a0, "a1": a1, "a2": a2,
}


def load_likelihood(name, data_dir=DATA):
    """Same parsing as WLLikelihood.__init__: line0=xvar, line1=yvar, cov -> L_factor."""
    with open(f'{data_dir}/{name}_ESD.txt') as f:
        lines = f.readlines()
    yvar = jnp.array(list(map(float, lines[1].split())))
    no_nans = ~np.isnan(yvar)
    yvar = yvar[no_nans]
    xvar = jnp.array(list(map(float, lines[0].split())))[no_nans]
    cov_matrix = np.load(f'{data_dir}/{name}_cov_matrix.npy')
    L_factor = jnp.linalg.cholesky(jnp.array(cov_matrix))
    return xvar, yvar, L_factor


def count_params(fn, max_param=4):
    """Reimplementation of esr.generation.simplifier.count_params (avoids mpi4py import)."""
    for j in range(max_param - 1, -1, -1):
        if f'a{j}' in fn:
            return j + 1
    return 0


def get_eq_numpy(fn):
    """Same sympify path as WLLikelihood.run_sympify(physicalize=False), then lambdify."""
    fcn_i = fn.replace("'", '')
    k = count_params(fcn_i)
    eq = sympy.sympify(fcn_i, locals=SYMPIFY_LOCALS)
    all_a = list(sympy.symbols([f'a{i}' for i in range(k)], real=True))
    eq_numpy = sympy.lambdify([x] + all_a, eq, modules=["jax"])
    return eq_numpy, k


def eval_nll(eq_numpy, xvar, yvar, L_factor, params, num_points, dtype=None):
    """Same pattern as audit_quadrature_stability.py::eval_nll, using esd_fixed.

    dtype=None uses whatever xvar/yvar/L_factor/params already are (float64, since
    jax_enable_x64 is on above). Pass jnp.float32 to explicitly downcast everything
    first and check whether that changes the answer -- requires jax_enable_x64=True
    to already be set (it is, module-wide) so float32 can be requested per-array
    without a global config flip mid-process.
    """
    try:
        if dtype is not None:
            xvar = jnp.asarray(xvar, dtype=dtype)
            yvar = jnp.asarray(yvar, dtype=dtype)
            L_factor = jnp.asarray(L_factor, dtype=dtype)
            params = [jnp.asarray(p, dtype=dtype) for p in params]
        eds = esd_fixed.ExcessSurfaceDensity.calculate(
            xvar, eq_numpy, params=tuple(params), num_points=num_points)
        residuals = eds - yvar
        y = solve_triangular(L_factor, residuals, lower=True)
        val = 0.5 * float(jnp.sum(y ** 2))
        return val if np.isfinite(val) else np.inf
    except Exception:
        return np.inf


def nll_jax(eq_numpy, xvar, yvar, L_factor, params, num_points):
    """Differentiable (jnp-scalar, no float()/isfinite short-circuit) version of
    eval_nll, for use under jax.grad in a warm-started single-basin refit."""
    eds = esd_fixed.ExcessSurfaceDensity.calculate(
        xvar, eq_numpy, params=tuple(params), num_points=num_points)
    residuals = eds - yvar
    y = solve_triangular(L_factor, residuals, lower=True)
    return 0.5 * jnp.sum(y ** 2)


def _bfgs_from(eq_numpy, xvar, yvar, L_factor, x0, num_points, maxiter):
    def loss(p):
        return nll_jax(eq_numpy, xvar, yvar, L_factor, p, num_points)

    grad_fn = jax.grad(loss)

    def fun_and_grad(p_np):
        p = jnp.array(p_np)
        val = float(loss(p))
        if not np.isfinite(val):
            val = 1e12
        g = np.array(grad_fn(p))
        g = np.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)
        return val, g

    try:
        res = minimize(fun_and_grad, np.array(x0, dtype=float), jac=True, method='BFGS',
                        options={'maxiter': maxiter, 'gtol': 1e-3})
        return res.x.tolist()
    except Exception:
        return list(x0)


def _nm_from(eq_numpy, xvar, yvar, L_factor, x0, num_points, maxiter):
    def obj(p):
        val = eval_nll(eq_numpy, xvar, yvar, L_factor, list(p), num_points)
        return val if np.isfinite(val) else 1e30

    try:
        res = minimize(obj, np.array(x0, dtype=float), method='Nelder-Mead',
                        options={'maxiter': maxiter, 'maxfev': maxiter})
        return res.x.tolist()
    except Exception:
        return list(x0)


def refit_one_cluster(eq_numpy, xvar, yvar, L_factor, p0, k, num_points, maxiter=150, nm_maxiter=100):
    """Warm-started refit from p0. Never returns worse than the warm start's own
    score (falls back to p0 if every attempt fails or doesn't improve) -- this is
    meant to only ever help, not replace a proper multi-start refit.

    Tries BOTH BFGS (jax.grad) and Nelder-Mead from each candidate starting point,
    keeps whichever converges better -- BFGS-only was confirmed WRONG, not just
    weaker: on a0/x+a1 (comp 5), BFGS(150it)@N=50000 "converged" (cross-checked
    against N=200000, differences shrinking to 0.02 nats -- looked completely solid)
    to nll=384.18, while NM(100it) from the SAME start finds nll=251.10, a 133-nat
    gap BFGS never closes at ANY resolution tested. A smooth BFGS convergence
    sequence only proves BFGS stopped moving, not that it found the true optimum --
    matches the project's own MIGHTEE_ESR lesson (see MEMORY.md): local optimizers
    can get stuck in a basin a derivative-free method escapes, so the production
    pipeline itself runs both and keeps the better one. This was BFGS-only until
    that gap was found (2026-08-06).

    Tries TWO starting points per method (four attempts total), keeping whichever
    converges best: p0 itself, and a second start with any absurdly-large parameter
    (|value| > 1e4) reset to 0 -- added after finding BFGS gets stuck at a stale
    extreme value (e.g. a0~1e6 for a0+a1/x**2, warm-started from an old N=1000-tuned
    fit) rather than moving back toward the honest optimum.
    """
    if k == 0:
        nll = eval_nll(eq_numpy, xvar, yvar, L_factor, p0, num_points)
        return nll, p0

    candidates = [list(p0)]
    if any(abs(v) > 1e4 for v in p0):
        candidates.append([0.0 if abs(v) > 1e4 else v for v in p0])

    best_nll, best_p = eval_nll(eq_numpy, xvar, yvar, L_factor, p0, num_points), list(p0)
    for x0 in candidates:
        for attempt_p in (_bfgs_from(eq_numpy, xvar, yvar, L_factor, x0, num_points, maxiter),
                           _nm_from(eq_numpy, xvar, yvar, L_factor, x0, num_points, nm_maxiter)):
            attempt_nll = eval_nll(eq_numpy, xvar, yvar, L_factor, attempt_p, num_points)
            if attempt_nll < best_nll:
                best_nll, best_p = attempt_nll, attempt_p
    return best_nll, best_p
