"""Adaptive-quadrature ground-truth verification for specific functions -- the
step check_numerical_stability.py's fixed-grid escalation (grid_flag/precision_flag
+ BFGS+NM refit + N-escalation) can't resolve on its own, confirmed 2026-08-06:
the mass-sheet-degeneracy exploit family (a0+a1/x**2 and its relatives) persists
even at N=50000/verified-at-100000, because it's a structural gap in the fixed
grid's r->infinity mapping, not a resolution question more num_points can fix.

For each target function:
  1. Refit at N=5000 (esr_lite.refit_one_cluster, BFGS+NM combined) from the
     original per-cluster params in params_comp{N}.pkl.
  2. Evaluate that refit result under adaptive quadrature (esr.esd.
     QuadExcessSurfaceDensity) -- ground truth, confirmed cheap (~0.4-0.6s/cluster,
     not the ~164x notes.md figure, which doesn't apply to this comparison).
  3. If fixed-grid and quad disagree (CONFIRM_ABS_TOL/CONFIRM_REL_TOL from
     check_numerical_stability.py -- re-evaluating a different objective at a point
     optimised for another one isn't evidence, only a tight match is): escalate to
     a LOCAL quad-based refit (Nelder-Mead, quad isn't JAX-differentiable so BFGS's
     gradient trick doesn't apply -- modest budget, warm-started from the
     fixed-grid-optimal point, not a fresh global search).

Usage:
    python quad_verify.py 9                          # RESCORED+UNRESOLVED from stability_check_comp9.csv
    python quad_verify.py 9 --fn "a0 - a1/(a2 - x**2)"  # one specific function
    python quad_verify.py 9 --classifications RESCORED,UNRESOLVED,CONFIRMED

Output: {results-dir}/quad_verify_comp{N}.csv
"""
import argparse
import os
import pickle
import sys
import time

sys.path.insert(0, '..')
from esr.esd import QuadExcessSurfaceDensity  # noqa: E402
from jax.scipy.linalg import solve_triangular  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from scipy.optimize import minimize  # noqa: E402

import esr_lite  # noqa: E402
import check_numerical_stability as cns  # noqa: E402

CLASH = os.path.dirname(os.path.abspath(__file__))
N_REFIT = 5000
NM_LOCAL_MAXITER = 60


def quad_nll(eq_numpy, xvar, yvar, L_factor, p):
    try:
        eds = QuadExcessSurfaceDensity.calculate(np.array(xvar), eq_numpy, params=tuple(p))
        residuals = jnp.array(eds) - yvar
        y = solve_triangular(L_factor, residuals, lower=True)
        val = 0.5 * float(jnp.sum(y ** 2))
        return val if np.isfinite(val) else np.inf
    except Exception:
        return np.inf


def verify_one(fn, idx, params, names, liks, codelen, ayfeyn, DL_reported):
    eq_numpy, k = esr_lite.get_eq_numpy(fn)

    fixed_total = 0.0
    quad_at_fixed_total = 0.0
    refit_params = []
    for c, name in enumerate(names):
        xvar, yvar, L_factor = liks[name]
        p0 = [float(params[f'a{j}'][idx, c]) for j in range(k)]
        nll_c, p_refit = esr_lite.refit_one_cluster(eq_numpy, xvar, yvar, L_factor, p0, k, N_REFIT)
        fixed_total += nll_c
        refit_params.append(p_refit)
        quad_at_fixed_total += quad_nll(eq_numpy, xvar, yvar, L_factor, p_refit)

    escalated = False
    if not cns.flagged(fixed_total, quad_at_fixed_total, cns.CONFIRM_ABS_TOL, cns.CONFIRM_REL_TOL):
        final_quad_total = quad_at_fixed_total
    else:
        escalated = True
        final_quad_total = 0.0
        for c, name in enumerate(names):
            xvar, yvar, L_factor = liks[name]
            res = minimize(lambda p: quad_nll(eq_numpy, xvar, yvar, L_factor, list(p)),
                            np.array(refit_params[c]), method='Nelder-Mead',
                            options={'maxiter': NM_LOCAL_MAXITER, 'maxfev': NM_LOCAL_MAXITER})
            final_quad_total += min(res.fun, quad_nll(eq_numpy, xvar, yvar, L_factor, refit_params[c]))

    quad_DL = final_quad_total + codelen + ayfeyn
    return {
        'fn': fn, 'idx': idx, 'DL_reported': DL_reported,
        'fixed_grid_DL': fixed_total + codelen + ayfeyn,
        'quad_at_fixed_DL': quad_at_fixed_total + codelen + ayfeyn,
        'escalated': escalated, 'quad_DL': quad_DL,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('comp', type=int)
    parser.add_argument('--results-dir', default=f'{CLASH}/results_sqrt_grid')
    parser.add_argument('--fn', action='append', default=None,
                         help='specific function string(s) to check, repeatable. '
                              'Default: read RESCORED+UNRESOLVED from stability_check_comp{N}.csv')
    parser.add_argument('--classifications', default='RESCORED,UNRESOLVED',
                         help='comma-separated classifications to pull from the CSV '
                              '(ignored if --fn given)')
    args = parser.parse_args()

    results_dir = os.path.abspath(args.results_dir)
    comp = args.comp
    cns.RESULTS = results_dir  # load_pool() reads this module-level global

    if args.fn:
        targets = list(args.fn)
    else:
        wanted = set(args.classifications.split(','))
        targets = []
        with open(f'{results_dir}/stability_check_comp{comp}.csv') as f:
            next(f)
            for line in f:
                parts = line.rstrip('\n').split(';')
                if parts[12] in wanted:
                    targets.append(parts[1])

    print(f'Verifying {len(targets)} functions from comp {comp} against adaptive quadrature')

    by_fn = {}
    for row in cns.load_pool(comp):
        by_fn.setdefault(row['function'], row)

    names = open(f'{CLASH}/cluster_names.txt').read().splitlines()
    liks = {name: esr_lite.load_likelihood(name) for name in names}
    with open(f'{results_dir}/params_comp{comp}.pkl', 'rb') as f:
        params = pickle.load(f)

    results = []
    for fn in targets:
        row = by_fn.get(fn)
        if row is None:
            print(f'  SKIP (not found in final_{comp}.dat): {fn}')
            continue
        t0 = time.time()
        r = verify_one(fn, row['idx'], params, names, liks, row['codelen'], row['ayfeyn'], row['DL'])
        dt = time.time() - t0
        esc = ' [escalated to local quad-refit]' if r['escalated'] else ''
        print(f"  DL_reported={r['DL_reported']:8.2f}  fixed-grid-refit={r['fixed_grid_DL']:8.2f}  "
              f"quad={r['quad_DL']:8.2f}{esc}  ({dt:.1f}s)  {fn}", flush=True)
        results.append(r)

    out_path = f'{results_dir}/quad_verify_comp{comp}.csv'
    with open(out_path, 'w') as f:
        f.write('idx;fn;DL_reported;fixed_grid_DL;quad_at_fixed_DL;escalated;quad_DL\n')
        for r in results:
            f.write(f"{r['idx']};{r['fn']};{r['DL_reported']:.4f};{r['fixed_grid_DL']:.4f};"
                     f"{r['quad_at_fixed_DL']:.4f};{r['escalated']};{r['quad_DL']:.4f}\n")
    print(f'-> {out_path}')


if __name__ == '__main__':
    main()
