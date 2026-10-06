"""Audit every function in results_fixed_match/ for the ESD-quadrature exploit.

For each (function, cluster) pair, re-evaluates negloglike at the fitting-time
resolution (num_points=120) AND at a much finer resolution (NPTS_HIGH), using
the SAME fitted a0..a3 params already on disk (no re-optimisation). If the two
disagree by more than the tolerance, that cluster's fit is flagged as exploiting
the under-resolved trapezoid quadrature in esr/esd.py (see notes.md for the
full diagnosis -- verified mechanism: pow(Abs(a_i), x)-type terms can make the
density profile collapse from ~1e15+ to O(1) within a tiny fraction of the
radial range, which a fixed 120-point linear grid can't see).

A function is dropped if ANY of its 20 clusters is flagged, since the reported
DL sums negloglike (and codelen) across all clusters -- one exploiting cluster
corrupts the whole row.

Outputs, per comp:
  stability_audit_comp{comp}.csv   -- full per-function verdict + per-cluster detail
  filtered_final_{comp}best_funcs_fixed_cosmo.dat  -- original table with flagged rows removed
"""
import os
import sys
sys.path.insert(0, '..')
import numpy as np
import jax.numpy as jnp
import sympy
from esr.fitting.WL_likelihood_CLASH import WLLikelihood
import esr.generation.simplifier as simplifier
from esr.fitting.sympy_symbols import *
from esr.esd import ExcessSurfaceDensity
from jax.scipy.linalg import solve_triangular
import pickle
import time

CLASH = '.'
RESULTS = f'{CLASH}/results_fixed_match'
DATA = '../data/CLASH_fixed_cosmo'

NPTS_LOW = 120     # matches what match.py actually used to fit
NPTS_HIGH = 1000    # cheap (per timing benchmarks) but resolves the exploit
ABS_TOL = 1.0        # nats -- bigger than any sane quadrature-refinement drift
REL_TOL = 0.10       # 10% relative

COMPS = [6, 7, 8, 9, 10]


def get_eq_numpy(fn, likelihood):
    fcn_i = fn.replace("'", '')
    k = simplifier.count_params([fcn_i], 4)[0]
    fcn_i, eq = likelihood.run_sympify(fcn_i)
    all_a = list(sympy.symbols([f'a{i}' for i in range(k)], real=True))
    eq_numpy = sympy.lambdify([x] + all_a, eq, modules=["jax"])
    return eq_numpy, k


def build_likelihoods(names):
    liks = {}
    for name in names:
        liks[name] = WLLikelihood(f'{DATA}/{name}_ESD.txt', f'{DATA}/{name}_cov_matrix.npy',
                                   f'WL_CLASH{name}', data_dir=None, fn_set='core_maths')
    return liks


def eval_nll(lik, eq_numpy, params, npts):
    try:
        eds = ExcessSurfaceDensity.calculate(lik.xvar, eq_numpy, params=tuple(params), num_points=npts)
        residuals = eds - lik.yvar
        y = solve_triangular(lik.L_factor, residuals, lower=True)
        val = 0.5 * float(jnp.sum(y ** 2))
        return val if np.isfinite(val) else np.inf
    except Exception:
        return np.inf


def main():
    names = open(f'{CLASH}/cluster_names.txt').read().splitlines()
    liks = build_likelihoods(names)

    for comp in COMPS:
        dat_path = f'{RESULTS}/final_{comp}best_funcs_fixed_cosmo.dat'
        pkl_path = f'{RESULTS}/params_comp{comp}best_funcs_fixed_cosmo.pkl'
        if not os.path.exists(dat_path):
            continue

        rows = []
        with open(dat_path) as f:
            for line in f:
                rows.append(line.rstrip('\n').split(';'))
        reported_nll = {int(r[0]): float(r[4]) for r in rows}

        with open(pkl_path, 'rb') as f:
            params = pickle.load(f)

        t0 = time.time()
        audit_rows = []
        kept_idx = []
        for row in rows:
            idx = int(row[0])
            fn = row[1]
            try:
                eq_numpy, k = get_eq_numpy(fn, liks[names[0]])
            except Exception as e:
                audit_rows.append([idx, fn, 'LAMBDIFY_ERROR', str(e), '', ''])
                continue

            flagged_clusters = []
            nll_low_total, nll_high_total = 0.0, 0.0
            for c, name in enumerate(names):
                p = jnp.array([params[f'a{j}'][idx, c] for j in range(k)])
                nll_low = eval_nll(liks[name], eq_numpy, p, NPTS_LOW)
                nll_high = eval_nll(liks[name], eq_numpy, p, NPTS_HIGH)
                nll_low_total += nll_low
                nll_high_total += nll_high
                diff = nll_high - nll_low
                if not np.isfinite(diff) or diff > max(ABS_TOL, REL_TOL * abs(nll_low)):
                    flagged_clusters.append(name)

            reported = reported_nll.get(idx, float('nan'))
            mismatch = np.isfinite(reported) and abs(nll_low_total - reported) > max(ABS_TOL, REL_TOL * abs(reported))
            verdict = 'UNSTABLE' if (flagged_clusters or mismatch) else 'stable'
            if mismatch and not flagged_clusters:
                flagged_clusters = flagged_clusters + ['MISMATCH_VS_REPORTED']
            audit_rows.append([idx, fn, verdict, ','.join(flagged_clusters),
                                f'{nll_low_total:.4f}', f'{nll_high_total:.4e}', f'{reported:.4f}'])
            if verdict == 'stable':
                kept_idx.append(idx)

        dt = time.time() - t0
        n_unstable = sum(1 for r in audit_rows if r[2] == 'UNSTABLE')
        n_error = sum(1 for r in audit_rows if r[2] == 'LAMBDIFY_ERROR')
        n_stable = sum(1 for r in audit_rows if r[2] == 'stable')
        print(f'comp={comp}: {len(rows)} functions, {n_unstable} UNSTABLE, '
              f'{n_error} LAMBDIFY_ERROR, {n_stable} kept  ({dt:.1f}s)', flush=True)

        with open(f'{RESULTS}/stability_audit_comp{comp}.csv', 'w') as f:
            f.write('idx;fn;verdict;flagged_clusters;nll_low_total;nll_high_total;reported_nll\n')
            for r in audit_rows:
                f.write(';'.join(str(x) for x in r) + '\n')

        kept_set = set(kept_idx)
        with open(f'{RESULTS}/filtered_final_{comp}best_funcs_fixed_cosmo.dat', 'w') as f:
            for row in rows:
                if int(row[0]) in kept_set:
                    f.write(';'.join(row) + '\n')


if __name__ == '__main__':
    main()
