import sys
sys.path.insert(0, '..')
sys.path.insert(0, '.')
import os
import numpy as np
import jax.numpy as jnp
import sympy
from esr.fitting.WL_likelihood_CLASH import WLLikelihood
import esr.generation.simplifier as simplifier
from esr.fitting.sympy_symbols import *
from esd_fixed import ExcessSurfaceDensity as Fixed
from jax.scipy.linalg import solve_triangular
import pickle
import time

CLASH = '.'
RESULTS = f'{CLASH}/results_fixed_match'
DATA = '../data/CLASH_fixed_cosmo'
ABS_TOL = 1.0
REL_TOL = 0.10
COMPS = [6, 7, 8, 9, 10]

def get_eq_numpy(fn, likelihood):
    fcn_i = fn.replace("'", '')
    k = simplifier.count_params([fcn_i], 4)[0]
    fcn_i, eq = likelihood.run_sympify(fcn_i)
    all_a = list(sympy.symbols([f'a{i}' for i in range(k)], real=True))
    eq_numpy = sympy.lambdify([x] + all_a, eq, modules=["jax"])
    return eq_numpy, k

def eval_nll(lik, eq_numpy, params):
    try:
        eds = Fixed.calculate(lik.xvar, eq_numpy, params=tuple(params), num_points=500)
        residuals = eds - lik.yvar
        y = solve_triangular(lik.L_factor, residuals, lower=True)
        val = 0.5 * float(jnp.sum(y ** 2))
        return val if np.isfinite(val) else np.inf
    except Exception:
        return np.inf

def main():
    names = open(f'{CLASH}/cluster_names.txt').read().splitlines()
    liks = {n: WLLikelihood(f'{DATA}/{n}_ESD.txt', f'{DATA}/{n}_cov_matrix.npy', f'x{n}',
                             data_dir=None, fn_set='core_maths') for n in names}

    all_rows = []
    for comp in COMPS:
        dat_path = f'{RESULTS}/final_{comp}best_funcs_fixed_cosmo.dat'
        pkl_path = f'{RESULTS}/params_comp{comp}best_funcs_fixed_cosmo.pkl'
        rows = []
        with open(dat_path) as f:
            for line in f:
                rows.append(line.rstrip('\n').split(';'))
        with open(pkl_path, 'rb') as f:
            params = pickle.load(f)

        t0 = time.time()
        n_bad, n_ok, n_err = 0, 0, 0
        for row in rows:
            idx = int(row[0]); fn = row[1]; reported = float(row[4])
            try:
                eq, k = get_eq_numpy(fn, liks[names[0]])
            except Exception:
                n_err += 1
                all_rows.append((comp, idx, fn, 'LAMBDIFY_ERROR', None, reported))
                continue
            total = 0.0
            for c, n in enumerate(names):
                p = [float(params[f'a{j}'][idx, c]) for j in range(k)]
                total += eval_nll(liks[n], eq, p)
            mismatch = (not np.isfinite(total)) or abs(total - reported) > max(ABS_TOL, REL_TOL * abs(reported))
            verdict = 'BAD' if mismatch else 'ok'
            if mismatch: n_bad += 1
            else: n_ok += 1
            all_rows.append((comp, idx, fn, verdict, total, reported))
        dt = time.time() - t0
        print(f'comp={comp}: {len(rows)} functions, {n_bad} BAD, {n_err} LAMBDIFY_ERROR, {n_ok} ok  ({dt:.1f}s)', flush=True)

    with open(f'{RESULTS}/fixed_module_audit.csv', 'w') as f:
        f.write('comp;idx;fn;verdict;fixed_nll;reported_nll\n')
        for r in all_rows:
            f.write(';'.join(str(x) for x in r) + '\n')

    n_bad = sum(1 for r in all_rows if r[3] == 'BAD')
    n_ok = sum(1 for r in all_rows if r[3] == 'ok')
    n_err = sum(1 for r in all_rows if r[3] == 'LAMBDIFY_ERROR')
    print(f'\nTOTAL: {len(all_rows)} functions, {n_bad} BAD (badly optimised), {n_ok} ok, {n_err} errors')

if __name__ == '__main__':
    main()
