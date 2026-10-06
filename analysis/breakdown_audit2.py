import sys
sys.path.insert(0, '..')
sys.path.insert(0, '.')
import numpy as np
import jax.numpy as jnp
import sympy
from esr.fitting.WL_likelihood_CLASH import WLLikelihood
import esr.generation.simplifier as simplifier
from esr.fitting.sympy_symbols import *
from esd_fixed import trapz_, atleast_kd, MIN_INTEGRATION_RADIUS, POLE_GUARD_MARGIN, POLE_GUARD_MAX_RATIO
from jax.scipy.linalg import solve_triangular
import pickle, time
from collections import Counter

CLASH = '.'
RESULTS = f'{CLASH}/results_fixed_match'
DATA = '../data/CLASH_fixed_cosmo'
ABS_TOL, REL_TOL = 0.05, 0.02   # per-cluster tolerance (tighter, since these are individual contributions)
COMPS = [6, 7, 8, 9, 10]

def get_eq_numpy(fn, likelihood):
    fcn_i = fn.replace("'", '')
    k = simplifier.count_params([fcn_i], 4)[0]
    fcn_i, eq = likelihood.run_sympify(fcn_i)
    all_a = list(sympy.symbols([f'a{i}' for i in range(k)], real=True))
    return sympy.lambdify([x] + all_a, eq, modules=["jax"]), k

def first_term_grid(radii, rho, params, N, kind):
    R = radii
    frac = jnp.linspace(0., 1., N)[:, None]
    if kind == 'sqrt':
        sqrt_rmin = jnp.sqrt(MIN_INTEGRATION_RADIUS)
        xs = (sqrt_rmin + frac * (jnp.sqrt(R[None, :]) - sqrt_rmin)) ** 2
    else:
        xs = jnp.linspace(MIN_INTEGRATION_RADIUS, R, N)
    rhos = rho(xs, *params)
    integrand = 4 * rhos * xs ** 2
    dxs = atleast_kd(jnp.gradient(xs, axis=0), integrand.ndim)
    return trapz_(integrand, axis=0, dx=dxs) / R ** 2, rhos

def second_term_grid(radii, rho, params, N, kind):
    R = radii
    if kind == 'sqrt':
        VMIN = 1e-10
        frac = jnp.linspace(0., 1., N)
        v = (jnp.sqrt(VMIN) + frac * (jnp.sqrt(np.pi / 2) - jnp.sqrt(VMIN))) ** 2
        thetas = jnp.sort(np.pi / 2 - v)
    else:
        thetas = jnp.linspace(0, np.pi / 2, N)
    thetas_ = atleast_kd(thetas[:, None], radii.ndim + 1)
    cos_t = jnp.abs(jnp.cos(thetas_))
    density_arg = R[None, ...] / cos_t
    rhos = rho(density_arg, *params)
    radii_b = atleast_kd(R[None, ...], rhos.ndim)
    thetas_b = atleast_kd(thetas_, rhos.ndim)
    integrand = 4 * radii_b * rhos / (4 * jnp.sin(thetas_b) + 3 - jnp.cos(2 * thetas_b))
    dthetas = atleast_kd(jnp.gradient(thetas, axis=0), integrand.ndim)
    return trapz_(integrand, axis=0, dx=dthetas), rhos

def check_invalid(rhos):
    return bool(jnp.any(rhos < -1e-8) or jnp.any(jnp.isinf(rhos)) or jnp.any(jnp.isnan(rhos)))

def pole_guard_ok(rho, params):
    try:
        r_lo, r_hi = jnp.asarray(MIN_INTEGRATION_RADIUS), jnp.asarray(MIN_INTEGRATION_RADIUS * POLE_GUARD_MARGIN)
        rho_lo, rho_hi = jnp.abs(rho(r_lo, *params)), jnp.abs(rho(r_hi, *params))
        ratio = rho_lo / jnp.maximum(rho_hi, 1e-300)
        return bool(jnp.isfinite(ratio) and ratio <= POLE_GUARD_MAX_RATIO)
    except Exception:
        return False

def nll_grid(lik, rho, params, grid_kind, N=500):
    try:
        ft, rf = first_term_grid(lik.xvar, rho, params, N, grid_kind)
        st, rs = second_term_grid(lik.xvar, rho, params, N, grid_kind)
        if check_invalid(rf) or check_invalid(rs):
            return np.inf
        esd = ft - st
        y = solve_triangular(lik.L_factor, esd - lik.yvar, lower=True)
        val = 0.5 * float(jnp.sum(y ** 2))
        return val if np.isfinite(val) else np.inf
    except Exception:
        return np.inf

def main():
    names = open(f'{CLASH}/cluster_names.txt').read().splitlines()
    liks = {n: WLLikelihood(f'{DATA}/{n}_ESD.txt', f'{DATA}/{n}_cov_matrix.npy', f'x{n}',
                             data_dir=None, fn_set='core_maths') for n in names}

    func_summary = []
    t_start = time.time()
    for comp in COMPS:
        dat_path = f'{RESULTS}/final_{comp}best_funcs_fixed_cosmo.dat'
        pkl_path = f'{RESULTS}/params_comp{comp}best_funcs_fixed_cosmo.pkl'
        rows = []
        with open(dat_path) as f:
            for line in f:
                parts = line.rstrip('\n').split(';')
                if len(parts) < 5 or parts[1] == '':
                    continue
                rows.append(parts)
        with open(pkl_path, 'rb') as f:
            params = pickle.load(f)

        for row in rows:
            idx = int(row[0]); fn = row[1]; reported = float(row[4])
            try:
                eq, k = get_eq_numpy(fn, liks[names[0]])
            except Exception:
                continue

            cluster_cats = []
            for c, n in enumerate(names):
                p = [float(params[f'a{j}'][idx, c]) for j in range(k)]
                orig = nll_grid(liks[n], eq, p, 'linear')       # what actually got saved (per-cluster)
                sqrt_val = nll_grid(liks[n], eq, p, 'sqrt')     # sqrt grid, no pole guard
                pg_ok = pole_guard_ok(eq, p)                     # would pole guard reject this cluster?

                def mism(a, b):
                    if not (np.isfinite(a) and np.isfinite(b)):
                        return not (np.isfinite(a) and np.isfinite(b) and abs(a-b) < 1e-6) if np.isfinite(a)==np.isfinite(b) else True
                    return abs(a - b) > max(ABS_TOL, REL_TOL * abs(b))

                collapse_issue = mism(sqrt_val, orig)   # sqrt grid disagrees with what was actually reported for this cluster
                pole_issue = not pg_ok

                if not collapse_issue and not pole_issue:
                    cluster_cats.append('ok')
                elif pole_issue and not collapse_issue:
                    cluster_cats.append('pole')
                elif collapse_issue and not pole_issue:
                    cluster_cats.append('collapse')
                else:
                    cluster_cats.append('both')

            cnt = Counter(cluster_cats)
            n_ok = cnt.get('ok', 0)
            n_pole = cnt.get('pole', 0)
            n_collapse = cnt.get('collapse', 0)
            n_both = cnt.get('both', 0)
            n_bad = 20 - n_ok
            if n_bad == 0:
                func_cat = 'ok'
            elif n_pole + n_both > 0 and n_collapse == 0:
                func_cat = 'pole_exploit'
            elif n_collapse + n_both > 0 and n_pole == 0:
                func_cat = 'collapse'
            else:
                func_cat = 'mixed'
            func_summary.append((comp, idx, fn, reported, n_ok, n_collapse, n_pole, n_both, func_cat))
        print(f'comp={comp} done ({time.time()-t_start:.1f}s elapsed)', flush=True)

    with open(f'{RESULTS}/breakdown_per_cluster.csv', 'w') as f:
        f.write('comp;idx;fn;reported_nll;n_clusters_ok;n_clusters_collapse;n_clusters_pole;n_clusters_both;function_category\n')
        for r in func_summary:
            f.write(';'.join(str(x) for x in r) + '\n')

    cats = Counter(r[8] for r in func_summary)
    print('\nFUNCTION-LEVEL CATEGORY COUNTS:', dict(cats))
    print("(category = dominant cause across that function's 20 clusters;")
    print(' "mixed" = different clusters within the same function hit different bugs)')

if __name__ == '__main__':
    main()
