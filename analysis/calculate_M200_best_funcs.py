import numpy as np
import sys
from sympy import *
import sympy

from esr.fitting.likelihood import Likelihood
from esr.generation.simplifier import time_limit
from esr.fitting.sympy_symbols import *
import esr.generation.simplifier as simplifier
from esr.fitting.fit_single import fit_from_string
from esr.fitting.WL_likelihood_CLASH import WLLikelihood
from esr.esd import ExcessSurfaceDensity
from numpyro import deterministic
from esr.fitting.fit_single import fit_from_string
from numpyro.diagnostics import gelman_rubin

import jax
jax.config.update("jax_enable_x64", True)
jax.numpy.array(1, dtype=int)

import jax.numpy as jnp

import corner
import numpyro.distributions as dist
from numpyro.infer.util import init_to_value, init_to_uniform
from numpyro.infer.initialization import init_to_median

import numpyro
from numpyro.infer import MCMC, NUTS
import jax.numpy as jnp
import jax.random
from matplotlib.colors import LogNorm
import matplotlib
matplotlib.use('Agg')  # non-interactive backend: save figures without a display (safe under MPI)
from matplotlib import pyplot as plt
from scipy.stats import multivariate_normal
from astropy.cosmology import FlatLambdaCDM
import os
import csv
from mpi4py import MPI

from numpyro import sample, factor, deterministic
from numpyro.distributions import Uniform

import pickle


#calculate mass for best fit params
import jax.numpy as jnp
from jax import jit, grad
from scipy.interpolate import interp1d
from scipy.optimize import root_scalar
import numpy as np
from scipy.optimize import fsolve


#-----------------------------------------M200----------------------------------
def get_M_delta_2(delta, fcn_i, params_uncertainties, z, k, tol=1e-2):

    params = params_uncertainties[:k]
    uncertainties = params_uncertainties[k:]
    # Constants
    G = 6.67430e-11  # Gravitational constant in m^3 kg^-1 s^-2
    H0 = 70.0 * 1e3 / (3.086e22)  # Hubble constant in s^-1 (H0 = 70 km/s/Mpc)

    # Define critical density at redshift z
    def rho_crit(z=0):
        H_z = H0 * np.sqrt(0.3 * (1 + z)**3 + 0.7)  # Flat Lambda-CDM model
        rho_crit_si = (3 * H_z**2) / (8 * np.pi * G)  # Critical density in kg/m³
        # conversion_factor = 1.477e37  # Convert kg/m³ to M_sun/Mpc³
        M_sun = 1.98847e30       # kg
        Mpc = 3.085677581e22     # m
        conversion_factor = (Mpc**3) / M_sun  # Convert kg/m³ to M_sun/Mpc³
        return rho_crit_si * conversion_factor  # Critical density in M_sun/Mpc³

    # @jit
    # def rho_profile(fcn_i, params, r):
    #     # Example: replace with your specific function
    #     rho = fcn_i(r, *params)*1e12 #units of solar masses per cubic Mpc
    #     return rho
    def make_rho_profile(fcn_i):
        @jit
        def rho_profile(r, params):
            # print(params)
            return fcn_i(r, *params) * 1e12  # Ensure returns M_sun/Mpc^3
        return rho_profile
    
    def make_mass_enclosed_grid(rho_profile):
        @jit
        def mass_enclosed_grid(r_vals, params):
            rho_vals = rho_profile(r_vals, params)
            integrand = 4 * jnp.pi * rho_vals * r_vals**2
            # dr = r_vals[1] - r_vals[0]
            # dr = jnp.diff(r_vals, prepend=0)
            # return jnp.cumsum(integrand) * dr
            mass_cumulative = jnp.cumsum(
            0.5 * (integrand[1:] + integrand[:-1]) * jnp.diff(r_vals)
            )
            # prepend zero mass for r=0
            mass_cumulative = jnp.concatenate([jnp.array([0.0]), mass_cumulative])
            return mass_cumulative
        return mass_enclosed_grid

    
    # r_vals = jnp.linspace(1e-6, 5.0, 1000)  # Mpc
    #do it in log
    r_vals = np.logspace(-6, 1, 1000)  # from 0.000001 to 10 Mpc
    rho_profile = make_rho_profile(fcn_i)
    mass_enclosed_grid = make_mass_enclosed_grid(rho_profile)
    M_vals = mass_enclosed_grid(r_vals, params)
    #print(f"Mass values: {M_vals}")
    # M_vals = mass_enclosed_grid(fcn_i, r_vals, params)

    # M_interp = interp1d(r_vals, np.array(M_vals), kind='cubic', bounds_error=False, fill_value='extrapolate')
    M_interp = lambda r: jnp.interp(r, r_vals, M_vals)

    rho_c = rho_crit(z)
    # print(rho_crit(0.5))

    # print(f"Critical density at z=0: {rho_c:.3e} M_sun/Mpc^3")

    def to_solve(r):
        M_r = M_interp(r)
        rho_mean = M_r / ((4/3) * np.pi * r**3)
        return rho_mean - delta * rho_c

    def safe_root_solve(func, bracket=None, x0=1.0, tol=1e-2,
                    check_func=None, target_value=None):
        # def check_root(r, rho_mean, rho_target, tol=1e-2):
        #     return np.abs(rho_mean(r) - rho_target) / rho_target < tol
        def check_root(r):
            if check_func is None or target_value is None:
                # fallback: check func(r) ~ 0 directly
                return np.isclose(func(r), 0, atol=tol*abs(target_value) if target_value else tol)
            return np.abs(check_func(r) - target_value) / target_value < tol
        
        if bracket is not None:
            if np.sign(func(bracket[0])) != np.sign(func(bracket[1])):
                try:
                    sol = root_scalar(func, bracket=bracket, method="brentq")
                    #print('brentq r200', sol.root, func(sol.root))
                    if check_root(sol.root):
                        #print('Succed')
                        return sol.root
                    
                except Exception as e:
                    #print(f"Warning: root finding with brentq failed with error: {e}")
                    pass
        # fallback
        r200 = fsolve(func, x0=x0)[0]
        #print('fsolve r200', r200, func(r200))
        if check_root(r200):

            return r200
        #print('HERE')
        return None
    
   
    r200 = safe_root_solve(
        to_solve, bracket=[1e-6, 6], x0=1e-2,
        check_func=lambda r: M_interp(r) / ((4/3) * np.pi * r**3),
        target_value=delta * rho_c, tol=1e-2
    )
    if r200 is None:
        # print("Warning: r200 root finding failed for func {fcn_i}")
        return None
    # # M200 = M_interp(r200)
    M200 = (4/3) * np.pi * r200**3 * (delta * rho_c)

    # #Calculate the concentration

    # # ---- Compute r0.1 for new concentration ----
    target_mass = 0.1 * M200
    def mass_fraction_eq(r):
        return M_interp(r) - target_mass
    

    # r01 = safe_root_solve(mass_fraction_eq, bracket=[r_vals[0], r200], x0=r200/10)
    r01 = safe_root_solve(
        lambda r: M_interp(r) - target_mass,
        bracket=[r_vals[0], r200], x0=r200/10,
        check_func=M_interp, target_value=target_mass, tol=1e-2
    )
    if r01 is None:
        print("Warning: r0.1 root finding failed.")
        return None
    

    c01 = r200 / r01

    return M200, r200, c01, r01

#-----------------------------------------MCMC----------------------------------
#get mcmc samples

def numerical_grad(f, params, eps=1e-6):
    """Finite-difference gradient for a scalar-output function."""
    params = jnp.atleast_1d(params)
    grad = []
    for i in range(len(params)):
        params_plus = params.at[i].add(eps)
        params_minus = params.at[i].add(-eps)
        f_plus = f(params_plus)
        f_minus = f(params_minus)
        grad_i = (f_plus - f_minus) / (2 * eps)
        grad.append(grad_i)
    return jnp.array(grad)

def wrap_with_numerical_grad(f):
    """Wrap f so that it provides numerical gradients for NUTS."""
    @jax.custom_jvp
    def f_wrap(params):
        return f(params)
    
    @f_wrap.defjvp
    def f_jvp(primals, tangents):
        params, = primals
        tangent, = tangents
        val = f(params)
        grad = numerical_grad(f, params)
        return val, jnp.dot(grad, tangent)
    
    return f_wrap
def model(chi2_fcn, xvar, yvar, yerr, nparam, signs, params, delta, max_vals=None, min_vals=None, use_numerical_grad = False):

    p = []

    for i in range(nparam):
        min_val_log = min_vals[i]
        max_val_log = max_vals[i]
        ai = numpyro.sample(f'log_a{i}', dist.Uniform(min_val_log, max_val_log))
        #using a dist improper uniform distribution
        # ai = numpyro.sample(f'log_a{i}', dist.ImproperUniform(dist.constraints.real, (), event_shape=())) 
        p.append(ai)

    def negloglike(par, *args, signs=None):
        val, g = chi2_fcn(par, *args, signs)
        return val, g

    # negloglike, grad_nll = negloglike(p, xvar, yvar, yerr, signs=signs)
    if use_numerical_grad:
        safe_chi2 = wrap_with_numerical_grad(lambda par: chi2_fcn(par, xvar, yvar, yerr, signs)[0])
        negloglike = safe_chi2(jnp.array(p))
    else:
        negloglike, grad_nll = negloglike(p, xvar, yvar, yerr, signs=signs)
    # jax.debug.print('params: {kk}', kk = p)
    # jax.debug.print('negloglike: {kk}', kk = negloglike)
    # jax.debug.print('grad_nll: {kk}', kk = grad_nll)
    deterministic("loglike", negloglike)
    numpyro.factor('negloglike', - negloglike)

def run_mcmc(chi2_fcn, xvar, yvar, yerr, nparam, num_sigma, signs, params, delta, log_opt=True, use_numerical_grad = False):

    params_copy = np.copy(params)
    delta_copy = np.copy(delta)
    initial_values = {}
    if log_opt:
        # print('params:', params_copy, flush=True)
        # params_copy = [-1.32361339e-01, -9.99809130e-07]
        # signs = [-1, -1]
        signs[params_copy == 0] = 1
        params_copy[params_copy == 0.0] = -9.99809130e-07 # Avoid log(0

        # print('params_copy:', params_copy)
        delta_copy = delta_copy/ np.abs(params_copy)/ np.log(10)
        delta_copy[delta_copy == 0.0] = 3
        params_copy = np.log10(np.abs(params_copy))
        # delta_copy = np.log10(np.abs(delta_copy))
    else:
        #  params_copy[params_copy == 0.0] = 0
         delta_copy[delta_copy == 0.0] = 3

    for i in range(0, nparam):
                initial_values[f'log_a{i}'] = params_copy[i]

    #calcuate min and max values for the priors
    prior_width = num_sigma * delta_copy
    max_vals = []
    min_vals = []
    for i in range(0, nparam):
        log_a_centre = initial_values[f'log_a{i}']
        min_val_log = log_a_centre - prior_width[i]
        max_val_log = log_a_centre + prior_width[i]
        min_vals.append(min_val_log)
        max_vals.append(max_val_log)


    # print(params_copy)
    initial_params = np.array([initial_values[f'log_a{i}'] for i in range(nparam)])

    print('initial values:', initial_values)
    print('all_deltas:', delta_copy)
    # print('prior width:', prior_width)
    # print('min vals:', min_vals)
    # print('max vals:', max_vals)

    kernel = NUTS(model, init_strategy = init_to_value(values = initial_values), target_accept_prob=0.95)
    # kernel = NUTS(model, init_strategy=init_to_uniform())
    # kernel = NUTS(model, init_strategy = init_to_median(num_samples = 1000))
    mcmc = MCMC(kernel, num_warmup=1000, num_samples=6000, num_chains=2, progress_bar=False) 
    chi2_val = chi2_fcn(initial_params, xvar, yvar, yerr, signs=signs)
    

    print('Initial chi2:', chi2_val)


    with numpyro.validation_enabled():
    # res = mcmc.run(init_rng_key, data)
        res = mcmc.run(jax.random.PRNGKey(0), chi2_fcn, xvar, yvar, yerr, nparam, signs, params_copy, delta_copy, max_vals=max_vals, min_vals=min_vals, use_numerical_grad=use_numerical_grad)

    #instead of printing this every time, save to a log file and check R stats
    mcmc.print_summary()

    return mcmc

def get_mcmc_samples(fn, params, delta, num_sigma, likelihood, log_opt=True, use_numerical_grad = False):
    fcn_i = fn.replace('\'', '')

    max_param = 4
    k = simplifier.count_params([fcn_i], max_param)[0]


    fcn_i, eq= likelihood.run_sympify(fcn_i)

    n_fun_params = k
    n_extra = 0

    if n_fun_params == 0 and n_extra == 0:
        eq_numpy = sympy.lambdify([x], eq, modules=["jax"])
    elif n_fun_params == 0 and n_extra > 0:
            rho0, rs = sympy.symbols("rho0 rs", real=True)
            eq_numpy = sympy.lambdify([x, rho0, rs], eq, modules=["jax"])
    elif n_fun_params > 1:
        all_a = ' '.join([f'a{i}' for i in range(n_fun_params)])
        all_a = list(sympy.symbols(all_a, real=True))
        if n_extra>0:
            # print('HERE')
            all_a += sympy.symbols("rho0 rs", real=True)
        eq_numpy = sympy.lambdify([x] + all_a, eq, modules=["jax"])
    elif n_fun_params == 1:
        if n_extra>0:
            rho0, rs = sympy.symbols("rho0 rs", real=True)
            eq_numpy = sympy.lambdify([x, a0, rho0, rs], eq, modules=["jax"])
        else:
            eq_numpy = sympy.lambdify([x, a0], eq, modules=["jax"])

    # print('eq:', eq)
    loss_template = likelihood.get_loss(eq_numpy)
    chi2_fcn = likelihood.get_wrapped_like(loss_template)

    nparams = n_fun_params + n_extra
    #get signs from the signs of the params
    if log_opt:
        signs = np.sign(params)[:n_fun_params]
    else:
        signs = None

    # print('params', params)
    params = np.array(params[:n_fun_params])
    delta = np.array(delta[:n_fun_params])
    mcmc = run_mcmc(chi2_fcn, likelihood.xvar, likelihood.yvar, likelihood.yerr, nparams, num_sigma, signs = signs, params = params, delta=delta, log_opt=log_opt, use_numerical_grad = use_numerical_grad)
    sampled_params = mcmc.get_samples()
    loglike = sampled_params.pop("loglike")
    posterior_samples = mcmc.get_samples(group_by_chain=True)


    # Compute R-hat (returns a dict with parameter names as keys)
    rhat_threshold = 1.01
    rhat_values = {kk: float(gelman_rubin(vv))
                   for kk, vv in posterior_samples.items() if kk != 'loglike'}
    max_rhat = max(rhat_values.values()) if rhat_values else np.nan

    bad_rhat = {kk: v for kk, v in rhat_values.items() if v > rhat_threshold}
    if bad_rhat:
        print(f"WARNING: poor convergence (R-hat > {rhat_threshold}):")
        for kk, v in bad_rhat.items():
            print(f"  {kk}: R-hat = {v:.3f}")
    else:
        print(f"All parameters converged (R-hat < {rhat_threshold}), max R-hat = {max_rhat:.3f}")

    # plt.figure(figsize=(10, 10))
    # corner.corner(sampled_params, labels=['a0', 'a1'], truths=np.log10(np.abs(params)), show_titles=True, title_kwargs={"fontsize": 12}, color='blue', smooth=True, smooth1d=True)
    # plt.show()

    # plt.figure(figsize=(10, 10))
    # corner.corner(sampled_params, range=None, plot_contours=True, plot_datapoints=False, 
    #         bins=30, quantiles=[0.16, 0.5, 0.84], 
    #         show_titles=True, title_fmt=".2f",
    #         label_kwargs={"fontsize": 14})
    # plt.show()

    #plot the trace plots
    # idata = az.from_numpyro(mcmc)
    # az.plot_trace(idata, var_names=[f'log_a{i}' for i in range(nparams)], compact=True)
    if log_opt:
        for i in range(0, n_fun_params):
            sampled_params[f'log_a{i}'] = 10**(sampled_params[f'log_a{i}']) * signs[i]
    # print('params', np.log10(np.abs(params)))

    #calculate the negloglike of the mean params
    # mean_param0 = np.median(sampled_params['log_a0'])
    # mean_param1 = np.median(sampled_params['log_a1'])
    # mean_params = np.array([mean_param0, mean_param1])
    # negloglike, _ = chi2_fcn(mean_params, likelihood.xvar, likelihood.yvar, likelihood.yerr, signs=signs)
    # print('Mean params:', mean_params)
    # print('Mean negloglike:', negloglike)
    # quit()


    # print(sampled_params)
    param_names = [f'log_a{i}' for i in range(nparams)]
    sampled_params = np.column_stack([sampled_params[name] for name in param_names])

    
    # sampled_params = np.array([sampled_params['log_a0'], sampled_params['log_a1']]).T

    return sampled_params, params, chi2_fcn, eq_numpy, likelihood, max_rhat


#----------------------------------PLOT M200 HISTOGRAMS----------------------------------
def plot_M200_histograms(cluster_name, mass_dir='esr/fitting/output/masses_3'):
    """Read the cached mass_results_cluster_<name>.npz and (re)generate one
    log10(M200) histogram per function. Decoupled from the (slow) mass loop so
    plots can be regenerated instantly."""
    mass_file = f'{mass_dir}/mass_results_cluster_{cluster_name}.npz'
    if not os.path.exists(mass_file):
        print(f"plot_M200_histograms: no mass file for {cluster_name} ({mass_file})")
        return

    data = np.load(mass_file, allow_pickle=True)
    mass_samples_by_func = data['mass_samples_by_func']
    os.makedirs(mass_dir, exist_ok=True)

    n_plotted = 0
    for func_result in mass_samples_by_func:
        fcn_str = func_result.get('func', '?')
        function_index = func_result.get('function_index', -1)
        mass_samples = np.asarray(func_result.get('M200_samples', []), dtype=float)
        m200_valid = mass_samples[~np.isnan(mass_samples)]
        if m200_valid.size == 0:
            continue

        max_rhat = func_result.get('max_rhat', np.nan)
        mean_mass = np.mean(m200_valid)
        median_mass = np.median(m200_valid)
        print(f"[M200] {fcn_str}: mean={mean_mass:.3e} M_sun, median={median_mass:.3e} M_sun")

        try:
            plt.figure(figsize=(8, 6))
            plt.hist(np.log10(m200_valid), bins=30, color='skyblue', edgecolor='black', alpha=0.7)
            plt.axvline(np.log10(median_mass), color='k', ls='--',
                        label=f'median={median_mass:.2e}')
            plt.xlabel('log10(M200) [M_sun]', fontsize=14)
            plt.ylabel('Number of Samples', fontsize=14)
            plt.title(f'{cluster_name} | {fcn_str} | max R-hat={max_rhat:.3f}')
            plt.legend(fontsize=10)
            plt.tight_layout()
            hist_path = f'{mass_dir}/M200_hist_{cluster_name}_func{function_index}.png'
            plt.savefig(hist_path, dpi=120)
            plt.close()
            n_plotted += 1
        except Exception as e:
            print(f"   (M200 histogram failed for {fcn_str}: {e})")
    print(f"plot_M200_histograms: saved {n_plotted} histograms for {cluster_name}")


#-----------------------------------------FUNC----------------------------------
def get_eq_numpy(fn, likelihood):
    fcn_i = fn.replace('\'', '')

    max_param = 4
    k = simplifier.count_params([fcn_i], max_param)[0]


    fcn_i, eq= likelihood.run_sympify(fcn_i)

    
    n_fun_params = k
    n_extra = 0

    if n_fun_params == 0 and n_extra == 0:
        eq_numpy = sympy.lambdify([x], eq, modules=["jax"])
    elif n_fun_params == 0 and n_extra > 0:
            rho0, rs = sympy.symbols("rho0 rs", real=True)
            eq_numpy = sympy.lambdify([x, rho0, rs], eq, modules=["jax"])
    elif n_fun_params > 1:
        all_a = ' '.join([f'a{i}' for i in range(n_fun_params)])
        all_a = list(sympy.symbols(all_a, real=True))
        if n_extra>0:
            # print('HERE')
            all_a += sympy.symbols("rho0 rs", real=True)
        eq_numpy = sympy.lambdify([x] + all_a, eq, modules=["jax"])
    elif n_fun_params == 1:
        if n_extra>0:
            rho0, rs = sympy.symbols("rho0 rs", real=True)
            eq_numpy = sympy.lambdify([x, a0, rho0, rs], eq, modules=["jax"])
        else:
            eq_numpy = sympy.lambdify([x, a0], eq, modules=["jax"])

    return eq_numpy, k

#----------------------------------READ DATA----------------------------------
def read_data():
    file_best = './combine_all_comp_all_clustersbest_funcs_change_snapping_merged.txt'
    params_file = './all_params_CLASH.csv'

    # ---- READ BEST FUNCTIONS FILE ----
    with open(file_best, 'r') as f:
        lines_best = f.read().split('\n')[3:-1]  # skip first 3 lines (header)
        lines_best = [line.split('|') for line in lines_best if line.strip()]

    # Some rows come out malformed (e.g. missing the leading '|', which shifts
    # every field left by one) — drop those instead of letting one bad row
    # crash the whole parse, but say which ones were dropped.
    good_lines_best = []
    for raw_idx, line in enumerate(lines_best):
        if len(line) <= 8 or not line[1].strip().lstrip('-').isdigit():
            print(f"Skipping malformed row {raw_idx} in {file_best}: {line}")
            continue
        good_lines_best.append(line)
    lines_best = good_lines_best

    rank = np.array([int(line[1]) for line in lines_best])
    fcn_list = [line[2].strip() for line in lines_best]
    DL = jnp.array([float(line[3].strip()) for line in lines_best])
    Prel = jnp.array([float(line[4].strip()) for line in lines_best])
    negloglike = jnp.array([float(line[5].strip()) for line in lines_best])
    codelen = jnp.array([float(line[6].strip()) for line in lines_best])
    ayfeyn = jnp.array([float(line[7].strip()) for line in lines_best])
    katz = jnp.array([float(line[8].strip()) for line in lines_best])

    # ---- READ PARAMS FILE ----
    with open(params_file, 'r') as f:
        reader = csv.DictReader(f, delimiter=',')  # params file is tab-separated
        params_raw = [row for row in reader]
        #print(params_raw)
    # Group params by function string
    cluster_indices = sorted(list(set(int(row['ClusterIndex']) for row in params_raw)))
    num_clusters = max(cluster_indices) + 1  # assuming indices start at 0

    # Build arrays: function -> cluster -> params
    unique_funcs = sorted(list(set(row['Function'] for row in params_raw)))
    params_list = []
    delta_list = []

    for func in unique_funcs:
        func_rows = [r for r in params_raw if r['Function'] == func]
        func_params = np.zeros((num_clusters, 2))
        func_delta = np.zeros((num_clusters, 2))
        for r in func_rows:
            idx = int(r['ClusterIndex'])
            func_params[idx, 0] = float(r['a0'])
            func_params[idx, 1] = float(r['a1'])
            func_delta[idx, 0] = float(r['d0'])
            func_delta[idx, 1] = float(r['d1'])
        params_list.append(func_params)
        delta_list.append(func_delta)


    # ---- SORT BY DL ----
    sorted_indices = np.argsort(DL)
    fcn_list = np.array([fcn_list[i] for i in sorted_indices])
    DL = DL[sorted_indices]
    Prel = Prel[sorted_indices]
    negloglike = negloglike[sorted_indices]
    codelen = codelen[sorted_indices]
    ayfeyn = ayfeyn[sorted_indices]
    katz = katz[sorted_indices]

    # ---- FILTER BY DL DIFFERENCE ----
    max_DL_diff = 5000
    good_mask = (DL - DL[0]) <= max_DL_diff

    DL = DL[good_mask]
    fcn_list = fcn_list[good_mask]
    negloglike = negloglike[good_mask]
    Prel = Prel[good_mask]

    # ---- Normalize Prel ----
    Prel_DL = DL - DL[0]
    Prel = np.exp(-Prel_DL)
    Prel /= np.sum(Prel)

    # ---- BUILD FINAL RESULT ----
    results = []
    for i, func in enumerate(fcn_list):
        func_idx = unique_funcs.index(func)
        results.append({
            "function": func,
            "DL": float(DL[i]),
            "Prel": float(Prel[i]),
            "negloglike": float(negloglike[i]),
            "codelen": float(codelen[i]),
            "ayfeyn": float(ayfeyn[i]),
            "katz": float(katz[i]),
            'params': params_list[func_idx],  # shape (num_clusters, 2)
            'delta': delta_list[func_idx]     # shape (num_clusters, 2)
        })

    print(f"Loaded {len(results)} unique functions.")
    return results


# ----------------------------------MAIN----------------------------------
# Set to True to force re-running the MCMC even if a cached .npz exists.
force_rerun_mcmc = False
# Set to True to force re-running the (slow) M200 sample loop even if a cached
# mass_results_cluster_*.npz exists. Leave False to skip it and only re-plot.
force_rerun_mass = False

# Set up MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

#read clusters
names_all_clusters = './cluster_names.txt'
# Load the names of all clusters
with open(names_all_clusters) as f:
    names = f.read().splitlines()

redshifts_file = 'esr/data_CLASH/cluster_redshifts.txt'
with open(redshifts_file) as f:
    lines = f.read().splitlines()

galaxy_redshifts = []
for line in lines:
    parts = line.split()
    galaxy_redshifts.append(float(parts[1]))

basis_functions = [["x", "a"], ["inv", "abs", "log", "exp", "re", "im"], ["+", "*", "-", "/", "pow"]]

#limit to first 20 clusters for testing
#names = names[20:]
names_to_do = names[:1]   # was names[0], which sliced a single name string by character
total_num_clusters = len(names_to_do)

#separate clusters in cores
clusters_per_core = total_num_clusters // size
remainder = total_num_clusters % size
if rank < remainder:
    start_index = rank * (clusters_per_core + 1)
    end_index = start_index + clusters_per_core + 1
else:
    start_index = rank * clusters_per_core + remainder
    end_index = start_index + clusters_per_core
clusters_for_this_core = names_to_do[start_index:end_index]

if rank == 0:
    results = read_data()
else:
    results = None
# Broadcast results to all cores
results = comm.bcast(results, root=0)

funcs = [res['function'] for res in results]
DL = jnp.array([res['DL'] for res in results])
Prel = jnp.array([res['Prel'] for res in results])
#params = [np.array([[p['a0'], p['a1']] for p in res['params']]) for res in results]
#delta  = [np.array([[p['d0'], p['d1']] for p in res['params']]) for res in results]

#find the number of fucntions until prel sum is 0.99
cumulative_prel = np.cumsum(Prel)
num_funcs_99 = np.searchsorted(cumulative_prel, 0.9) + 1
print(f"Number of functions to reach 90% of total Prel: {num_funcs_99}")

#params = np.array([[p['a0'], p['a1']] for res in results for p in res['params']])
#delta = np.array([[d['d0'], d['d1']] for res in results for d in res['params']])
for i, cluster_name in enumerate(clusters_for_this_core):
    print(f"Core {rank} processing cluster {cluster_name} ({i+1}/{len(clusters_for_this_core)})")

    #read data
    z = galaxy_redshifts[names.index(cluster_name)]

    likelihood = WLLikelihood(f'esr/data_CLASH/' + str(cluster_name) + '_ESD.txt', 'esr/data_CLASH/' + str(cluster_name) + '_cov_matrix.npy', f'WL_CLASH{cluster_name}', data_dir=None, fn_set='core_maths')
    # funcs, DL, Prel, params, delta, negloglike, comp = read_data(cluster_name)

    # Path to the cached MCMC results for this cluster
    output_file = f'./output_masses/mcmc_results_cluster_{cluster_name}.npz'

    # If cached MCMC exists (and we're not forcing a rerun), load it instead of
    # re-running NUTS. The MCMC is the expensive part.
    if (not force_rerun_mcmc) and os.path.exists(output_file):
        print(f"Core {rank}: loading cached MCMC results for {cluster_name} from {output_file}")
        mcmc_results = list(np.load(output_file, allow_pickle=True)['mcmc_results'])
        run_mcmc_this_cluster = False
    else:
        run_mcmc_this_cluster = True
        mcmc_results = []

    sum_prel = 0
    #get samples for functions until sum_prel = 0.99
    for j in (range(len(funcs)) if run_mcmc_this_cluster else []):
        if sum_prel < 0.9999 and Prel[j] > 0.0:
            sum_prel += Prel[j]
            fcn_str = funcs[j]
            #print(fcn_str)
            eq_numpy, k = get_eq_numpy(fcn_str, likelihood)
            if k == 0:
                print(f"Function {fcn_str} has 0 parameters, skipping MCMC...")
                result_dict = {
                'func': fcn_str,
                'params': np.array([]),
                'delta': np.array([]),
                'sampled_params': [np.array([])],
                'Prel': Prel[j],
                'function_index': j  # Add this to track which function it was
            }
                mcmc_results.append(result_dict)
                continue
            #print(i, k)
            #params_i = params[i, :k]
            #print(params_i)
            #uncertainties_i = delta[i, :k]
            cluster_idx = names.index(cluster_name)
            print(cluster_idx)
            params_i = results[j]['params'][cluster_idx, :k]      # get first k params
            print(params_i)
            uncertainties_i = results[j]['delta'][cluster_idx, :k]  # get first k uncertainties
            print(params_i, fcn_str, cluster_name)
            params_uncertainties = np.concatenate([params_i, uncertainties_i])

            mask = params_i == 0.0
            params_i[mask] = 1e-10
            uncertainties_i[mask] = 5*1e-10
            #print(f"Processing function {i}: {fcn_str} for cluster {cluster_name} with Prel={Prel[i]:.4f}")
            sampled_params = None
            max_rhat = np.nan
            try:
                sampled_params, params_mcmc, chi2_fcn, eq_numpy_mcmc, likelihood_mcmc, max_rhat = get_mcmc_samples(
                    fcn_str, params_i, uncertainties_i, num_sigma=5, likelihood=likelihood, log_opt=True, use_numerical_grad = False
                )
            except Exception as e:
                #print(f"Error running MCMC for func {fcn_str}: {e} for cluster {cluster_name}")
                try:
                    sampled_params, params_mcmc, chi2_fcn, eq_numpy_mcmc, likelihood_mcmc, max_rhat = get_mcmc_samples(
                        fcn_str, params_i, uncertainties_i, num_sigma=5, likelihood=likelihood, log_opt=True, use_numerical_grad = True
                    )
                except Exception as e:
                    print(f"Second attempt failed for func {fcn_str}: {e} for cluster {cluster_name}")
                    #sampled_params = [np.array([])]

            rhat_threshold = 1.01
            converged = bool(np.isfinite(max_rhat) and max_rhat < rhat_threshold)
            if sampled_params is not None and not converged:
                print(f"WARNING: func {fcn_str} for cluster {cluster_name} did NOT converge "
                      f"(max R-hat = {max_rhat:.3f}) -- treat its M200 with caution.")
            #print(sampled_params)
            result_dict = {
                'func': fcn_str,
                'params': params_i,
                'delta': uncertainties_i,
                'sampled_params': sampled_params,
                'Prel': Prel[j],
                'max_rhat': max_rhat,
                'converged': converged,
                'function_index': j  # Add this to track which function it was
            }
            
            mcmc_results.append(result_dict)

    # Save all results for this cluster (only if we re-ran the MCMC)
    if run_mcmc_this_cluster:
        np.savez_compressed(output_file, mcmc_results=mcmc_results)
        print(f"Core {rank} saved MCMC results for cluster {cluster_name} to {output_file}")

    #how cna I make it wait here until all cores are done before proceeding
    #comm.Barrier()

    # If the (slow) M200 results are already cached, skip the whole mass loop.
    output_file_mass = f'esr/fitting/output/masses_3/mass_results_cluster_{cluster_name}.npz'
    if (not force_rerun_mass) and os.path.exists(output_file_mass):
        print(f"Core {rank}: M200 results already exist for {cluster_name} "
              f"({output_file_mass}) -- skipping mass loop.")
        plot_M200_histograms(cluster_name)
        continue

    #now calculate mass and all that for each
    all_masses = []
    all_weights = []
    mass_samples = []
    mass_samples_by_func = []
    r200_list = []
    c200_list = []

    for result in mcmc_results:
        masses = []
        fcn_str = result['func']
        print('Processing function:', fcn_str)
        eq_numpy, k = get_eq_numpy(fcn_str, likelihood)
        sampled_params = result['sampled_params']
        #print('sampled_params', sampled_params)
        Prel_i = result['Prel']
        function_index = result.get('function_index', -1)

        mass_func = []
        r200_func = []
        c01_func = []
        r01_func = []
        valid_samples = []
        if sampled_params is None or len(sampled_params) == 0:
            print(f"Skipping mass calculation for func {fcn_str}: No valid MCMC samples found.")
            # Still create a placeholder result so the output file is consistent
            func_result_dict = {
                'func': fcn_str, 'function_index': function_index, 'Prel': Prel_i,
                'original_params': result['params'], 'original_delta': result['delta'],
                'n_total_samples': 0, 'n_valid_samples': 0,
                'M200_samples': np.array([]), 'r200_samples': np.array([]),
                'c01_samples': np.array([]), 'r01_samples': np.array([]),
                'valid_mcmc_samples': np.array([])
            }
            mass_samples_by_func.append(func_result_dict)
            continue # Move to the next function in mcmc_results
        if k == 0:
            # No parameters, just evaluate once
            sampled_params = [np.array([])]
        
        # Loop over all MCMC samples for this function
        #for params_sample in sampled_params:
        total_samples = len(sampled_params)
        for i, params_sample in enumerate(sampled_params, 1):
         #print the number of samples in chucks of 100 that are done
            if i % 1000 == 0 or i == total_samples:
                print(f"  Processed {i}/{total_samples} samples...")
            params_i = params_sample[:k]
            uncertainties_i = np.zeros_like(params_i)  # Uncertainties not used for each sample
            params_uncertainties = np.concatenate([params_i, uncertainties_i])
            mass_result = get_M_delta_2(200, eq_numpy, params_uncertainties, z, k)
            if mass_result is not None:
                M200, r200, c01, r01 = mass_result
                all_masses.append(M200)
                all_weights.append(Prel_i)  # Weight by Prel of the function
                mass_func.append(M200)
                r200_func.append(r200)
                c01_func.append(c01)
                r01_func.append(r01)
                valid_samples.append(params_sample)
                masses.append(M200)
                # print(f"M200: {M200:.3e} M_sun")
            else:
                #print(f"Failed to compute M200 for func {fcn_str} for cluster {cluster_name}")
                all_masses.append(np.nan)
                all_weights.append(Prel_i)
                mass_func.append(np.nan)
                r200_func.append(np.nan)
                c01_func.append(np.nan)
                r01_func.append(np.nan)
        
        if np.all(np.isnan(masses)):
            print(f"All M200 calculations failed for function {fcn_str} for cluster {cluster_name}")
        #print(masses)

        # -------- M200 diagnostic: does the parameter degeneracy propagate to the mass? --------
        max_rhat = result.get('max_rhat', np.nan)
        converged = result.get('converged', None)
        m200_arr = np.asarray(mass_func, dtype=float)
        m200_valid = m200_arr[np.isfinite(m200_arr)]
        m200_med = m200_frac_spread = m200_lo = m200_hi = np.nan
        if m200_valid.size > 0:
            m200_med = np.median(m200_valid)
            m200_std = np.std(m200_valid)
            m200_lo, m200_hi = np.percentile(m200_valid, [16, 84])
            m200_frac_spread = (m200_hi - m200_lo) / (2 * m200_med) if m200_med != 0 else np.nan
            print(f"[M200] func {fcn_str}: median={m200_med:.3e} Msun, "
                  f"std/median={m200_std/m200_med:.2f}, "
                  f"68%-spread/median={m200_frac_spread:.2f}, "
                  f"max_rhat={max_rhat:.3f}, valid={m200_valid.size}/{m200_arr.size}")
            if m200_frac_spread > 0.3:
                print(f"   -> WARNING: M200 poorly constrained for {fcn_str} "
                      f"(68% spread > 30% of median); parameter degeneracy is propagating to the mass.")

        func_result_dict = {
            'func': fcn_str,
            'function_index': function_index,
            'Prel': Prel_i,
            'max_rhat': max_rhat,
            'converged': converged,
            'original_params': result['params'],
            'original_delta': result['delta'],
            'n_total_samples': len(sampled_params),
            'n_valid_samples': len([m for m in mass_func if not np.isnan(m)]),

            # M200 summary diagnostics
            'M200_median': m200_med,
            'M200_16': m200_lo,
            'M200_84': m200_hi,
            'M200_frac_spread': m200_frac_spread,

            # Mass and structural parameters
            'M200_samples': np.array(mass_func),
            'r200_samples': np.array(r200_func),
            'c01_samples': np.array(c01_func),
            'r01_samples': np.array(r01_func),

            # Valid MCMC parameter samples (only those that gave valid masses)
            'valid_mcmc_samples': np.array(valid_samples) if valid_samples else np.array([])}
        
        #save mass samples for this function
        mass_samples_by_func.append(func_result_dict)
    
    #save this cluster results (output_file_mass defined above)
    os.makedirs('esr/fitting/output/masses_3', exist_ok=True)
    np.savez_compressed(output_file_mass, all_masses=np.array(all_masses), all_weights=np.array(all_weights), mass_samples_by_func=mass_samples_by_func)
    print(f"Core {rank} saved mass results for cluster {cluster_name} to {output_file_mass}")

    # generate the per-function log10(M200) histograms from the just-saved results
    plot_M200_histograms(cluster_name)







