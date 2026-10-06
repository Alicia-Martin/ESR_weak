import numpy as np
import sys
from sympy import *
import sympy
import time
import matplotlib.pyplot as plt
from astropy.cosmology import FlatLambdaCDM

from esr.fitting.likelihood import Likelihood
from esr.generation.simplifier import time_limit
from esr.fitting.sympy_symbols import *
import esr.generation.simplifier as simplifier
from esr.fitting.fit_single import fit_from_string
from esr.fitting.WL_likelihood import WLLikelihood
from esr.esd import ExcessSurfaceDensity
from numpyro import deterministic
from esr.fitting.fit_single import fit_from_string

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
from scipy.stats import multivariate_normal
from astropy.cosmology import FlatLambdaCDM
import os
import csv
from mpi4py import MPI

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
            dr = jnp.diff(r_vals, prepend=0)
            return jnp.cumsum(integrand) * dr
        return mass_enclosed_grid

    
    r_vals = jnp.linspace(1e-6, 5.0, 1000)  # Mpc
    rho_profile = make_rho_profile(fcn_i)
    mass_enclosed_grid = make_mass_enclosed_grid(rho_profile)
    M_vals = mass_enclosed_grid(r_vals, params)
    # print(f"Mass values: {M_vals}")
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
                    # print('brentq r200', sol.root, func(sol.root))
                    if check_root(sol.root):
                        return sol.root
                    
                except Exception as e:
                    #print(f"Warning: root finding with brentq failed with error: {e}")
                    pass
        # fallback
        r200 = fsolve(func, x0=x0)[0]
        #print('fsolve r200', r200, func(r200))
        if check_root(r200):

            return r200
        return None
    
   
    r200 = safe_root_solve(
        to_solve, bracket=[1e-6, 6], x0=1e-2,
        check_func=lambda r: M_interp(r) / ((4/3) * np.pi * r**3),
        target_value=delta * rho_c, tol=1e-2
    )
    if r200 is None:
        #print("Warning: r200 root finding failed.")
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
def model(chi2_fcn, xvar, yvar, yerr, nparam, signs, params, delta, max_vals=None, min_vals=None):

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

    negloglike, grad_nll = negloglike(p, xvar, yvar, yerr, signs=signs)
    # jax.debug.print('params: {kk}', kk = p)
    # jax.debug.print('negloglike: {kk}', kk = negloglike)
    # jax.debug.print('grad_nll: {kk}', kk = grad_nll)
    deterministic("loglike", negloglike)
    numpyro.factor('negloglike', - negloglike)

def run_mcmc(chi2_fcn, xvar, yvar, yerr, nparam, num_sigma, signs, params, delta, log_opt=True):

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

    #print('initial values:', initial_values)
    #print('all_deltas:', delta_copy)
    # print('prior width:', prior_width)
    # print('min vals:', min_vals)
    # print('max vals:', max_vals)

    kernel = NUTS(model, init_strategy = init_to_value(values = initial_values), target_accept_prob=0.95)
    # kernel = NUTS(model, init_strategy=init_to_uniform())
    # kernel = NUTS(model, init_strategy = init_to_median(num_samples = 1000))
    mcmc = MCMC(kernel, num_warmup=1000, num_samples=6000, num_chains=2, progress_bar=False) 
    # print('signs:', signs)
    chi2_val = chi2_fcn(initial_params, xvar, yvar, yerr, signs=signs)


    #print('Initial chi2:', chi2_val)


    with numpyro.validation_enabled():
    # res = mcmc.run(init_rng_key, data)
        res = mcmc.run(jax.random.PRNGKey(0), chi2_fcn, xvar, yvar, yerr, nparam, signs, params_copy, delta_copy, max_vals=max_vals, min_vals=min_vals)
    mcmc.print_summary()

    return mcmc

def get_mcmc_samples(fn, params, delta, num_sigma, likelihood, log_opt=True):
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
    mcmc = run_mcmc(chi2_fcn, likelihood.xvar, likelihood.yvar, likelihood.yerr, nparams, num_sigma, signs = signs, params = params, delta=delta, log_opt=log_opt)
    sampled_params = mcmc.get_samples()
    loglike = sampled_params.pop("loglike")

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

    return sampled_params, params, chi2_fcn, eq_numpy, likelihood


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
def read_data(name):
    files = os.listdir('esr/fitting/output/output_WL_' + name + '/')
    files = [f for f in files if f.startswith('final_') and f.endswith('.dat')]
    files = sorted(files, key=lambda x: int(x.split('_')[1].split('.')[0]))  # Sort by the number after 'final_'
    #print(files)

    funcs = []
    DL = []
    Prel = []
    params = []
    delta = []
    negloglike = []
    comp = []

    for file in files:
        #print(f'Processing file: {file}')
        file_path = os.path.join('esr/fitting/output/output_WL_' + name + '/', file)
        data = np.loadtxt(file_path, delimiter=';', dtype=str, ndmin=2)
        func = data[:, 1]
        DL_list = data[:, 2].astype(float)
        #caluclate prel
        Prel_list = data[:, 3].astype(float)
        params_list = data[:, 7:11].astype(float)
        delta_list = data[:, 11:14].astype(float)
        negloglike_lits = data[:, 4].astype(float)
        #comp is the number before .dat
        comp_number = int(file.split('_')[1].split('.')[0])
        comp.extend([comp_number]*len(func))
        # print(comp_number)
        #append to lists

        funcs.extend(func)
        DL.extend(DL_list)
        Prel.extend(Prel_list)
        params.extend(params_list)
        delta.extend(delta_list)
        negloglike.extend(negloglike_lits)

    funcs = np.array(funcs)
    DL = np.array(DL)
    Prel = np.array(Prel)
    params = np.array(params)
    delta = np.array(delta)
    negloglike = np.array(negloglike)
    #order list by DL
    sorted_indices = np.argsort(DL)
    funcs = funcs[sorted_indices]
    DL = DL[sorted_indices]
    Prel = Prel[sorted_indices]
    params = params[sorted_indices]
    delta = delta[sorted_indices]
    negloglike = negloglike[sorted_indices]
    comp = np.array(comp)[sorted_indices]

    # Choose a maximum acceptable DL difference (tune if needed)
    max_DL_diff = 50

    # Pre-slice arrays to only keep models within max_DL_diff
    good_mask = (DL - DL[0]) <= max_DL_diff
    DL = DL[good_mask]
    funcs = funcs[good_mask]
    negloglike = negloglike[good_mask]
    params = params[good_mask]
    delta = delta[good_mask]
    comp = np.array(comp)[good_mask]

    # Initialize Prel_DL with inf
    Prel_DL = np.full(len(negloglike), np.inf)

    # Use sets for fast duplicate checking
    seen_negloglike = set()
    seen_funcs = set()

    best_DL = DL[0]

    for i in range(len(negloglike)):
        # Early stopping (arrays are sorted, so once DL is too large, break)
        if DL[i] - best_DL > max_DL_diff:
            print(f"Stopped early at i={i}, DL diff = {DL[i] - best_DL:.2f}")
            break

        # Round negloglike for faster uniqueness checks
        nl_rounded = round(negloglike[i], 5)
        if nl_rounded in seen_negloglike or funcs[i] in seen_funcs:
            continue

        seen_negloglike.add(nl_rounded)
        seen_funcs.add(funcs[i])

        Prel_DL[i] = DL[i] - best_DL

    # Compute Prel only for valid entries
    Prel = np.exp(-Prel_DL)
    Prel[~np.isfinite(Prel)] = 0.0
    Prel /= np.sum(Prel)  # Normalize

    #print(f"Sum of Prel: {np.sum(Prel):.4f} (should be ~1.0)")
    #number of funcs until prel sums to 0.99
    cumulative_prel = np.cumsum(Prel)
    num_funcs_99 = np.searchsorted(cumulative_prel, 0.9) + 1
    print(f"Number of functions to reach 90% of total Prel: {num_funcs_99}")

    print(DL, comp)

    return funcs, DL, Prel, params, delta, negloglike, comp

# ----------------------------------MAIN----------------------------------
# Set up MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

#read clusters
names_all_clusters = 'all_clusters.txt'
# Load the names of all clusters
with open(names_all_clusters) as f:
    names = f.read().splitlines()

redshifts_file = 'cluster_redshifts.txt'
with open(redshifts_file) as f:
    lines = f.read().splitlines()

galaxy_redshifts = []
for line in lines:
    parts = line.split()
    galaxy_redshifts.append(float(parts[1]))

basis_functions = [["x", "a"], ["inv", "abs", "log", "exp", "re", "im"], ["+", "*", "-", "/", "pow"]]

#limit to first 20 clusters for testing
names = names[20:]
total_num_clusters = len(names)

#separate clusters in cores
clusters_per_core = total_num_clusters // size
remainder = total_num_clusters % size
if rank < remainder:
    start_index = rank * (clusters_per_core + 1)
    end_index = start_index + clusters_per_core + 1
else:
    start_index = rank * clusters_per_core + remainder
    end_index = start_index + clusters_per_core
clusters_for_this_core = names[start_index:end_index]

for i, cluster_name in enumerate(clusters_for_this_core):
    print(f"Core {rank} processing cluster {cluster_name} ({i+1}/{len(clusters_for_this_core)})")

    #read data
    z = galaxy_redshifts[names.index(cluster_name)]
    likelihood = WLLikelihood(f'XXL/{cluster_name}.txt', f'WL_{cluster_name}', data_dir=None, fn_set='core_maths')
    funcs, DL, Prel, params, delta, negloglike, comp = read_data(cluster_name)

    comm.Barrier()
    print('FINISH REDAING DATA') 
    mcmc_results = []
    sum_prel = 0
    #get samples for functions until sum_prel = 0.99
    for i in range(len(funcs)):
        if sum_prel < 0.9 and Prel[i] > 0.0:
            sum_prel += Prel[i]
            fcn_str = funcs[i]
            eq_numpy, k = get_eq_numpy(fcn_str, likelihood)
            
            if k == 0:
                print(f"Function {fcn_str} has 0 parameters, skipping MCMC...")
                result_dict = {
                'func': fcn_str,
                'params': np.array([]),
                'delta': np.array([]),
                'sampled_params': [np.array([])],
                'Prel': Prel[i],
                'function_index': i  # Add this to track which function it was
            }
                mcmc_results.append(result_dict)
                continue

            params_i = params[i, :k]
            uncertainties_i = delta[i, :k]
            params_uncertainties = np.concatenate([params_i, uncertainties_i])

            print(f"Processing function {i}: {fcn_str} for cluster {cluster_name} with Prel={Prel[i]:.4f}")

            try:
                sampled_params, params_mcmc, chi2_fcn, eq_numpy_mcmc, likelihood_mcmc = get_mcmc_samples(
                    fcn_str, params_i, uncertainties_i, num_sigma=5, likelihood=likelihood, log_opt=True
                )
            except Exception as e:
                print(f"Error running MCMC for func {fcn_str}: {e} for cluster {cluster_name}")
                try:
                    sampled_params, params_mcmc, chi2_fcn, eq_numpy_mcmc, likelihood_mcmc = get_mcmc_samples(
                    fcn_str, params_i, uncertainties_i, num_sigma=5, likelihood=likelihood, log_opt=False)
                except Exception as e:
                    print(f"Second attempt failed for func {fcn_str}: {e} for cluster {cluster_name}")
                    continue

            result_dict = {
                'func': fcn_str,
                'params': params_i,
                'delta': uncertainties_i,
                'sampled_params': sampled_params,
                'Prel': Prel[i],
                'function_index': i  # Add this to track which function it was
            }
            
            mcmc_results.append(result_dict)

    # Save all results for this cluster to a single file
    output_file = f'esr/fitting/output/masses/mcmc_results_cluster_{cluster_name}.npz'
    np.savez_compressed(output_file, mcmc_results=mcmc_results)
    print(f"Core {rank} saved MCMC results for cluster {cluster_name} to {output_file}")
    
    comm.Barrier()
    #now calculate mass and all that for each
    all_masses = []
    all_weights = []
    mass_samples = []
    mass_samples_by_func = []
    r200_list = []
    c200_list = []

    for result in mcmc_results:
        fcn_str = result['func']
        print('Processing function:', fcn_str)
        eq_numpy, k = get_eq_numpy(fcn_str, likelihood)
        sampled_params = result['sampled_params']
        Prel_i = result['Prel']
        function_index = result.get('function_index', -1)

        if k == 0:
            # No parameters, just evaluate once
            sampled_params = [np.array([])]

        mass_func = []
        r200_func = []
        c01_func = []
        r01_func = []
        valid_samples = []
        # Loop over all MCMC samples for this function
        for params_sample in sampled_params:
            #print the number of samples in chucks of 100 that are done
            if len(all_masses) % 500 == 0 and len(all_masses) > 0:
                print(f"Processed {len(all_masses)} samples so far...")
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

                # print(f"M200: {M200:.3e} M_sun")
            else:
                print(f"Failed to compute M200 for func {fcn_str} for cluster {cluster_name}")
                all_masses.append(np.nan)
                all_weights.append(Prel_i)
                mass_func.append(np.nan)
                r200_func.append(np.nan)
                c01_func.append(np.nan)
                r01_func.append(np.nan)


        func_result_dict = {
            'func': fcn_str,
            'function_index': function_index,
            'Prel': Prel_i,
            'original_params': result['params'],
            'original_delta': result['delta'],
            'n_total_samples': len(sampled_params),
            'n_valid_samples': len([m for m in mass_func if not np.isnan(m)]),
            
            # Mass and structural parameters
            'M200_samples': np.array(mass_func),
            'r200_samples': np.array(r200_func),
            'c01_samples': np.array(c01_func),
            'r01_samples': np.array(r01_func),
            
            # Valid MCMC parameter samples (only those that gave valid masses)
            'valid_mcmc_samples': np.array(valid_samples) if valid_samples else np.array([])}
        
        #save mass samples for this function
        mass_samples_by_func.append(func_result_dict)
    
    #save this cluster results
    output_file_mass = f'esr/fitting/output/masses/mass_results_cluster_{cluster_name}.npz'
    np.savez_compressed(output_file_mass, all_masses=np.array(all_masses), all_weights=np.array(all_weights), mass_samples_by_func=mass_samples_by_func)
    print(f"Core {rank} saved mass results for cluster {cluster_name} to {output_file_mass}")







