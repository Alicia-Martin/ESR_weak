import numpy as np
import jax.numpy as jnp
import jax
import numpyro
from numpyro import sample, factor, deterministic
from numpyro.distributions import Uniform
from esr.fitting.WL_likelihood import WLLikelihood
import numpyro.distributions as dist
from numpyro.infer.util import init_to_value, init_to_uniform
import sympy
from matplotlib import pyplot as plt
#import arviz as az
from numpyro.infer import MCMC, NUTS, HMC
from jax import random
import time
import corner
import esr.generation.simplifier as simplifier
import os
from esr.fitting.sympy_symbols import *
import jax.numpy as jnp
from jax import jit, grad
from scipy.interpolate import interp1d
from scipy.optimize import root_scalar
import numpy as np
from scipy.optimize import fsolve
from mpi4py import MPI
import pickle
from io import StringIO
import sys

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

jax.config.update("jax_enable_x64", True)

# ================================= FUNCTIONS =================================

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

def model(chi2_fcn, xvar, yvar, yerr, nparam, signs, params, delta, max_vals=None, min_vals=None):
    p = []
    for i in range(nparam):
        min_val_log = min_vals[i]
        max_val_log = max_vals[i]
        ai = numpyro.sample(f'log_a{i}', dist.Uniform(min_val_log, max_val_log))
        p.append(ai)

    def negloglike(par, *args, signs=None):
        val, g = chi2_fcn(par, *args, signs)
        return val, g

    negloglike, grad_nll = negloglike(p, xvar, yvar, yerr, signs=signs)
    deterministic("loglike", negloglike)
    numpyro.factor('negloglike', - negloglike)

def run_mcmc(chi2_fcn, xvar, yvar, yerr, nparam, num_sigma, signs, params, delta, log_opt=True):
    params_copy = np.copy(params)
    delta_copy = np.copy(delta)
    initial_values = {}
    
    if log_opt:
        signs[params_copy == 0] = 1
        params_copy[params_copy == 0.0] = -9.99809130e-07
        delta_copy = delta_copy / np.abs(params_copy) / np.log(10)
        delta_copy[delta_copy == 0.0] = 3
        params_copy = np.log10(np.abs(params_copy))
    else:
        delta_copy[delta_copy == 0.0] = 3

    for i in range(nparam):
        initial_values[f'log_a{i}'] = params_copy[i]

    prior_width = num_sigma * delta_copy
    max_vals = []
    min_vals = []
    for i in range(nparam):
        log_a_centre = initial_values[f'log_a{i}']
        min_val_log = log_a_centre - prior_width[i]
        max_val_log = log_a_centre + prior_width[i]
        min_vals.append(min_val_log)
        max_vals.append(max_val_log)

    initial_params = np.array([initial_values[f'log_a{i}'] for i in range(nparam)])

    kernel = NUTS(model, init_strategy=init_to_value(values=initial_values), target_accept_prob=0.95)
    mcmc = MCMC(kernel, num_warmup=1000, num_samples=3500, num_chains=2, progress_bar = False)
    
    chi2_val = chi2_fcn(initial_params, xvar, yvar, yerr, signs=signs)
    #print(f'Rank {rank}: Initial chi2: {chi2_val}')

    # Suppress MCMC print output by redirecting stdout temporarily
    
    
    #try:
    with numpyro.validation_enabled():
        res = mcmc.run(jax.random.PRNGKey(0), chi2_fcn, xvar, yvar, yerr, nparam, signs, params_copy, delta_copy, max_vals=max_vals, min_vals=min_vals)
    mcmc.print_summary()
    #finally:
    #    sys.stdout = old_stdout
    
    return mcmc

def get_eq_numpy(fn, likelihood):
    fcn_i = fn.replace('\'', '')
    max_param = 4
    k = simplifier.count_params([fcn_i], max_param)[0]
    fcn_i, eq = likelihood.run_sympify(fcn_i)

    
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
        if n_extra > 0:
            all_a += sympy.symbols("rho0 rs", real=True)
        eq_numpy = sympy.lambdify([x] + all_a, eq, modules=["jax"])
    elif n_fun_params == 1:
        if n_extra > 0:
            rho0, rs = sympy.symbols("rho0 rs", real=True)
            eq_numpy = sympy.lambdify([x, a0, rho0, rs], eq, modules=["jax"])
        else:
            eq_numpy = sympy.lambdify([x, a0], eq, modules=["jax"])

    return eq_numpy, k

def get_M_delta_2(delta, fcn_i, params_uncertainties, z, k, tol=1e-2, r_min=1e-6, n_points=1000):
    params = params_uncertainties[:k]
    
    # Constants
    G = 6.67430e-11
    H0 = 70.0 * 1e3 / (3.086e22)

    def rho_crit(z=0):
        H_z = H0 * np.sqrt(0.3 * (1 + z)**3 + 0.7)
        rho_crit_si = (3 * H_z**2) / (8 * np.pi * G)
        M_sun = 1.98847e30
        Mpc = 3.085677581e22
        conversion_factor = (Mpc**3) / M_sun
        return rho_crit_si * conversion_factor

    def make_rho_profile(fcn_i):
        @jit
        def rho_profile(r, params):
            return fcn_i(r, *params) * 1e12
        return rho_profile
    
    def make_mass_enclosed_grid(rho_profile):
        @jit
        def mass_enclosed_grid(r_vals, params):
            rho_vals = rho_profile(r_vals, params)
            integrand = 4 * jnp.pi * rho_vals * r_vals**2
            mass_cumulative = jnp.cumsum(
                0.5 * (integrand[1:] + integrand[:-1]) * jnp.diff(r_vals)
            )
            mass_cumulative = jnp.concatenate([jnp.array([0.0]), mass_cumulative])
            return mass_cumulative
        return mass_enclosed_grid

    r_vals = np.logspace(np.log10(r_min), 1, n_points)
    rho_profile = make_rho_profile(fcn_i)
    mass_enclosed_grid = make_mass_enclosed_grid(rho_profile)
    
    try:
        M_vals = mass_enclosed_grid(r_vals, params)
        M_interp = lambda r: jnp.interp(r, r_vals, M_vals)
        rho_c = rho_crit(z)

        def to_solve(r):
            M_r = M_interp(r)
            rho_mean = M_r / ((4/3) * np.pi * r**3)
            return rho_mean - delta * rho_c

        def safe_root_solve(func, bracket=None, x0=1.0, tol=1e-2, check_func=None, target_value=None):
            def check_root(r):
                if check_func is None or target_value is None:
                    return np.isclose(func(r), 0, atol=tol*abs(target_value) if target_value else tol)
                return np.abs(check_func(r) - target_value) / target_value < tol
            
            if bracket is not None:
                if np.sign(func(bracket[0])) != np.sign(func(bracket[1])):
                    try:
                        sol = root_scalar(func, bracket=bracket, method="brentq")
                        if check_root(sol.root):
                            return sol.root
                    except Exception:
                        pass
            
            r_solution = fsolve(func, x0=x0)[0]
            if check_root(r_solution):
                return r_solution
            return None

        r200 = safe_root_solve(
            to_solve, bracket=[1e-4, 20], x0=1,
            check_func=lambda r: M_interp(r) / ((4/3) * np.pi * r**3),
            target_value=delta * rho_c, tol=1e-2
        )
        
        if r200 is None:
            return None

        M200 = (4/3) * np.pi * r200**3 * (delta * rho_c)

        target_mass = 0.1 * M200
        r01 = safe_root_solve(
            lambda r: M_interp(r) - target_mass,
            bracket=[r_vals[0], r200], x0=r200/10,
            check_func=M_interp, target_value=target_mass, tol=1e-2
        )
        
        if r01 is None:
            return None

        c01 = r200 / r01
        return M200, r200, c01, r01
        
    except Exception as e:
        return None

def compute_mass_robustness(fcn_i, params_uncertainties, z, k,
                             cutoffs=(1e-6, 1e-4, 1e-3, 1e-2, 1e-1),
                             grid_npoints=(1000, 4000), dex_threshold=0.1):
    """Robustness diagnostic for a single (ML) parameter point: how much log10(M200)
    moves as the inner integration cutoff varies over a fixed, cluster-independent
    span (cutoff_dex_shift), separately from pure grid-resolution error at fixed
    cutoff (grid_dex_shift). A cluster whose mass is set by the data (not by where
    the integral is truncated) barely moves under the cutoff sweep; a cluster whose
    fit has collapsed to a tiny/degenerate scale radius (rs -> 0, pure r^-3) has its
    mass dominated by unconstrained inward extrapolation and swings by ~1 dex or more.
    mass_robust = both shifts <= dex_threshold (0.1 dex, i.e. ~26% in M200)."""
    M200_by_cutoff = {}
    for cutoff in cutoffs:
        res = get_M_delta_2(200, fcn_i, params_uncertainties, z, k, r_min=cutoff)
        if res is not None:
            M200_by_cutoff[cutoff] = float(res[0])

    if len(M200_by_cutoff) < 2:
        return {'M200_by_cutoff': M200_by_cutoff, 'cutoff_dex_shift': np.nan,
                'grid_dex_shift': np.nan, 'mass_robust': False}

    log10_M200 = np.log10(list(M200_by_cutoff.values()))
    cutoff_dex_shift = float(np.max(log10_M200) - np.min(log10_M200))

    default_cutoff = cutoffs[0]
    res_lo = get_M_delta_2(200, fcn_i, params_uncertainties, z, k, r_min=default_cutoff, n_points=grid_npoints[0])
    res_hi = get_M_delta_2(200, fcn_i, params_uncertainties, z, k, r_min=default_cutoff, n_points=grid_npoints[1])
    if res_lo is not None and res_hi is not None:
        grid_dex_shift = float(abs(np.log10(res_hi[0]) - np.log10(res_lo[0])))
    else:
        grid_dex_shift = np.nan

    mass_robust = bool(cutoff_dex_shift <= dex_threshold and
                        (not np.isnan(grid_dex_shift)) and grid_dex_shift <= dex_threshold)

    return {'M200_by_cutoff': M200_by_cutoff, 'cutoff_dex_shift': cutoff_dex_shift,
            'grid_dex_shift': grid_dex_shift, 'mass_robust': mass_robust}

def process_single_cluster(cluster_name, cluster_index, nfw_params, nfw_deltas, galaxy_redshifts, fcn_str):
    """Process a single cluster: run MCMC and calculate masses immediately"""
    
    print(f"Rank {rank}: Processing cluster {cluster_name}")
    
    # Setup likelihood
    data_file = f'XXL/{cluster_name}.txt'
    if not os.path.exists(data_file):
        print(f"Rank {rank}: Data file {data_file} not found, skipping cluster {cluster_name}")
        return None
    
    try:
        likelihood = WLLikelihood(data_file, f'WL_{cluster_name}', data_dir=None, 
                                 fn_set='core_maths')
    except Exception as e:
        print(f"Rank {rank}: Error creating likelihood for {cluster_name}: {e}")
        return None
    
    # Prepare parameters
    params = np.array([nfw_params[0], nfw_params[1]])
    delta = np.array([nfw_deltas[0], nfw_deltas[1]])
    
    # Handle zero parameters
    mask = params == 0
    params[mask] = 1e-10
    delta[mask] = 5*1e-10
    
    # Get equation
    eq_numpy, k = get_eq_numpy(fcn_str, likelihood)
    
    # Setup MCMC
    n_fun_params = k
    n_extra = 0

    loss_template = likelihood.get_loss(eq_numpy)
    chi2_fcn = likelihood.get_wrapped_like(loss_template)
    nparams = n_fun_params + n_extra
    signs = np.sign(params)[:n_fun_params]
    
    params_mcmc = np.array(params[:n_fun_params])
    delta_mcmc = np.array(delta[:n_fun_params])

    # a1 is rs for this fixed NFW form (fcn_str above), so the inner integration
    # limit used for the per-sample mass loop can be tied to it. Never coarser
    # than the old fixed 1e-6 floor; only kicks in for clusters whose ML rs is
    # small enough that 1e-6 sits at/above rs, which breaks the mass_enclosed_grid
    # assumption that M(r_vals[0]) = 0. Exception: if a1's true fit was exactly 0
    # (mask[1], before the 1e-10 placeholder substitution above), rs isn't a real
    # measurement -- keep the old floor rather than tying it to a placeholder.
    a1_ml = params_mcmc[1]
    if mask[1]:
        r_min_100 = 1e-6
        print(f"Rank {rank}: {cluster_name} has unconstrained rs (a1 fit to exactly 0); "
              f"keeping r_min=1e-6 rather than tying it to the 1e-10 placeholder")
    else:
        r_min_100 = min(1e-6, np.abs(a1_ml) / 100)

    # Mass robustness check at the ML point: how much log10(M200) moves under a
    # fixed, cluster-independent sweep of the inner integration cutoff, separately
    # from pure grid-resolution error. Matches the cutoff_dex_shift / grid_dex_shift
    # / mass_robust diagnostic documented in dm-esr/HSC-CLASH/NOTES.md (2026-07-14):
    # a >0.1 dex swing flags a fit that has collapsed to a degenerate rs->0 (pure
    # r^-3) solution, whose mass is set by unconstrained inward extrapolation, not
    # by the data.
    ml_uncertainties = np.zeros_like(params_mcmc)
    ml_params_uncertainties = np.concatenate([params_mcmc, ml_uncertainties])
    robustness = compute_mass_robustness(eq_numpy, ml_params_uncertainties, galaxy_redshifts[cluster_index], k)
    print(f"Rank {rank}: {cluster_name} ML rs={a1_ml:.3e}, cutoff_dex_shift={robustness['cutoff_dex_shift']:.3f}, "
          f"grid_dex_shift={robustness['grid_dex_shift']:.2e}, mass_robust={robustness['mass_robust']}")
    if not robustness['mass_robust']:
        print(f"Rank {rank}: WARNING - {cluster_name} M200 is NOT robust to the inner integration cutoff "
              f"(cutoff_dex_shift={robustness['cutoff_dex_shift']:.3f} > 0.1 dex) -- likely a degenerate "
              f"rs->0 fit; see calculate_M200_c200_nfw_parallel.py for a physically-bounded alternative")

    try:
        # Run MCMC
        print(f"Rank {rank}: Starting MCMC for cluster {cluster_name}")
        mcmc = run_mcmc(chi2_fcn, likelihood.xvar, likelihood.yvar, likelihood.yerr, 
                       nparams, num_sigma=3, signs=signs, params=params_mcmc, 
                       delta=delta_mcmc, log_opt=True)
        
        # Check convergence and print R-hat values
        
        # Get samples
        sampled_params = mcmc.get_samples()
        loglike = sampled_params.pop("loglike")
        
        # Convert back from log space
        for i in range(n_fun_params):
            sampled_params[f'log_a{i}'] = 10**(sampled_params[f'log_a{i}']) * signs[i]
        
        param_names = [f'log_a{i}' for i in range(nparams)]
        sampled_params_array = np.column_stack([sampled_params[name] for name in param_names])
        
        print(f"Rank {rank}: MCMC completed for cluster {cluster_name}, got {len(sampled_params_array)} samples")
        
        # Calculate masses immediately
        print(f"Rank {rank}: Calculating masses for cluster {cluster_name}")
        z = galaxy_redshifts[cluster_index]
        
        mass_func = []
        r200_func = []
        c01_func = []
        r01_func = []
        valid_samples = []
        
        total_samples = len(sampled_params_array)
        valid_count = 0
        
        for i, params_sample in enumerate(sampled_params_array):
            params_i = params_sample[:k]
            uncertainties_i = np.zeros_like(params_i)
            params_uncertainties = np.concatenate([params_i, uncertainties_i])
            
            mass_result = get_M_delta_2(200, eq_numpy, params_uncertainties, z, k, r_min=r_min_100)
            
            if mass_result is not None:
                M200, r200, c01, r01 = mass_result
                mass_func.append(M200)
                r200_func.append(r200)
                c01_func.append(c01)
                r01_func.append(r01)
                valid_samples.append(params_sample)
                valid_count += 1
            else:
                mass_func.append(np.nan)
                r200_func.append(np.nan)
                c01_func.append(np.nan)
                r01_func.append(np.nan)
        
        print(f"Rank {rank}: Mass calculation completed. Valid samples: {valid_count}/{total_samples}")
        
        # Create result dictionary
        result_dict = {
            'name': cluster_name,
            'index': cluster_index,
            'original_params': params,
            'original_delta': delta,
            'n_total_samples': len(sampled_params_array),
            'n_valid_samples': valid_count,
            'M200_samples': np.array(mass_func),
            'r200_samples': np.array(r200_func),
            'c01_samples': np.array(c01_func),
            'r01_samples': np.array(r01_func),
            'valid_mcmc_samples': np.array(valid_samples) if valid_samples else np.array([]),
            'mcmc_samples_all': sampled_params_array,
            'redshift': z,
            'M200_by_cutoff': robustness['M200_by_cutoff'],
            'cutoff_dex_shift': robustness['cutoff_dex_shift'],
            'grid_dex_shift': robustness['grid_dex_shift'],
            'mass_robust': robustness['mass_robust'],
        }

        # Calculate and print statistics
        valid_masses = result_dict['M200_samples'][~np.isnan(result_dict['M200_samples'])]
        if len(valid_masses) > 0:
            median_mass = np.median(valid_masses)
            mass_16th = np.percentile(valid_masses, 16)
            mass_84th = np.percentile(valid_masses, 84)
            print(f"Rank {rank}: Results for cluster {cluster_name}:")
            print(f"  Median M200: {median_mass:.3e} M_sun (+{mass_84th - median_mass:.3e}/-{median_mass - mass_16th:.3e})")
            print(f"  Mass robust: {result_dict['mass_robust']} (cutoff_dex_shift={result_dict['cutoff_dex_shift']:.3f})")
        
        # Save results immediately for this cluster
        output_dir = f'results/nfw_individual'
        os.makedirs(output_dir, exist_ok=True)
        
        output_file = f'{output_dir}/nfw_results_{cluster_name}_rank_{rank}.npz'
        np.savez_compressed(output_file, **result_dict)
        print(f"Rank {rank}: Saved results for cluster {cluster_name}")
        
        # Create plots (suppressed plotting for faster execution)
        # Uncomment if you want plots
        # if len(sampled_params_array) > 0:
        #     plt.figure(figsize=(10, 10))
        #     if n_fun_params == 2:
        #         corner.corner(sampled_params, labels=['a0', 'a1'], 
        #                     truths=np.log10(np.abs(params_mcmc)), 
        #                     show_titles=True, title_kwargs={"fontsize": 12}, 
        #                     color='blue', smooth=True, smooth1d=True)
        #     elif n_fun_params == 1:
        #         corner.corner(sampled_params, labels=['a0'], 
        #                     truths=np.log10(np.abs(params_mcmc)), 
        #                     show_titles=True, title_kwargs={"fontsize": 12}, 
        #                     color='blue', smooth=True, smooth1d=True)
        #     
        #     plot_dir = f'../results/nfw_plots'
        #     os.makedirs(plot_dir, exist_ok=True)
        #     plt.savefig(f'{plot_dir}/corner_plot_{cluster_name}_nfw_rank_{rank}.png')
        #     plt.close()
        
        return result_dict
        
    except Exception as e:
        print(f"Rank {rank}: Error processing cluster {cluster_name}: {e}")
        return None

# ================================= MAIN EXECUTION =================================

def main():
    # Load cluster data
    print(f"Rank {rank}: Loading cluster data")
    
    # Load NFW parameters
    nfw_file = '1_(a0_(pow(Abs(a1) + x,2)_x))_2.txt'
    nfw_data = np.loadtxt(nfw_file, dtype=str, delimiter=' ')
    nfw_a0 = nfw_data[:,4].astype(float)
    nfw_a1 = nfw_data[:,5].astype(float)
    nfw_d0 = nfw_data[:,6].astype(float)
    nfw_d1 = nfw_data[:,7].astype(float)
    
    # Load cluster names
    names_all_clusters = 'all_clusters.txt'
    with open(names_all_clusters) as f:
        names = f.read().splitlines()
    
    # Load redshifts
    redshifts_file = 'cluster_redshifts.txt'
    with open(redshifts_file) as f:
        lines = f.read().splitlines()
    
    galaxy_redshifts = []
    for line in lines:
        parts = line.split()
        galaxy_redshifts.append(float(parts[1]))
    
    # NFW function
    fcn_str = '1/(a0*(pow(Abs(a1) + x,2)*x))'
    
    # Distribute clusters among ranks
    total_clusters = len(names)
    clusters_per_rank = total_clusters // size
    remainder = total_clusters % size
    
    if rank < remainder:
        start_idx = rank * (clusters_per_rank + 1)
        end_idx = start_idx + clusters_per_rank + 1
    else:
        start_idx = rank * clusters_per_rank + remainder
        end_idx = start_idx + clusters_per_rank
    
    my_clusters = names[start_idx:end_idx]
    print(f"Rank {rank}: Processing {len(my_clusters)} clusters")
    
    # Process clusters assigned to this rank
    results_this_rank = []
    
    for i, cluster_name in enumerate(my_clusters):
        global_index = start_idx + i
        print(f"Rank {rank}: Starting cluster {cluster_name} ({i+1}/{len(my_clusters)})")
        
        nfw_params = [nfw_a0[global_index], nfw_a1[global_index]]
        nfw_deltas = [nfw_d0[global_index], nfw_d1[global_index]]
        
        result = process_single_cluster(cluster_name, global_index, nfw_params, 
                                      nfw_deltas, galaxy_redshifts, fcn_str)
        
        if result is not None:
            results_this_rank.append(result)
        
        print(f"Rank {rank}: Completed cluster {cluster_name}")
        print("-" * 50)  # Separator between clusters
    
    # Gather all results
    print(f"Rank {rank}: Gathering results from all ranks")
    all_results = comm.gather(results_this_rank, root=0)
    
    # Root process saves combined results
    if rank == 0:
        print("Root: Combining and saving all results")
        combined_results = []
        for rank_results in all_results:
            if rank_results is not None:
                combined_results.extend(rank_results)
        
        # Save combined results
        output_file = 'results/nfw_mcmc_mass_results_combined.npz'
        np.savez_compressed(output_file, mass_samples_by_func=combined_results)
        print(f"Root: Saved combined results to {output_file}")
        print(f"Root: Total successful clusters: {len(combined_results)}")
        
        # Print robustness summary
        robust_count = sum(1 for result in combined_results if result.get('mass_robust', False))
        print(f"Root: Mass robustness summary: {robust_count}/{len(combined_results)} clusters have a robust M200")

        # Save summary statistics
        summary_file = 'results/nfw_summary_statistics.txt'
        with open(summary_file, 'w') as f:
            f.write("NFW Mass Analysis Summary\n")
            f.write("=" * 50 + "\n")
            f.write(f"Total clusters processed: {len(combined_results)}\n")
            f.write(f"Mass-robust clusters: {robust_count}\n")

            for result in combined_results:
                if result['n_valid_samples'] > 0:
                    valid_masses = result['M200_samples'][~np.isnan(result['M200_samples'])]
                    median_mass = np.median(valid_masses)
                    mass_16th = np.percentile(valid_masses, 16)
                    mass_84th = np.percentile(valid_masses, 84)

                    f.write(f"\nCluster: {result['name']}\n")
                    f.write(f"  Redshift: {result['redshift']:.3f}\n")
                    f.write(f"  Mass robust: {result.get('mass_robust', 'Unknown')} "
                            f"(cutoff_dex_shift={result.get('cutoff_dex_shift', float('nan')):.3f})\n")
                    f.write(f"  Valid samples: {result['n_valid_samples']}/{result['n_total_samples']}\n")
                    f.write(f"  Median M200: {median_mass:.3e} M_sun\n")
                    f.write(f"  16th-84th percentile: [{mass_16th:.3e}, {mass_84th:.3e}] M_sun\n")
        
        print(f"Root: Summary statistics saved to {summary_file}")
    
    print(f"Rank {rank}: Process completed")

if __name__ == "__main__":
    main()
