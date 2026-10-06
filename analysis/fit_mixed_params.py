#global local params

import sys
import numpy as np

from esr.fitting.test_all import mixed_optimise_fun
from esr.fitting.test_all_Fisher import convert_params

import esr.generation.generator as generator
import esr.generation.simplifier as simplifier
import sympy

import math
import jax
import jax.numpy as jnp
from jax import vmap
import scipy
import scipy.optimize
import matplotlib.pyplot as plt
from esr.fitting.sympy_symbols import *
from esr.fitting.WL_likelihood import WLLikelihood
import numdifftools as nd
from esr.esd import ExcessSurfaceDensity
import ast
from mpi4py import MPI
import time
import os

sys.path.append(os.path.abspath("katz"))  
from katz.prior import KatzPrior

from esr.fitting.fit_single import fit_from_string as fit_from_string_local

from prettytable import PrettyTable

import itertools

def get_sigma_from_integral(p, Sigma, fcn_i, negloglike_all, nparams, fop, number_points=10**3):
        def fraction_likelihood(x, p, j, factor):
            def compute_negloglike(params):
                return jnp.abs(fop(params) - negloglike_all - jnp.log(1e3))
            def inf_value(_):
                # jax.debug.print('aqui')
                return 0.0999
            
            # Modify parameter
            params = p.at[j].set(p[j] + factor * 10**x[0])
            
            negloglike = jax.lax.cond(jnp.isinf(fop(params)), inf_value, compute_negloglike, params)
            return negloglike

        def get_loss():
            return jax.jit(fraction_likelihood)

        def wrap_loss(fraction_likelihood):
            def loss(x, p, j, factor):
                # print(fraction_likelihood(x, p, j, factor))
                return fraction_likelihood(x, p, j, factor)
            return loss
            
        def get_boundary(res):
            return 10**res.x[0]
        
        def get_integral(theta, boundary, param, fop, negloglike, number_points):
            def compute_like(theta):
                return jnp.exp(-fop(theta) + negloglike)

            boundary_range = np.array([-boundary, boundary])
            # Loop through left and right boundaries
            integrals = []
            for boundary_shift in boundary_range:
                a_range = np.linspace(theta, theta + boundary_shift, number_points)
                thetas = np.tile(param, (len(a_range), 1))
                thetas[:, j] = a_range
                like = vmap(compute_like)(thetas)

                # Integrate
                integral = scipy.integrate.cumtrapz(like, a_range)
                integrals.append(integral)

            # Combine left and right integrals
            integral_left, integral_right = [np.array(integral) for integral in integrals]
            integral = - integral_left + integral_right

            return integral, a_range
        
        def test_success(res):
            if res.success == False or res.fun > 0.1:
                return False
            else:
                return True

        param = p[:nparams]

        # initial_guesses = [np.ones(nparams), np.ones(nparams)*(-1)]
        arg_integral = np.append(np.argwhere(Sigma < 0), np.argwhere(Sigma == np.inf))
        arg_integral = np.append(arg_integral, np.argwhere(np.isnan(Sigma)))

        loss_template = get_loss()
        likelihood_fcn = wrap_loss(loss_template)
        
        for j in arg_integral:
                theta = param[j]

                initial_guesses = [-20, -10, -5, 0.5, 1, 5, np.log10(np.abs(theta))]
                initial_guesses = np.sort(initial_guesses)

                #Plot likelihood
                # Delta_plot = 150
                # x_range = np.linspace(theta - Delta_plot, theta + Delta_plot, 10**2)
                # # x_range = np.append(x_range, theta)
                # params_range = np.tile(param, (len(x_range), 1))
                # params_range[:, j] = x_range

                # nll = []
                # for params in params_range:
                #     negloglike = fop(params)
                #     nll = np.append(nll, negloglike)        
                # plt.plot(x_range, np.exp(-nll + jnp.min(nll)))
                # plt.yscale('log')
                # # plt.plot(theta_ML, np.exp(-chi2_fcn(theta_ML, xvar, yvar, yerr) + jnp.min(nll)), 'ro')
                # plt.show()
                # sys.exit()

                for factor in [1, -1]:
                    for initial_guess in initial_guesses:
                    # print('initial_guess', initial_guess, flush=True)
                        res = scipy.optimize.minimize(likelihood_fcn, initial_guess, args=(param, j, factor), method='Nelder-Mead', tol=1e-8)
                        # print('res', res, flush=True)
                        # res_minus = scipy.optimize.minimize(fraction_likelihood, initial_guess[j], args=(param, j, False), method='Nelder-Mead', tol=1e-8)
                        # res_plus = scipy.optimize.minimize(fraction_likelihood, initial_guess[j], args=(param, j, True), method = 'Nelder-Mead', tol=1e-8)

                        boundary_found = False
                        if test_success(res)== True:
                            if factor == -1:
                                boundary_left = get_boundary(res)
                            else:
                                boundary_right = get_boundary(res)
                            boundary_found = True
                            break

                    if not boundary_found:
                        break

                # print('boundary_left', boundary_left, flush=True)
                # print('boundary_right', boundary_right, flush=True)

                if not boundary_found:
                    #print(f"Couldn't find integral limits for factor {factor}, function {fcn_i}, parameter {theta}", flush=True)
                    Sigma[j] = np.inf
                    continue
            
                #do integral
                boundary = np.max([boundary_left, boundary_right])
                integral, a_range = get_integral(theta, boundary, param, fop, negloglike_all, number_points=number_points)
                                        
                #get the 68% confidence interval from the integral
                arg_min = np.argmin(abs(0.68 - integral/integral[-1]))
                param68 = a_range[arg_min + 1]
                # print(param68)
                sigma = np.abs(theta - param68)
                Sigma[j] = sigma

        return Sigma

def get_fisher_diag_from_deriv(deriv, nparam, max_param=4):
    Hmat_max = np.zeros((max_param, max_param))
    triu_indices = np.triu_indices(max_param)
    Hmat_max[triu_indices] = deriv
    fisher_diag = np.diag(Hmat_max[:nparam, :nparam])
    
    return fisher_diag

def convert_params2(eq_numpy, likelihood,xvar, yvar, L_factor, theta_ML, nparams, max_param=4):
    def get_deriv(Hmat, nparam_fun, max_param=4):

        Hmat_max = np.zeros((max_param,max_param))
        Hmat_max[:nparam_fun, :nparam_fun] = Hmat[:nparam_fun, :nparam_fun]

        deriv = Hmat_max[np.triu_indices(max_param)]
        return deriv
    
    #CHANGE THIS
    theta_ML = theta_ML[:nparams]
    hessian_template = likelihood.get_loss(eq_numpy, value = 'hessian')
    Hmat = hessian_template(theta_ML, xvar,yvar, L_factor)
    
    #Other related quantities
    Fisher_diag = jnp.diag(Hmat)
    deriv = get_deriv(Hmat, nparams,  max_param=max_param)

    return deriv, Fisher_diag

def snap_params(negloglike_orig, params_orig,  Delta_orig, Nsteps, nparams, fop):
    p = np.copy(params_orig)
    ptrue = np.copy(params_orig)
    Delta = np.copy(Delta_orig)

    try:
        p[Nsteps<1] = 0. 
    except (IndexError, TypeError):
        p[:]=0.


    #first try setting all parameters to zero      
    negloglike = fop(p)
                
    if np.isfinite(negloglike):
        mask = Nsteps<1
        params = p
        params[mask] = 0
        Delta[mask] = 0

    else:
        # Let's see if setting any of the parameters to zero is ok
        number_params = len(p)
        #print('number_params', number_params)
        try_idx = np.arange(number_params)[Nsteps < 1]
        for r in reversed(range(1, len(try_idx))):
            for idx in itertools.combinations(try_idx, r):
                p = np.copy(ptrue)
                for idx_ in idx:
                    p[idx_] = 0.

                negloglike = fop(p)

                if np.isfinite(negloglike):
                    # If valid, update k and break out of loop
                    # kept_mask = np.ones(len(p), dtype=bool)  # Keep all initially
                    # kept_mask[list(idx)] = False # Exclude params set to zero
                    Delta[Nsteps<1] = abs(p[Nsteps<1]) # the rest of the params with Nsteps<1 are set to the absolute value of the parameter
                    break
    
        if np.isfinite(negloglike):
            params = p 
            # params = p[kept_mask]
            # Delta = Delta[kept_mask]
        # elif not np.isfinite(negloglike_all[i]) and not np.isnan(negloglike_all[i]): # infinite nll
        elif not np.isfinite(negloglike):
            params = ptrue
            Delta[Nsteps<1] = abs(ptrue[Nsteps<1])
            negloglike = negloglike_orig

        #print('params', params)
        #print('Delta', Delta)
        #print('negloglike', negloglike)
    
    return negloglike, params, Delta

def codelen_from_vector(p, Delta, active_idx):
    """Parametric codelen over the active params only:
    len(use)*log(2) + sum(log(|p|/Delta)), counting only params that are
    nonzero with a finite, positive Delta. Ported from MIGHTEE match.py."""
    p = np.asarray(p, dtype=float)
    Delta = np.asarray(Delta, dtype=float)
    active_idx = np.asarray(active_idx, dtype=int)

    use = []
    for j in active_idx:
        if j < 0 or j >= len(p):
            continue
        if np.isfinite(p[j]) and np.isfinite(Delta[j]) and (Delta[j] > 0) and (p[j] != 0):
            use.append(j)

    if len(use) == 0:
        return 0.0

    use = np.asarray(use, dtype=int)
    return len(use) * math.log(2.0) + float(np.sum(np.log(np.abs(p[use]) / Delta[use])))


def snap_with_dl_test(p, Delta, active_idx, snap_allowed_idx, fop, negloglike_orig):
    """MIGHTEE-style snapping with a total-DL test.

    Baseline = do NOT snap: unresolved active params (Nsteps = |p|/Delta < 1)
    have their Delta capped to |p|, so each still contributes log(2) to the
    codelen (log(|p|/|p|) = 0). Then try snapping every unresolved param in
    ``snap_allowed_idx`` to zero (all at once, then every strict subset) and
    keep a snap only if the total DL (negloglike + codelen) strictly improves
    by more than 1e-8. Otherwise the params are kept (baseline).

    Returns (best_p, best_negloglike, best_codelen, best_Delta).
    """
    p = np.asarray(p, dtype=float).copy()
    Delta = np.asarray(Delta, dtype=float).copy()
    active_idx = np.asarray(active_idx, dtype=int)
    snap_allowed_idx = np.asarray(snap_allowed_idx, dtype=int)

    def nsteps(j):
        if not (np.isfinite(Delta[j]) and Delta[j] > 0):
            return np.inf
        return abs(p[j]) / Delta[j]

    # Baseline Delta: cap unresolved active params at |p| -> each contributes log(2).
    Delta_base = Delta.copy()
    for j in active_idx:
        if nsteps(j) < 1:
            Delta_base[j] = abs(p[j])

    # Params that may actually be snapped to zero.
    snap_idx = np.array([j for j in snap_allowed_idx if nsteps(j) < 1], dtype=int)

    def eval_total(p_cand, Delta_cand):
        neglog = float(fop(np.asarray(p_cand, dtype=float)))
        if not np.isfinite(neglog):
            return np.inf, np.inf, np.inf
        code = codelen_from_vector(p_cand, Delta_cand, active_idx)
        return neglog, code, neglog + code

    # Baseline: no snap, capped Delta.
    best_p = p.copy()
    best_Delta = Delta_base.copy()
    best_neglog, best_code, best_total = eval_total(best_p, best_Delta)
    if not np.isfinite(best_total):
        best_neglog = float(negloglike_orig)

    if len(snap_idx) > 0:
        # Candidate 1: snap all unresolved params.
        p_all = p.copy();          p_all[snap_idx] = 0.0
        D_all = Delta_base.copy(); D_all[snap_idx] = 0.0
        nl, cd, tot = eval_total(p_all, D_all)
        if np.isfinite(tot) and tot < best_total - 1e-8:
            best_p, best_Delta, best_neglog, best_code, best_total = p_all, D_all, nl, cd, tot

        # Candidate 2: every strict subset of unresolved params.
        if len(snap_idx) > 1:
            for r in range(len(snap_idx) - 1, 0, -1):
                for comb in itertools.combinations(snap_idx, r):
                    comb = list(comb)
                    p_t = p.copy();          p_t[comb] = 0.0
                    D_t = Delta_base.copy(); D_t[comb] = 0.0
                    nl, cd, tot = eval_total(p_t, D_t)
                    if np.isfinite(tot) and tot < best_total - 1e-8:
                        best_p, best_Delta, best_neglog, best_code, best_total = p_t, D_t, nl, cd, tot

    return best_p, best_neglog, best_code, best_Delta


def single_function(labels, basis_functions, likelihood, global_index, method, pmin=0, pmax=5, tmax=5,
    try_integration=False, verbose=False, Niter=30, Nconv=5, log_opt=False,
    return_params=False):

    # (1) Convert the string to a sympy function
    s = generator.labels_to_shape(labels, basis_functions)
    success, _, tree = generator.check_tree(s)
    fstr = generator.node_to_string(0, tree, labels)
    max_param = simplifier.get_max_param([fstr], verbose=verbose)
    fstr, fsym = simplifier.initial_sympify(
        [fstr], max_param, parallel=False, verbose=verbose)
    fstr = fstr[0]
    fsym = fsym[fstr]
    nparams = simplifier.count_params([fstr], max_param)[0]
    n_fun_params = nparams

    # (1.5) Get the global and local indices
    # global_index = [0]
    n_global = len(global_index)
    n_local = nparams - n_global
    n_extra = 0
    #points_per_cluster = 10

    physicalize = False
    if physicalize:
        n_extra = 2
        n_local = n_local + n_extra
        nparams = nparams + n_extra
        max_param += n_extra
    
    #print('n_global', n_global)
    #print('n_local', n_local)
    # local_index = [i for i in range(nparams) if i not in global_index]
    local_index = np.array([i for i in range(nparams) if i not in global_index], dtype=int)

    # (2) Fit this function to the data
    tmax = 1
    #try:
        # with simplifier.time_limit(tmax):
    chi2, params, count_lowest, j, success = mixed_optimise_fun(fstr,
                                    likelihood,
                                    global_index,
                                    tmax,
                                    pmin,
                                    pmax,
                                    try_integration=try_integration,
                                    max_param=max_param,
                                    log_opt=log_opt,
                                    method=method)
            
    
    if np.isinf(chi2):
        #print("Failed to optimise function")
        return np.inf, np.inf, np.inf, np.inf,np.inf, np.inf, params, 0, 0
    # params here are in format [global, local]
    
    #print('best:', fstr, chi2, params,count_lowest, j, success)

    # chi2 = 668.4944746494293 


    def get_global_loss(chi2_fcn, xvar, yvar, cov_list, nclusters):
        #calculate total chi2 for all clusters
        def global_loss(params):
            chi2_total = 0
            sum_Fisher_diag = 0
            for i in range(nclusters):
                params_cluster = np.zeros(nparams)
                params_cluster[global_index] = params[:n_global]
                params_cluster[local_index] = params[n_global + i*n_local:n_global + (i+1)*n_local]
                
                xvar_cluster = xvar[i]
                yvar_cluster = yvar[i]
                cov_cluster = cov_list[i]
                L_factor = jnp.linalg.cholesky(cov_cluster)

                chi2_cluster = chi2_fcn(params_cluster, xvar_cluster, yvar_cluster, L_factor)[0]
    
                chi2_total += chi2_cluster
            return chi2_total
        return global_loss


    
    #(3) Calculate codelen
    
    #Obtain the Fisher matrix for each cluster independently
    fcn, eq = likelihood.run_sympify(fstr,
                                            tmax=tmax,
                                            try_integration=try_integration)
    
    # all_a = ' '.join([f'a{i}' for i in range(nparams)])
    # all_a = list(sympy.symbols(all_a, real=True))
    # eq_numpy = sympy.lambdify([x] + all_a, eq, modules=["jax"])

    # if nparams > 1:
    #     all_a = ' '.join([f'a{i}' for i in range(nparams)])
    #     all_a = list(sympy.symbols(all_a, real=True))
    #     eq_numpy = sympy.lambdify([x] + all_a, eq, modules=["jax"])
    # else:
    #     eq_numpy = sympy.lambdify([x, a0], eq, modules=["jax"])
    
    try:
        if n_fun_params == 0 and n_extra == 0:
            eq_numpy = sympy.lambdify([x], eq, modules=["jax"])
        elif n_fun_params == 0 and n_extra > 0:
                rho0, rs = sympy.symbols("rho0 rs", real=True)
                eq_numpy = sympy.lambdify([x, rho0, rs], eq, modules=["jax"])
        elif n_fun_params > 1:
            all_a = ' '.join([f'a{i}' for i in range(n_fun_params)])
            all_a = list(sympy.symbols(all_a, real=True))
            if n_extra>0:
                all_a += sympy.symbols("rho0 rs", real=True)
            eq_numpy = sympy.lambdify([x] + all_a, eq, modules=["jax"])
        elif n_fun_params == 1:
            if n_extra>0:
                rho0, rs = sympy.symbols("rho0 rs", real=True)
                eq_numpy = sympy.lambdify([x, a0, rho0, rs], eq, modules=["jax"])
            else:
                eq_numpy = sympy.lambdify([x, a0], eq, modules=["jax"])
    except Exception:
        # print("BAD:", fcn_i, negloglike, np.isfinite(negloglike))
        Fisher_diag = np.nan
        deriv[:] = np.nan
        return params, chi2, deriv, codelen
    


    loss_template = likelihood.get_loss(eq_numpy)
    chi2_fcn = likelihood.get_wrapped_like(loss_template) #change this
    
    divider = likelihood.yerr
    cov = likelihood.cov_matrix
    xvar = np.array_split(likelihood.xvar, np.where(divider)[0][1:])
    yvar = np.array_split(likelihood.yvar, np.where(divider)[0][1:])
    #print(yvar)
    #sys.exit()
    split_indices = np.where(divider)[0][1:]
    starts = np.concatenate(([0], split_indices))
    ends   = np.concatenate((split_indices, [cov.shape[0]]))

    cov_list = []
    for s, e in zip(starts, ends):
        # This extracts the square sub-matrix for the cluster
        cov_list.append(cov[s:e, s:e])


    #CHANGE THIS - use divider
    #nclusters = len(likelihood.xvar)//points_per_cluster
    nclusters = len(np.where(divider)[0])
    print('num_clusters',  nclusters)
    # fop_global = get_global_loss(chi2_fcn,likelihood.xvar, likelihood.yvar, likelihood.yerr, nclusters)
    # chi2 = fop_global(params)
    # print('chi2', chi2)
    # sys.exit()

    # print('nclusters', nclusters)
    #xvar = np.array_split(likelihood.xvar, nclusters)
    #yvar = np.array_split(likelihood.yvar, nclusters)
    #yerr = np.array_split(likelihood.yerr, nclusters)

    # ------------------------------------------------------------------
    # (3b) Codelen with MIGHTEE-style snapping (Delta=|param| baseline; snap a
    #      parameter to zero only if the total DL strictly improves). Global
    #      params are counted once (Fisher summed over clusters); local params
    #      are counted per cluster.
    # ------------------------------------------------------------------
    Fisher_diag_global = np.zeros(n_global)
    local_codelen = 0.0
    local_Delta = []
    cluster_negloglike = []      # -logL per cluster after LOCAL snapping
    cluster_fops = []            # per-cluster fop, reused for the global step

    for i in range(nclusters):
        xvar_cluster = xvar[i]
        yvar_cluster = yvar[i]
        cov_cluster = cov_list[i]
        L_factor = jnp.linalg.cholesky(cov_cluster)

        params_cluster = np.zeros(nparams)
        params_cluster[global_index] = params[:n_global]
        params_cluster[local_index] = params[n_global + i*n_local:n_global + (i+1)*n_local]

        def get_fop(chi2_fcn, xv, yv, L):
            def fop(pv):
                return chi2_fcn(pv, xv, yv, L)[0]
            return fop
        fop = get_fop(chi2_fcn, xvar_cluster, yvar_cluster, L_factor)
        cluster_fops.append(fop)

        chi2_cluster = float(fop(params_cluster))

        # Fisher diagonal for this cluster (Hessian through the covariance).
        deriv, Fisher_diag = convert_params2(eq_numpy, likelihood, xvar_cluster,
                                             yvar_cluster, L_factor, np.copy(params_cluster),
                                             nparams, max_param=max_param)
        Fisher_diag = np.asarray(Fisher_diag, dtype=float)

        Sigma = np.full(nparams, np.inf)
        good = np.isfinite(Fisher_diag) & (Fisher_diag > 0)
        Sigma[good] = 1.0 / np.sqrt(Fisher_diag[good])

        # Integral fallback where the Fisher estimate is unusable.
        if (np.sum(Sigma <= 0.) > 0) or (np.sum(np.isnan(Sigma)) > 0) or (np.sum(np.isinf(Sigma)) > 0):
            Sigma = get_sigma_from_integral(params_cluster, Sigma, fstr, chi2_cluster,
                                            nparams, fop, number_points=10**3)
            Sigma[np.isnan(Sigma)] = np.inf

        Delta = np.zeros(nparams)
        finite = np.isfinite(Sigma) & (Sigma > 0)
        Delta[finite] = np.sqrt(12.0) * Sigma[finite]
        Delta[~finite] = np.inf

        # Global Fisher is the sum over clusters.
        Fisher_diag_global = Fisher_diag_global + Fisher_diag[global_index]

        if n_local == 0:
            local_Delta.append(np.array([]))
            cluster_negloglike.append(chi2_cluster)
            continue

        # Snap only this cluster's local params; codelen counts local params only.
        best_p, best_neglog, best_code, best_Delta = snap_with_dl_test(
            params_cluster, Delta, active_idx=local_index,
            snap_allowed_idx=local_index, fop=fop, negloglike_orig=chi2_cluster)

        params[n_global + i*n_local:n_global + (i+1)*n_local] = best_p[local_index]
        local_codelen += best_code
        cluster_negloglike.append(best_neglog)
        local_Delta.append(best_Delta[local_index])

    # ------------------------------------------------------------------
    # Global params: snap once, using the TOTAL DL across all clusters.
    # ------------------------------------------------------------------
    negloglike_total = float(np.sum(cluster_negloglike))
    gi = np.asarray(global_index, dtype=int)

    p_global_template = np.zeros(nparams)
    p_global_template[gi] = params[:n_global]

    def fop_global_fn(pv):
        pv = np.asarray(pv, dtype=float)
        total = 0.0
        for i in range(nclusters):
            pc = np.zeros(nparams)
            pc[gi] = pv[gi]
            pc[local_index] = params[n_global + i*n_local:n_global + (i+1)*n_local]
            total += float(cluster_fops[i](pc))
        return total

    if n_global > 0:
        # Sigma for the global params; dummy 1.0 in the other slots so the
        # integral fallback only ever touches the global directions.
        Sigma_g = np.ones(nparams)
        Fisher_diag_global = np.asarray(Fisher_diag_global, dtype=float)
        good_g = np.isfinite(Fisher_diag_global) & (Fisher_diag_global > 0)
        sg = np.full(n_global, np.inf)
        sg[good_g] = 1.0 / np.sqrt(Fisher_diag_global[good_g])
        Sigma_g[gi] = sg
        if np.any(~np.isfinite(Sigma_g[gi])) or np.any(Sigma_g[gi] <= 0):
            Sigma_g = get_sigma_from_integral(p_global_template, Sigma_g, fstr,
                                              negloglike_total, nparams, fop_global_fn,
                                              number_points=10**3)
            Sigma_g[np.isnan(Sigma_g)] = np.inf

        Delta_global = np.zeros(nparams)
        fin_g = np.isfinite(Sigma_g) & (Sigma_g > 0)
        Delta_global[fin_g] = np.sqrt(12.0) * Sigma_g[fin_g]
        Delta_global[~fin_g] = np.inf
        # only the global slots contribute to the global codelen
        non_global = np.ones(nparams, dtype=bool); non_global[gi] = False
        Delta_global[non_global] = 0.0

        best_pg, best_neglog_g, global_codelen, best_Delta_g = snap_with_dl_test(
            p_global_template, Delta_global, active_idx=gi,
            snap_allowed_idx=gi, fop=fop_global_fn, negloglike_orig=negloglike_total)

        params[:n_global] = best_pg[gi]

        # Recompute the total -logL with the (possibly) snapped global params.
        chi2_final = 0.0
        for i in range(nclusters):
            pc = np.zeros(nparams)
            pc[gi] = params[:n_global]
            pc[local_index] = params[n_global + i*n_local:n_global + (i+1)*n_local]
            chi2_final += float(cluster_fops[i](pc))
    else:
        global_codelen = 0.0
        chi2_final = negloglike_total

    #Finally calculate the codelen
    params = np.array(params)
    codelen = global_codelen + local_codelen
    
    # (4) Get the functional complexity
    param_list = ['a%i'%j for j in range(max_param)]
    # print('param_list', param_list)
    # print('labels', labels)
    aifeyn = generator.aifeyn_complexity(labels, param_list)

    #genarate logcnst
    n = np.array([int(tt) for tt in labels if tt.lstrip("-").isdigit()])  # Integers
    n[n==0] = 1  # So we have log(1) for 0 instead of log(0)
    logconst = np.sum(np.log(np.abs(n)))

    #generate katz value:
    in_eqfile = 'katz/data/PhysicsEquations.csv'
    out_eqfile = 'katz/data/NewPhysics.csv'
    kp = KatzPrior(10, basis_functions, in_eqfile, out_eqfile, input_delimiter=';')
    katz_prior = kp.logprior(fstr)
    katz_and_const = - katz_prior + logconst
    # print('katz', katz_prior)
    # print('logconst', logconst)
    # print('katz_and_const', katz_and_const)

    if verbose:
        print('Function:', aifeyn)

    # (5) Combine to get description length
    print('best:', fstr,global_index, chi2, params,count_lowest, j, success)
    print('Function:', aifeyn)
    print('chi2', chi2_final)
    print('codelen', codelen)

    # print('chi2', chi2_final)
    # print('codelen', codelen)
    DL = chi2_final + codelen + aifeyn
    DL_katz = chi2_final + codelen + katz_and_const
    if verbose:
        print('\nDescription length:', DL)
            
    if return_params:
        return chi2, DL, DL_katz, codelen, aifeyn, katz_and_const, params, count_lowest, j

    return chi2, DL, DL_katz, codelen, aifeyn, katz_and_const, count_lowest, j


def fit_from_string(fun, basis_functions, likelihood, indices, pmin=0, pmax=5, tmax=5,
    try_integration=False, verbose=False, Niter=30, Nconv=5, maxvar=20,
    log_opt=False, replace_floats=False, return_params=False, method='BFGS'):

    expr, nodes, complexity = generator.string_to_node(fun, basis_functions, evalf=True)
    labels = nodes.to_list(basis_functions)
    
    # Prepare to get parents
    new_labels = [None] * len(labels)
    for j, lab in enumerate(labels):
        if lab == 'Mul':
            new_labels[j] = '*'
            labels[j] = '*'
        elif lab == 'Add':
            new_labels[j] = '+'
            labels[j] = '+'
        elif lab == 'Div':
            new_labels[j] = '/'
            labels[j] = '/'
        elif lab == 'Sub':
            new_labels[j] = '-'
            labels[j] = '-'
        else:
            new_labels[j] = lab.lower()
            labels[j] = lab.lower()
    param_idx = [j for j, lab in enumerate(new_labels) if generator.is_float(lab) or (lab.startswith('a') and generator.is_float(lab[1:]))]
    assert len(param_idx) <= maxvar
    for k, j in enumerate(param_idx):
        new_labels[j] = f'a{k}'
        
    # Get parent operators
    s = generator.labels_to_shape(new_labels, basis_functions)
    success, _, tree = generator.check_tree(s)
    parents = [None] + [labels[p.parent] for p in tree[1:]]
    
    # Replace floats with symbols (except exponents)
    if replace_floats:
        param_idx = [j for j, lab in enumerate(labels) if (generator.is_float(lab) and not (parents[j].lower() =='pow')) or (lab.startswith('a') and generator.is_float(lab[1:]))]
        for k, j in enumerate(param_idx):
            labels[j] = f'a{k}'
    fstr = generator.node_to_string(0, tree, labels)

    res = single_function(
            labels,
            basis_functions,
            likelihood,
            indices,
            pmin=pmin,
            pmax=pmax,
            tmax=tmax,
            try_integration=try_integration,
            verbose=verbose,
            Niter=Niter,
            Nconv=Nconv,
            log_opt=log_opt,
            return_params=return_params,
            method=method
    )

    if return_params:
        return res[0], res[1],res[2], res[3], res[4], res[5], res[6], res[7], res[8], labels
    
    return res[0], res[1],res[2], res[3], res[4] , res[5], res[7], res[8], labels

def plot_results(likelihood, fn, params_global):

    #plot global fit

    fcn_i = fn.replace('\'', '')
    k = simplifier.count_params([fn], 4)[0]
    fcn_i, eq= likelihood.run_sympify(fn)
    
    if k == 0:
        eq_numpy = sympy.lambdify([x], eq, modules=["numpy"])
    elif k > 1:
        all_a = ' '.join([f'a{i}' for i in range(k)])
        all_a = list(sympy.symbols(all_a, real=True))
        eq_numpy = sympy.lambdify([x] + all_a, eq, modules=["numpy"])
    else:
        eq_numpy = sympy.lambdify([x, a0], eq, modules=["numpy"])

    x_array = np.linspace(0, likelihood.xvar.max(), 1000)
    esd = ExcessSurfaceDensity.calculate(x_array, eq_numpy, params=params_global)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(x_array, esd, label=fcn_i)
    ax.errorbar(likelihood.xvar, likelihood.yvar, yerr=likelihood.yerr, fmt='.')
    # ax.set_xscale(xscale)
    # ax.set_yscale(yscale)
        # plt.show()

    plt.show()


def update_table(results, use_katz = False):
    print('use', use_katz)
    #order table by DL
    # print('results', results)
    #results = sorted(results, key=lambda x: x[2])
    results = np.array(results, dtype=object)
    #print(results)

    #print as a pretty table
    headers = ['Rank', 'Function', 'DL', 'Prel', '- log L', 'Codelen', 'Aifeyn', "Katz", 'Mean a0', 'Std a0', ' Mean a1', 'Std a1', 'Divergence', 'Complexity', "Global indices"]

    if use_katz:
        #change DL to DL_katz
        #results[:, 2] = results[:, 4] + results[:, 5] + results[:, 7]
            
        # Define input and output equation files
        in_eqfile = 'katz/data/PhysicsEquations.csv'
        out_eqfile = 'katz/data/NewPhysics.csv'
        basis_functions = [
            ["x", "a"],  # Nullary operators
            ["sqrt", "exp", "log", "sin", "cos", "arcsin", "arccos", "tanh", "abs", "re", "im"],  # Unary operators
            ["+", "-", "*", "/", "pow"]]  # Binary operators

        # Recompute Katz Prior only where it is 0
        kp = KatzPrior(10, basis_functions, in_eqfile, out_eqfile, input_delimiter=';')
            # Column indices for clarity
        #kp = KatzPrior(10, basis_functions, in_eqfile, out_eqfile, input_delimiter=';')

        for i, fstr in enumerate(results[:][1]): # Assuming column 1 stores function strings 
            #if results[i, 7] == 0: # Check if the Katz value is 0 
            try: 
                katz_prior = kp.logprior(fstr) #print(katz_prior) 
                results[i][ 7] = - katz_prior 
            except Exception as e: 
                print(fstr, e) 
            results[i][2] = results[i][4] + results[i][5] + results[i][7]
        #results[:, 2] = results[:, 2] - results[:, 6] + results[:, 7] 
        #results[:][2] = results[:][4] + results[:][5] + results[:][7]

            #for i, fstr in enumerate(results[:][1]):  # Assuming column 1 stores function strings
                #if results[i, 7] == 0:  # Check if the Katz value is 0
                #try:
                    #katz_prior = kp.logprior(fstr)
                    #print(katz_prior)
                    #results[i][ 7] = - katz_prior
                #except Exception as e:
                    #print(fstr, e)
            #results[:, 2] = results[:, 2] - results[:, 6] + results[:, 7]
        #results[:][2] = results[:][4] + results[:][5] + results[:][7]
    
    #results = sorted(results, key=lambda x: x[2])
    #results = results[results[:, 2].argsort()]
    #results = np.array(results, dtype=object)
    results = results[np.argsort([row[2] for row in results])]
    #print(results)
    table_data = []
    seen_funcs = set()
    #print(len(results[0][:]))
    #for i in range(len(results[:][0])):
    counter = 0
    for i in range(4000):
        func_string = results[i][1]      # the function expression
        no_global = (str(results[i][-1]) == '---')
        #print(no_global)
        if no_global:
            # check if func was already seen
            if func_string in seen_funcs:
                #print(func_string)
                #print(no_global)
                # skip it if already processed before
                continue
            else:
                # mark this function as processed
                seen_funcs.add(func_string)
        #row = [
        #    i + 1,
        #   results[i][ 1],
        #    f'{results[i, 2]:.2f}',
        #    f'{results[i, 3]:.2f}',
        #    f'{results[i, 4]:.2f}',
        #    f'{results[i, 5]:.2f}',
        #    f'{results[i, 6]:.2f}',
        #    f'{results[i, 7]:.2f}'
        #] + [f'{p:.2e}' for p in results[i, 8:12]] + [results[i, -3]] + [results[i, -2]] + [results[i, -1]]
        #table_data.append(row)
        #if results[i][-2] != 10:
        counter += 1
        row = [
                counter,
                results[i][1],
                f'{results[i][2]:.2f}',
                f'{results[i][3]:.2f}',
                f'{results[i][4]:.2f}',
                f'{results[i][5]:.2f}',
                f'{results[i][6]:.2f}',
                f'{results[i][7]:.2f}'
                ] + [f'{p:.2e}' for p in results[i][8:12]] + [results[i][-3]] + [results[i][-2]] + [results[i][-1]]
        table_data.append(row)

    pretty_table = PrettyTable()
    pretty_table.field_names = headers

    for row in table_data:
        pretty_table.add_row(row)
    print(pretty_table)
    #calculate prel

    #save table
    #with open('esr/fitting/output/combining_clusters/global_local_results.txt', 'w') as f:
    #    for row in table_data:
    #        f.write('|'.join([str(r) for r in row]) + '\n')

    with open('esr/fitting/output/combining_clusters/combine_all_comp_all_clusters_global_local.txt', 'w') as f:
            print('Saving to' + 'pretty_all_clusters_comp' + str(i) + '.txt')
            f.write(str(pretty_table))

    return None

def load_table(file_path):

    def try_convert(value):
        try:
            # Attempt to convert to a float
            return float(value)
        except ValueError:
            # Return the original value if conversion fails
            return value

    loaded_headers = []
    loaded_table_data = []

    # Read the file
    with open(file_path, "r") as file:
        # Read headers
        loaded_headers = file.readline().strip().split("\t")
        
        # Read the rest of the rows
        for line in file:
            row = line.strip().split("\t")
            row = [try_convert(element) for element in row]
            row.append("---")
            loaded_table_data.append(row)

    return loaded_table_data

def process_function(fn, index_tuple, comp, data_file, cov_file, run_name, basis_functions, physicalize):
    """
    Process a single function and return the results.
    """
    indices = np.array(index_tuple)
    # print(f"Processing function: {fn} {indices}")

    # Initialize the likelihood for this process
    likelihood = WLLikelihood(data_file, cov_file, run_name, data_dir=None, fn_set='core_maths', keep_nans=True)

    # Perform the computation
    chi2, DL, DL_katz, codelen, aifeyn, katz, params, Niter, Nconv, labels = fit_from_string(
        fn,
        basis_functions,
        likelihood,
        indices,
        method='BFGS',
        try_integration=False,
        verbose=True,
        log_opt=True,
        return_params=True
    )

    num_params = simplifier.count_params([fn], 4)[0]

    if physicalize:
        num_params += 2

    if num_params == 1:
        combined_params = np.zeros(2)
        combined_params[indices] = params
        combined_params[~indices] = 0

    else:
        params_global = params[:len(indices)]
        combined_params = np.zeros(num_params)
        combined_params[indices] = params_global
        if num_params != len(indices):
            params_local = np.mean(params[len(indices):])
            combined_params[~indices] = params_local

    # print('combined_params', combined_params)

    divergence = '---'
    prel = 0
    rank = 0

    result = [
            rank, fn, DL, prel, chi2, codelen, aifeyn, katz,
            combined_params[0], 0, combined_params[1], 0, 
            divergence, comp, indices
            ]
    
    return result, Niter, Nconv

def run_functions(func_index_list, data_file, cov_file, run_name='kk',
                  basis_functions=None, physicalize=False, verbose=False):
    """Run the mixed global/local optimisation + codelen for a list of functions.

    Args:
        func_index_list: iterable of (function_string, global_indices) pairs,
            e.g. [('a0/x', [0]), ('a0*pow(x,a1)', [0, 1])]. `global_indices`
            are the parameter indices treated as global (shared across clusters);
            the rest are local (one value per cluster).
        data_file, cov_file: CLASH ESD data file and covariance matrix (.npy).
        run_name: label passed to WLLikelihood.
        basis_functions: ESR basis (defaults to the full core-maths set).
        physicalize: whether to append the two physical (rho0, rs) parameters.
        verbose: print a one-line summary per function.

    Returns:
        list of result rows (same layout as `process_function`'s result).
    """
    if basis_functions is None:
        basis_functions = [
            ["x", "a"],
            ["sqrt", "exp", "log", "sin", "cos", "arcsin", "arccos", "tanh", "abs", "re", "im"],
            ["+", "-", "*", "/", "pow"],
        ]

    results = []
    for fn, indices in func_index_list:
        result, Niter, Nconv = process_function(
            fn, tuple(indices), 0, data_file, cov_file, run_name,
            basis_functions, physicalize)
        results.append(result)
        if verbose:
            print(f"[run_functions] {fn}  global={list(indices)}  "
                  f"DL={result[2]:.3f}  -logL={result[4]:.3f}  codelen={result[5]:.3f}",
                  flush=True)
    return results

def load_intermediate_results(intermediate_dir):
    """Helper function to load intermediate results from a file."""
    results = []
    count_inf = 0
    for filename in os.listdir(intermediate_dir):
        #print(os.listdir(intermediate_dir))
        file_path = os.path.join(intermediate_dir, filename)

        if os.path.isfile(file_path) and filename.startswith("results"):
        #if os.path.exists(intermediate_file):
            print(f"Rank 0: Loading results from {filename}")
            
            with open(file_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    line = line.replace("Array", "np.array")
                    data = eval(line, {"np": np, "array": np.array, "float32": np.float32, "float64": np.float64, "inf": np.inf, "nan": np.nan})

                    # Safely replace 'Array(...)' with 'np.array(...)' so that eval can handle it
                    #line = line.replace("Array", "np.array")
                    #parts = line.strip().split('|')
                    #parts = line.strip()[1:-1].split(', ')
                    fn = data[1]
                    result = data[1:]  # Convert string to Python object
                    #print(results)
                    #print(data)
                    #line = line.replace("Array", "np.array")
                    # Safely evaluate the line using eval with context
                    result = eval(line, {"np": np, "array": np.array, "float32": np.float32, "float64": np.float64, "inf": np.inf, "nan": np.nan})
                    result = list(result)
                    if np.isinf(result[4]):
                        if 'a2' in fn and len(result[-1]) !=3:
                            count_inf += 1
                        elif 'a2' not in fn and len(result[-1]) !=2:
                            count_inf += 1
                        elif 'a1' not in fn and len(result[-1]) !=1:
                            count_inf += 1
                    result[4] = result[2] - result[5] - result[6]            
                    result[4] = np.nan_to_num(result[4])
                    #if np.isinf(result[4]):
                        #count_inf += 1
                    results.append(result)
    else:
        print("Rank 0: No intermediate results found. Set `rerun=True` to process functions.")
    print('Count infs', count_inf)
    return results

def run_all_functions():
    use_katz = False
    rerun = False
    #input_file = "esr/fitting/output/combining_clusters/combine_all_comp_all_clusters.dat"
    #input_file = "esr/fitting/output/combining_clusters/combine_all_comp_all_clusters_duplicate_to_comp_9.dat"
    input_file = "esr/fitting/output/combining_clusters/combine_all_comp_all_clusters.dat"
    physicalize = False

    intermediate_dir = "esr/fitting/output/combining_clusters/intermediate_results"
    #intermediate_file = os.path.join(intermediate_dir, f"results_rank_{rank}.txt")

    local_results = []

    if not rerun:
        if rank == 0:
            # Load intermediate results from a file
            local_results = load_intermediate_results(intermediate_dir)
        
        local_results = comm.bcast(local_results if rank == 0 else None, root=0)

        if rank != 0:
            print(f"Rank {rank}: Skipping processing since rerun=False")
        else:
            print(f"Rank 0: Loaded {len(local_results)} results. Ready to proceed.")


    else:
        data_file = 'esr/data/all_clash_esd.txt'
        cov_file = 'esr/data/all_clash_cov_matrix.npy'
        run_name = 'kk'
        # basis_functions = [["x", "a"],  # type0
        #                 ["inv", "abs", "log", "exp", "re", "im"],  # type1
        #                 ["+", "*", "-", "/", "pow"]]  # type2


        basis_functions = [
            ["x", "a"],  # Nullary operators
            ["sqrt", "exp", "log", "sin", "cos", "arcsin", "arccos", "tanh", "abs", "re", "im"],  # Unary operators
            ["+", "-", "*", "/", "pow"]  # Binary operators
]

        # Read the input functions and parameters (only by rank 0)
        if rank == 0:

            funcs, indices, comps = [], [], []
            file_path = 'esr/fitting/output/combining_clusters/all_global_local_funcs_part_5.txt'

            with open(file_path, 'r') as f:
                for line in f:
                    parts = line.strip().split('|')
                    funcs.append(parts[0])  # Function string
                    indices.append(eval(parts[1]))  # Convert string to tuple
                    comps.append(int(parts[2]))  # Convert string to integer

            # Split the tasks among processes
            #cut  = 1
            #funcs = funcs[:cut]
            #indices = indices[:cut]
            #comps = comps[:cut]
            tasks = list(zip(funcs, indices, comps))
            tasks_per_rank = [tasks[i::size] for i in range(size)]  # Split evenly

            # print(tasks_per_rank)

            print("Number of cores:  ", size)
            print("number of funcs:  ", len(funcs))
            print("Number of funcs per core:   ", len(tasks_per_rank[0]))

            # sys.exit()
        else:
            tasks_per_rank = None

        comm.Barrier()

        # Scatter tasks to all processes

        local_tasks = comm.scatter(tasks_per_rank, root=0)


        comm.Barrier()

        # Directory to store intermediate results
        if rank == 0 and not os.path.exists(intermediate_dir):
            os.makedirs(intermediate_dir)

        # Process local tasks
        local_results = []
        Niters = []
        Nconvs = []
        function_metrics = []  # Track metrics per function
        total_tasks = len(local_tasks)
        for i, (fn, index_tuple, comp) in enumerate(local_tasks):
        #for fn, index_tuple, comp in local_tasks:
            start = time.time()
            #print(f"Processing function: {fn} {index_tuple}")
            print(f"{rank}, Processing function: {i + 1}/{total_tasks} -> {fn} {index_tuple}")
            result, Niter, Nconv = process_function(fn, index_tuple, comp, data_file,cov_file, run_name, basis_functions, physicalize)
            local_results.append(result)
            
            Niters.append(Niter)
            Nconvs.append(Nconv)
            function_metrics.append((fn, index_tuple, Niter, Nconv))

            end = time.time()
            print(f"Time taken: {end - start:.2f}")

            # Save intermediate results to a file
            intermediate_file = os.path.join(intermediate_dir, f"results_rank_{rank}.txt")
            with open(intermediate_file, 'a') as f:
                f.write(f"{result}\n")

            metrics_file = os.path.join(intermediate_dir, f"metrics_rank_{rank}.txt")
            with open(metrics_file, 'a') as f:
                f.write("Function | Index Tuple | Iterations | Convergence\n")
                for fn, index_tuple, Niter, Nconv in function_metrics:
                    f.write(f"{fn} | {index_tuple} | {Niter} | {Nconv}\n")

    comm.Barrier()
    # Gather all results back to rank 0
    all_results = comm.gather(local_results, root=0)

    # Save results (only rank 0)
    if rank == 0:
        # esr/fitting/output_decreasing/output/12_clusters/combine_all_comp_12_clusters.txt
        results = load_table(input_file)
        #print('aqui', results)    
        # Combine all results
        for process_results in all_results:
            results.extend(process_results)

        # Save the updated results
        update_table(results, use_katz)
        # print("All functions processed and results saved.")

    if rerun and rank == 0:
        all_metrics = comm.gather(function_metrics, root=0)
        # Save all metrics to a summary file
        summary_metrics_file = os.path.join(intermediate_dir, "summary_metrics.txt")
        file_exists = os.path.exists(summary_metrics_file)
        
        with open(summary_metrics_file, 'a') as f:  # 'a' for append
            # Write the header only if the file is being created for the first time
            if not file_exists:
                f.write("Function | Index Tuple | Iterations | Convergence\n")
            for rank_metrics in all_metrics:
                for fn, index_tuple, Niter, Nconv in rank_metrics:
                    f.write(f"{fn} | {index_tuple} | {Niter} | {Nconv}\n")
    
        print("All function metrics saved to summary_metrics.txt")

def run_single_function():

    data_file = 'esr/data/all_clash_esd.txt '
    cov_file = 'esr/data/all_clash_cov_matrix.npy'
    run_name = 'kk'

    # Define the function and global indices
    basis_functions = [["x", "a"],  # type0
                ["inv", "abs", "log", "exp"],  # type1
                ["+", "*", "-", "/", "pow"]]  # type2
    
    # fn = 'a0/(pow(Abs(a1),x) - 1/(a2*x))'
    # fn = 'pow(Abs(a0),(x/(x + 1/x)))/a1'

    fn = 'a0/x'
    indices = [0]


    # Run single function

    result = process_function(fn, indices, 0, data_file, run_name, basis_functions, physicalize=True)

    # chi2, DL, codelen, aifeyn, params, labels = fit_from_string(fn,
    #                                                 basis_functions,
    #                                                 likelihood,
    #                                                 indices,
    #                                                 method = 'BFGS',
    #                                                 try_integration = False,
    #                                                 verbose=True,
    #                                                 log_opt=True,
    #                                                 return_params=True)
    
    # print(chi2, DL, labels, params)
    print(result)


    #Do global fit to compare
    # print('GLOBAL FIT')
    # indices_all_global = [0, 1]
    # likelihood = WLLikelihood(data_file, run_name, data_dir=None, fn_set = 'core_maths') 
    # chi2, DL, codelen, aifeyn, params, labels = fit_from_string(fn,
    #                                             basis_functions,
    #                                             likelihood,
    #                                             indices_all_global,
    #                                             method = 'Nelder-Mead',
    #                                             try_integration = False,
    #                                             verbose=True,
    #                                             log_opt=True,
    #                                             return_params=True)
    # print('params', params)



if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    run_all_functions()
    # run_single_function()
    # input_file = "esr/fitting/output_decreasing/output/12_clusters/combine_all_comp_all_clusters.dat"
    # load_table(input_file)
    # sys.exit()



    




