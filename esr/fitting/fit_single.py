import sys
import numpy as np

from esr.fitting.test_all import optimise_fun
# from esr.fitting.test_all_2 import optimise_fun
from esr.fitting.test_all_Fisher import convert_params

import esr.generation.generator as generator
import esr.generation.simplifier as simplifier
import sympy

import math
import jax
jax.config.update("jax_enable_x64", True)
jax.numpy.array(1, dtype=int)
import jax.numpy as jnp
from jax import vmap
import scipy
import scipy.optimize
import matplotlib.pyplot as plt
from esr.fitting.sympy_symbols import *
import itertools
from esr.esd import ExcessSurfaceDensity
from scipy.integrate import cumulative_trapezoid as cumtrapz



def get_sigma_from_integral(p, Sigma, fcn_i, negloglike_all, nparams, max_fun_params, fop, number_points=10**3):
        #j is the index of the parameter we are looking at
        #i is the index of the data point we are looking at

        def fraction_likelihood(x, p, j, factor, log=True):
            # params = jnp.copy(p)
            # params[j] += factor * 10**x
            # if np.isinf(fop(params)):
            #     negloglike = 0.1
            # else:
            #     negloglike = np.abs(fop(params) - negloglike_all - np.log(1e3))
            # return negloglike

            def compute_negloglike(params):
                # jax.debug.print('aqui {}', params)
                # jax.debug.print('aqui {}', fop(params))
                # jax.debug.print('aqui 2 {}', negloglike_all)
                # jax.debug.print('aqui 3 {}', negloglike_all + jnp.log(1e3))
                # jax.debug.print('aqui 4 {}', jnp.log(1e3))

                # neg = jnp.array(negloglike_all)

                # # diff = jnp.array(negloglike_all) - jnp.log(1e3)
                # jax.debug.print('Difference: {}', neg.dtype)

                return jnp.abs(fop(params) - negloglike_all - jnp.log(1e3))
            def inf_value(_):
                # jax.debug.print('aqui')
                return 0.0999
            
            # Modify parameter
            params = jax.lax.cond(log,
                  lambda _: p.at[j].set(p[j] + factor * 10**x[0]),
                  lambda _: p.at[j].set(p[j] + factor * x[0]),            
                  operand=None)
            # if log:
            #     params = p.at[j].set(p[j] + factor * 10**x[0])
            # else:
            #     params = p.at[j].set(p[j] + factor * x[0])
            
            negloglike = jax.lax.cond(jnp.isinf(fop(params)), inf_value, compute_negloglike, params)
            return negloglike

        def get_loss():
            return jax.jit(fraction_likelihood)

        def wrap_loss(fraction_likelihood):
            def loss(x, p, j, factor, log=True):
                # print(fraction_likelihood(x, p, j, factor))
                return fraction_likelihood(x, p, j, factor, log)
            

            return loss
        
        def get_boundary(res, log=True):
            if log:
                return 10**res.x[0]
            else:
                return res.x[0]
        
        def get_integral(theta, boundary_left, boundary_right, param, fop, negloglike, number_points=5*10**2):
            def compute_like(theta):
                return jnp.exp(-fop(theta) + negloglike)

            max_boundary = np.max([boundary_left, boundary_right])
            boundary_range = np.array([-boundary, boundary])

            boundaries = [boundary_left, boundary_right]
            # Loop through left and right boundaries
            integrals = []
            for index_boundary, boundary_shift in enumerate(boundary_range):
            # for boundary_shift in boundary_range:
                a_range = np.linspace(theta, theta + boundary_shift, number_points)
                thetas = np.tile(param, (len(a_range), 1))
                thetas[:, j] = a_range
                like = vmap(compute_like)(thetas)
                # print(boundaries[index_boundary])
                # print(a_range)
                # print('theta', theta)
                like = jnp.where(np.abs(theta - a_range) > boundaries[index_boundary], 0, like)
                # like[a_range  > boundaries[index_boundary]] = 0

                # print('like', like, flush=True)

                # plt.plot(a_range, like)
                # plt.show()

                # Integrate
                integral = cumtrapz(like, a_range)
                integrals.append(integral)

            # Combine left and right integrals
            integral_left, integral_right = [np.array(integral) for integral in integrals]
            # print('integral_left', integral_left, flush=True)
            # print('integral_right', integral_right, flush=True)
            integral = - integral_left + integral_right
            # print('integral', integral, flush=True)

            return integral, a_range
        
        def test_success(res):
            # p = np.copy(param)
            # factor = 1
            # p[j] = param[j] + factor*10**(res.x[0])
            # print(param)
            # print(p)
            # print('fop', fop(p))
            # print('negloglike_all', negloglike_all)
            # likelihood_diff =  fop(p) - negloglike_all
            # print('likelihood_diff', likelihood_diff, flush=True)
            # print(np.abs(fop(p) - negloglike_all - np.log(1e4)))
            if res.success == False or res.fun > 0.1:
            # if res.success == False or likelihood_diff <= np.log(5*1e2):
                return False
            else:
                return True

        # param = np.append(p[:nparams], p[-2:])
        param = np.append(p[:nparams], p[max_fun_params:])
        # print('param', param, flush=True)
        # sys.exit()
        # initial_guesses = [np.ones(nparams), np.ones(nparams)*(-1), np.log10(param)]
        # initial_guesses = [[-20]*nparams, [-10]*nparams,[0]*nparams, [10]*nparams, [20]*nparams]
        # initial_guesses = [- 20* np.log10(np.abs(param)), - 10*np.log10(np.abs(param)), - 5*np.log10(np.abs(param)), np.log(np.abs(param)), 5*np.log10(np.abs(param)), 10*np.log10(np.abs(param))]

        for j in range(len(param)):
        # for j in [2]:
                theta = param[j]
                # print('theta', theta, flush=True)

                #try to optimise not in log scale firts
                # initial_guesses_log_space_small = [-20, -10, -5]
                # initial_guesses_normal_space = np.sort([0, np.abs(theta)])
                # initial_guesses_log_space_large = [1, 5, 10]

                # initial_guesses = [(val, True) for val in initial_guesses_log_space_small] + \
                #   [(val, False) for val in initial_guesses_normal_space] + \
                #   [(val, True) for val in initial_guesses_log_space_large]

                # initial_guesses.sort(key=lambda x: 10**x[0] if x[1] else x[0])

                initial_guesses = [-20, -10, -5,-1, 0, 0.5, 1, np.log10(np.abs(theta)), 2, 5]
                initial_guesses = [(val, True) for val in initial_guesses]

                # print('initial_guesses', initial_guesses, flush=True)
                # sys.exit()
                # initial_guesses = np.append(initial_guesses_log_space_small, initial_guesses_normal_space)
                # initial_guesses = np.append(initial_guesses, initial_guesses_log_space)

                # print('initial_guesses', initial_guesses, flush=True)

                # #Plot likelihood
                # Delta_plot = 20**np.abs(theta)
                # x_range = np.linspace(theta - Delta_plot, theta + Delta_plot, 10**4)
                # # x_range = np.linspace(theta, theta + Delta_plot, 10**4)
                # print('x_range', x_range)
                # # x_range = np.append(x_range, theta)
                # params_range = np.tile(param, (len(x_range), 1))
                # params_range[:, j] = x_range

                # nll = []
                # for params in params_range:
                #     # print('params', params, flush=True)
                #     negloglike = fop(params)
                #     # print('negloglike', negloglike, flush=True)
                #     nll = np.append(nll, negloglike)  
                # # print('Max', np.max(nll), flush=True)     
                # # print(np.exp(-np.max(nll) + jnp.min(nll)))
                # # print('nll', nll, flush=True)
                # # print(np.exp(-nll + jnp.min(nll[np.isnan(nll) == False])))
                # # print(x_range)
                # plt.figure()
                # plt.plot(x_range, np.exp(-nll + jnp.min(nll[np.isnan(nll) == False])))
                # # plt.plot(x_range, np.exp(-nll))
                # plt.yscale('log')
                # plt.show()
                # sys.exit()

                loss_template = get_loss()
                chi2_fcn = wrap_loss(loss_template)

                for factor in [-1, 1]:
                    # print('factor', factor, flush=True)
                    for initial_guess, is_log in initial_guesses:  
                        success = False
                        # print('initial_guess', initial_guess, flush=True)

                        log = is_log                    
                        res = scipy.optimize.minimize(chi2_fcn, initial_guess, args=(param, j, factor, log), method='Nelder-Mead', tol=1e-8)
                        # print('res', res, flush=True)
                        params_res = param.copy()
                        params_res[j] = param[j] + factor * 10**res.x[0]
                        # print('params', params_res, param)
                        # print('chi2fcn', chi2_fcn([-9.63662553], param, j, factor, log))
                        # print('res', res.fun, res.x, fop(params_res), fop(param))


                        if test_success(res)== True:
                            if factor == -1:
                                boundary_left = get_boundary(res, log)
                                # print('boundary_left', boundary_left, flush=True)
                            else:
                                boundary_right = get_boundary(res, log)
                                # print('boundary_right', boundary_right, flush=True)
                            success = True
                            break
                        # else:
                        #     continue
                    if not success:
                        break

                    # if test_success(res)== False:
                    #     print("Couldn't find integral limits", factor, fcn_i, theta, flush=True)
                        
                        # return np.array([np.inf] * nparams)

                if not success:
                    # print("Couldn't find integral limits", fcn_i, theta, flush=True)
                    Sigma[j] = np.inf
                    continue

                #do integral
                boundary = np.max([boundary_left, boundary_right])
                # print('boundary', boundary)
                integral, a_range = get_integral(theta, boundary_left, boundary_right, param, fop, negloglike_all, number_points=number_points)
                                        
                #get the 68% confidence interval from the integral
                # print(integral/integral[-1])
                # print(theta+boundary)
                arg_min = np.argmin(abs(0.68 - integral/integral[-1]))
                param68 = a_range[arg_min + 1]
                # print(param68)
                sigma = np.abs(theta - param68)
                # print('sigma', sigma, flush=True)
                Sigma[j] = sigma
        print(Sigma)

        return Sigma

def codelength(params, Sigma, nparams):
    k = nparams
    params_arr = np.atleast_1d(np.array(params, dtype=float))
    Delta = np.atleast_1d(np.sqrt(12.)*Sigma)

    if (
        not np.all(np.isfinite(params_arr))
        or not np.all(np.isfinite(Delta))
        or np.any(Delta <= 0)
        or np.all(params_arr == 0)
    ):
        return np.inf

    if np.isinf(Delta).any():
        # print('Delta', Delta)
        # print('params', params)
        mask_inf = Delta == np.inf
        Delta[mask_inf] = params_arr[mask_inf]

    codelen = k*math.log(2.) + np.sum(np.log(np.abs(params_arr)/Delta))
    return codelen


def get_codelen(likelihood, params, nparams, max_param, fstr, eq, negloglike):
        loss_template = likelihood.get_loss(eq, value = 'evaluate')
        chi2_fcn =likelihood.get_wrapped_like(loss_template)
        xvar, yvar, yerr = likelihood.xvar, likelihood.yvar, likelihood.yerr

        def get_fop(chi2_fcn, total_param):
            if total_param > 0:
                def fop(x):
                    return chi2_fcn(x, xvar, yvar, yerr)
            else:
                def fop(x):
                    return chi2_fcn([x], xvar, yvar, yerr)
            return fop
        fop = get_fop(chi2_fcn, nparams)
        # print('AQUI', fop(params))
        Sigma = get_sigma_from_integral(params, np.zeros(nparams), fstr, negloglike, nparams, max_param, get_fop(chi2_fcn, nparams), number_points=10**3)
        # print('Sigma', Sigma)
        Delta = np.sqrt(12.)*Sigma
    
        codelen = codelength(params, Sigma, nparams)
        # print('codelen', codelen)
        return codelen, Delta

def get_fisher_diag_from_deriv(deriv, nparam, max_param=4):
    Hmat_max = np.zeros((max_param, max_param))
    triu_indices = np.triu_indices(max_param)
    Hmat_max[triu_indices] = deriv
    fisher_diag = np.diag(Hmat_max[:nparam, :nparam])
    
    return fisher_diag

def snap_params(negloglike_orig, params_orig,  Delta_orig, Nsteps, nparams, fop):
    p = np.copy(params_orig)
    ptrue = np.copy(params_orig)
    Delta = np.copy(Delta_orig)
    # print('Delta aqui', Delta, flush=True)

    try:
        p[Nsteps<1] = 0. 
    except (IndexError, TypeError):
        p[:]=0.


    #first try setting all parameters to zero  
    # print('Nsteps', Nsteps)
    # print('p', p)    
    negloglike = fop(p)
    likelihood = np.exp(-negloglike)
    # print('negloglike', negloglike, flush=True)
    # print('likelihood', likelihood, flush=True)
                
    if np.isfinite(negloglike) and likelihood !=0:
        mask = Nsteps<1
        # print('MASK', mask)
        params = p
        params[mask] = 0
        Delta[mask] = 0
        mask = np.logical_not(mask)

        # params  =params[mask]
        # Delta = Delta[mask]
        # print('AQUI')
        # print('params 1', params)
        # print('Delta', Delta)
        # print('maks here', mask)

    else:
        # Let's see if setting any of the parameters to zero is ok
        try_idx = np.arange(nparams)[Nsteps < 1]

        # print('try_idx', try_idx, flush=True)
        for r in reversed(range(1, len(try_idx))):
            for idx in itertools.combinations(try_idx, r):
                p = np.copy(ptrue)
                for idx_ in idx:
                    p[idx_] = 0.

                negloglike = fop(p)
                likelihood = np.exp(-negloglike)
                # print('negloglike some params', negloglike, flush=True)

                if np.isfinite(negloglike) and likelihood != 0:
                    # If valid, update k and break out of loop
                    kept_mask = np.ones(len(p), dtype=bool)  # Keep all initially
                    kept_mask[list(idx)] = False # Exclude params set to zero
                    #mask is the oppsoite to kept_mask
                    mask =  kept_mask
                    # mask = np.logical_not(kept_mask)
                    Delta[Nsteps<1] = abs(p[Nsteps<1]) # the rest of the params with Nsteps<1 are set to the absolute value of the parameter
                    # params = p[kept_mask]
                    params = p
                    # print('no entiendo', params)
                    # params[kept_mask] = 0

                    # print('params 2', params)
                    break
    
        # if np.isfinite(negloglike):
        #     print('params 2', params)
        #     print('Delta', Delta)
            # params = p[kept_mask]
            # Delta = Delta[kept_mask]
        # elif not np.isfinite(negloglike_all[i]) and not np.isnan(negloglike_all[i]): # infinite nll
        if not np.isfinite(negloglike) or likelihood == 0:
            params = ptrue
            Delta[Nsteps<1] = abs(ptrue[Nsteps<1])
            negloglike = negloglike_orig
            mask = np.ones(len(params), dtype=bool)

        # print('params', params)
        # print('Delta', Delta)
        # print('negloglike', negloglike)
    
    return negloglike, params, Delta, mask



def single_function(labels, basis_functions, likelihood, method, pmin=-10, pmax=10, tmax=5,
    try_integration=False, verbose=False, Niter=30, Nconv=5, log_opt=False,
    return_params=False):
    """Run end-to-end fitting of function for a single function
    
    Args:
        :labels (list): list of strings giving node labels of tree
        :basis_functions (list): list of lists basis functions. basis_functions[0] are
            nullary, basis_functions[1] are unary and basis_functions[2] are
            binary operators
        :likelihood (fitting.likelihood object): object containing data, likelihood
            functions and file paths
        :pmin (float, default=0.): minimum value for each parameter to consider when
            generating initial guess
        :pmax (float, default=3.): maximum value for each parameter to consider when
            generating initial guess
        :tmax (float, default=5.): maximum time in seconds to run any one part of
            simplification procedure for a given function
        :try_integration (bool, default=False): when likelihood requires integral,
            whether to try to analytically integrate (True) or just numerically
            integrate (False)
        :verbose (bool, default=True): Whether to print results (True) or not (False)
        :Niter (int, default=30): Maximum number of parameter optimisation iterations
            to attempt.
        :Nconv (int, default=5): If we find Nconv solutions for the parameters which are
            within a logL of 0.5 of the best, we say we have converged and stop
            optimising parameters
        :log_opt (bool, default=False): whether to optimise 1 and 2 parameter cases in
            log space
        :return_params (bool, default=False): whether to return the parameters of the
            maximum likelihood point
    
    Returns:
         :negloglike (float): the minimum value of -log(likelihood) (corresponding to
            the maximum likelihood)
         :DL (float): the description length of this function
         :params (optional, list): the maximum likelihood parameters. Only returned if
            `return_params` is true
    
    """
    # print('likelihood', likelihood.physicalize)
    # (1) Convert the string to a sympy function
    s = generator.labels_to_shape(labels, basis_functions)
    success, _, tree = generator.check_tree(s)
    fstr = generator.node_to_string(0, tree, labels)
    max_param = simplifier.get_max_param([fstr], verbose=verbose)

    if likelihood.physicalize:
        max_param += 2
    fstr, fsym = simplifier.initial_sympify(
        [fstr], max_param, parallel=False, verbose=verbose)
    fstr = fstr[0]
    fsym = fsym[fstr]
    nparams = simplifier.count_params([fstr], max_param)[0]
    # print(fstr)
    # (2) Fit this function to the data
    Niter =3000
    Nconv = 500
    # method = 'Nelder-Mead'
    # print(method)
    # # print('Niter', Niter)

    chi2, params, count_lowest, j, success = optimise_fun(fstr,
                            likelihood,
                            tmax,
                            pmin,
                            pmax,
                            try_integration=try_integration,
                            max_param=max_param,
                            Niter_params=[Niter],
                            Nconv_params=[Nconv],
                            Niter=Niter,
                            Nconv=Nconv,
                            log_opt=log_opt,
                            method=method,
                    )
    # params = [-7.88964842e+01, -5.04381489e+19, -1.48554410e-02, -1.34980246e+00]
    # chi2 = 1.67 
    # count_lowest = 0
    # j = 0
    # success = True
    print(fstr, chi2, params, count_lowest, j, success, flush=True)

    #now if this didn't convere or any of the params are bigger tha. 1e5 in abs or smaller than 1e-5 rerun with log space optimisation
    # if (not success) or (np.any(np.abs(params) > 1e6)) or (np.any(np.abs(params) < 1e-6)):
    #     Niter = 800
    #     Nconv = 200
    #     print('Re-running in log space optimisation', flush=True)
    #     chi2_log, params_log, count_lowest_log, j_log, success_log = optimise_fun(fstr,
    #                             likelihood,
    #                             tmax,
    #                             pmin,
    #                             pmax,
    #                             try_integration=try_integration,
    #                             max_param=max_param,
    #                             Niter_params=[Niter],
    #                             Nconv_params=[Nconv],
    #                             Niter=Niter,
    #                             Nconv=Nconv,
    #                             log_opt=True,
    #                             method=method,
    #                     )
    #     print('fstr log', chi2_log, params_log, count_lowest_log, j_log, success_log, flush=True)

    #     #if the log space better than the previous one use it
    #     if (chi2_log < chi2):
    #         chi2 = chi2_log
    #         params = params_log
    #         count_lowest = count_lowest_log
    #         j = j_log
    #         success = success_log

    # return chi2, chi2, params, params, 0, 0
    
    # chi2 = 1.7251848595448611
    # params = jnp.array([-5.50783545e+10, -8.90707414e-01])
    
    # j = 0
    # count_lowest = 0

    if np.isnan(chi2) or np.isinf(chi2):
        print('chi2 is nan or inf')
        return chi2, np.nan, params, params, np.nan, np.nan


    # DL = 0
    # Delta  = params
    # print('best:', chi2, params, count_lowest, j, success)
    
    fcn, eq = likelihood.run_sympify(fstr,
                                        tmax=tmax,
                                        try_integration=try_integration)
    
    # print(fcn)
    
    if likelihood.physicalize:
        n_fun_params = nparams
        nparams  = n_fun_params + 2
        n_extra = 2
    else:
        n_fun_params = nparams
        n_extra = 0
    # # print(eq)
    # # sys.exit()
    # # print('nparams', n_extra)
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
            # print('HERE')
            rho0, rs = sympy.symbols("rho0 rs", real=True)
            eq_numpy = sympy.lambdify([x, a0, rho0, rs], eq, modules=["jax"])
        else:
            eq_numpy = sympy.lambdify([x, a0], eq, modules=["jax"])


    # G = 4.302e-6  # kpc km^2 / (Msun s^2)

    # def r1_from_a0(a0):
    #     return 0.0731 * (a0)**(3/2)

    # def rs_from_r1(r1):
    #     return r1 / 2.16258

    # def rho_s_from_a0_r1(a0, r1):
    #     A = 2.16258 * (1 + 2.16258)**2  
    #     numerator = A * a0**2
    #     denominator = 2 * jnp.pi * G * r1**2
    #     return numerator / denominator

    # def rho_sis(r, a0):
    #     return a0**2 / (2 * jnp.pi * G * r**2)

    # def rho_nfw(r, rs, rho_s):
    #     x = r / rs
    #     return rho_s / (x * (1 + x)**2)

    # def eq_numpy(r_Mpc, a0):
    #     a0 = jnp.abs(a0)
    #     r = r_Mpc * 1000
    #     r1 = r1_from_a0(a0)
    #     rs = rs_from_r1(r1)
    #     rho_s = rho_s_from_a0_r1(a0, r1)

    #     rho = jnp.where(r <= r1,
    #                     rho_sis(r, a0),
    #                     rho_nfw(r, rs, rho_s))
    #     return rho * 1000**3 /1e12


    # grad_density = jax.grad(eq_numpy, argnums=0)(30., *params)
    # print('grad_density', grad_density)
    # print('value', eq_numpy(30., *params))

    # return chi2, chi2, params, params

    # print(eq_numpy(1, *params))

    # eds = ExcessSurfaceDensity.calculate(likelihood.xvar, eq_numpy, params=params)
    # # print(eds)
    # params1 = np.array([100, 7.93551118e+01, 6.48554867e-01])
    # eds1 = ExcessSurfaceDensity.calculate(likelihood.xvar, eq_numpy, params=params1)
    # print(eds1)
            
    def get_fop(chi2_fcn):
        def fop(x):
            # print(chi2_fcn(x, likelihood.xvar, likelihood.yvar, likelihood.yerr))
            return chi2_fcn(x, likelihood.xvar, likelihood.yvar, likelihood.yerr)[0]
        return fop

    loss_template = likelihood.get_loss(eq_numpy)
    chi2_fcn = likelihood.get_wrapped_like(loss_template) #change this
    fop = get_fop(chi2_fcn)

    # print('xvar', likelihood.xvar)
    # print('yvar', likelihood.yvar)
    # print('yerr', likelihood.yerr)

    # print('AQUI', fop([23.216849, 0.21663236,-1.1718663]))

    # 

    # params2 = np.array([0.00309482, 948.87853, 3.6805378])
    # params2 = np.array([ 7.0770481e+00, -3.7901390e-06, 11.479928, 0.26384255])
    # print('other params', fop(params2))
    
    # sys.exit()
    

                            
    if likelihood.is_mse:
        print('Not computing DL as using MSE')
        DL = np.nan
        negloglike = chi2
    else:
        # print('params AQUI', params)
        p = np.copy(params)
        # (3) Obtain the Fisher matrix for this function
        # print(fcn)
        fcn, eq = likelihood.run_sympify(fstr,
                                                tmax=tmax,
                                                try_integration=try_integration)
        params_convert, negloglike_convert, deriv, codelen = convert_params(
            fcn, eq, p, likelihood, chi2, max_param=max_param)
        
        # params_convert, negloglike_convert, deriv, codelen = p, chi2, np.zeros(max_param), np.inf
        
        print('params_convert', params_convert, negloglike_convert, codelen, deriv)

        #if convert params fails (there is a cutoff or other porblems), calculate codelen from integral
        # codelen = np.nan
        if np.isnan(codelen) or np.isinf(codelen) or (codelen < 0):
            print('Calculating codelen from integral')
            negloglike = chi2

            if nparams == 0:
                eq_numpy = sympy.lambdify([x], eq, modules=["jax"])
            elif nparams == 1:
                eq_numpy = sympy.lambdify([x, a0], eq, modules=["jax"])
            else:
                all_a = ' '.join([f'a{i}' for i in range(nparams)])
                all_a = list(sympy.symbols(all_a, real=True))
                eq_numpy = sympy.lambdify([x] + all_a, eq, modules=["jax"])

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
                    # print('HERE')
                    rho0, rs = sympy.symbols("rho0 rs", real=True)
                    eq_numpy = sympy.lambdify([x, a0, rho0, rs], eq, modules=["jax"])
                else:
                    eq_numpy = sympy.lambdify([x, a0], eq, modules=["jax"])

            # print('params here', params)
            codelen, Delta = get_codelen(likelihood, params, nparams, max_param, fstr, eq_numpy, negloglike)
            print('codelen', codelen, Delta)
            # sys.exit()

            # print(chi2.dtype)

        else:
            # print('AQUI', eq_numpy(x, *params))
            if np.all(eq_numpy(likelihood.xvar, *params) == 0):
                params = np.zeros(max_param)
                fish = np.zeros(max_param)
                Delta = np.zeros(max_param)
                codelen = 0

            params = params_convert
            chi2 = negloglike_convert
            fisher_diag = get_fisher_diag_from_deriv(deriv, nparams, max_param=max_param)
            Delta = np.sqrt(12./fisher_diag)
            # print('Delta', Delta)


        #snap params to 0
        def get_fop(chi2_fcn):
            def fop(x):
                return chi2_fcn(x, likelihood.xvar, likelihood.yvar, likelihood.yerr)[0]
            return fop
        Nsteps = abs(np.array(params))/Delta
        # print('Nsteps before snapping', Nsteps)
        #if extra params, don't condiser them for snapping
        if np.sum(Nsteps<1)>0:
            if n_extra > 0:
                N_steps_extra = Nsteps[-n_extra:]
                Delta_extra = Delta[-n_extra:]
                Delta_extra[N_steps_extra <1] = abs(params[-n_extra:][N_steps_extra <1])
                Delta[-n_extra:] = Delta_extra
                Nsteps[-2:] = 1
                # print('Nsteps', Nsteps, Delta)
            loss_template = likelihood.get_loss(eq_numpy)
            chi2_fcn = likelihood.get_wrapped_like(loss_template) #change this
            fop = get_fop(chi2_fcn)
            # print('Nsteps', Nsteps)
            chi2_before = chi2
            params_before = np.copy(params)
            Delta_before = np.copy(Delta)
            codelen_before = codelen
            chi2, params, Delta, mask = snap_params(chi2, params,  Delta, Nsteps, nparams, fop)
            # print('params snapped', params, Delta, chi2)
            # print('mask', mask) 
            params_codelen = params[mask]
            # print('Delta_codelen', Delta)
            Delta_codelen = Delta[mask]
            Sigma = Delta_codelen/np.sqrt(12.)
            codelen = codelength(params_codelen, Sigma, len(params_codelen))

            dl_after = chi2 + codelen
            dl_param = np.inf
            codelen_param = codelen_before
            if np.any(Delta_before > np.abs(params_before)):
                safe_params_before = np.abs(params_before)
                safe_mask_before = safe_params_before > 0
                if np.any(safe_mask_before):
                    # Delta_param = safe_params_before
                    Delta_param = np.minimum(Delta_before, safe_params_before)
                    Delta_param_codelen = Delta_param[safe_mask_before]
                    params_param_codelen = params_before[safe_mask_before]
                    Sigma_param = Delta_param_codelen / np.sqrt(12.)
                    codelen_param = codelength(params_param_codelen, Sigma_param, len(params_param_codelen))
                    dl_param = chi2_before + codelen_param
            print(
                f"Snap DL: param={dl_param:.12f} after={dl_after:.12f} "
                f"nparams_before={len(params_before)} nparams_after={len(params_codelen)}"
            )
            # Keep the best of: snapped or Delta=|params| (only if Delta>params).
            if np.isfinite(dl_param) and (not np.isfinite(dl_after) or dl_param <= dl_after):
                chi2 = chi2_before
                params = params_before
                Delta = np.abs(params_before)
                codelen = codelen_param

            # print('AQUI', eq_numpy(x, *params))
            # if np.all(eq_numpy(likelihood.xvar, *params) == 0):
            #     params = np.zeros(max_param)
            #     fish = np.zeros(max_param)
            #     Delta = np.zeros(max_param)
            #     codelen = 0

                # if not np.allclose(params, 0):
                #     eq_values = eq_numpy(likelihood.xvar, *params)
                #     if np.count_nonzero(eq_values) == 0:
                #         print('HERE')
                #     #if np.all(eq_numpy(xvar, *params_no_zeroes) == 0) and params_no_zeroes != np.zeros(len(params_no_zeroes)):
                #         fop = get_fop(chi2_fcn, total_param=nparams)
                #         all_zeroes = np.zeros(len(params))
                #         if fop(all_zeroes) == negloglike:
                #             #print(fcn_i)
                #             #print(params_no_zeroes)
                #             p = np.zeros(max_param)
                #             fish = np.zeros(max_param)
                #             Delta = np.zeros(max_param)
                #             codelen = 0


        if verbose:
            print('\ntheta_ML:', params)
            print('Residuals:', chi2)
            print('Parameter:', codelen)

        #need to add the part where it tries to cut params to 0 

        # (4) Get the functional complexity
        param_list = ['a%i'%j for j in range(max_param)]
        aifeyn = generator.aifeyn_complexity(labels, param_list)
        # aifeyn = 18.61
        # print('aifeyn', aifeyn)
        # sys.exit()
        if verbose:
            print('Function:', aifeyn)

        # (5) Combine to get description length
        DL = chi2 + codelen + aifeyn
        if verbose:
            print('\nDescription length:', DL)
            
    if return_params:
        print('FINAL:', chi2, params, Delta, codelen, count_lowest, j, success)
        return chi2, DL, params, Delta, codelen, aifeyn
    
    # print('best:', chi2, params, codelen, aifeyn)

    return chi2, DL, codelen, aifeyn

    # return chi2, params
    
    
def fit_from_string(fun, basis_functions, likelihood, pmin=0, pmax=5, tmax=5,
    try_integration=False, verbose=False, Niter=30, Nconv=5, maxvar=20,
    log_opt=False, replace_floats=False, return_params=False, method='BFGS'):
    """Run end-to-end fitting of function for a single function, given as a string.
    Note that this is not guaranteed to find the optimimum representation as a tree,
    so there could be a lower description-length representation of the function
    
    Args:
        :fun (str): String representation of the function to be fitted
        :basis_functions (list): list of lists basis functions. basis_functions[0] are
            nullary, basis_functions[1] are unary and basis_functions[2] are binary
            operators
        :likelihood (fitting.likelihood object): object containing data, likelihood
            functions and file paths
        :pmin (float, default=0.): minimum value for each parameter to consider when
            generating initial guess
        :pmax (float, default=3.): maximum value for each parameter to consider when
            generating initial guess
        :tmax (float, default=5.): maximum time in seconds to run any one part of
            simplification procedure for a given function
        :try_integration (bool, default=False): when likelihood requires integral,
            whether to try to analytically integrate (True) or just numerically
            integrate (False)
        :verbose (bool, default=True): Whether to print results (True) or not (False)
        :Niter (int, default=30): Maximum number of parameter optimisation iterations
            to attempt.
        :Nconv (int, default=5): If we find Nconv solutions for the parameters which
            are within a logL of 0.5 of the best, we say we have converged and stop
            optimising parameters
        :maxvar (int): The maximum number of variables which could appear in the
            function
        :log_opt (bool, default=False): whether to optimise 1 and 2 parameter cases in
            log space
        :replace_floats (bool, default=False): whether to replace any numbers found in
            the function with variables to optimise
        :return_params (bool, default=False): whether to return the parameters of the
            maximum likelihood point
    
    Returns:
         :negloglike (float): the minimum value of -log(likelihood) (corresponding to
            the maximum likelihood)
         :DL (float): the description length of this function
         :labels (list): list of strings giving node labels of tree
         :params (optional, list): the maximum likelihood parameters. Only returned if
            `return_params` is true
    
    """

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
    # print(labels)
    res = single_function(
            labels,
            basis_functions,
            likelihood,
            pmin=pmin,
            pmax=pmax,
            tmax=tmax,
            try_integration=try_integration,
            verbose=verbose,
            Niter=Niter,
            Nconv=Nconv,
            log_opt=log_opt,
            return_params=return_params,
            method=method,
    )

    
    if return_params:
        return res[0], res[1], labels, res[2], res[3], res[4], res[5]
    
    return res[0], res[1], labels, res[4], res[5]
    
    
def tree_to_aifeyn(labels, basis_functions, verbose=True):
    """
    Takes a list of labels defining a function and returns the AIFeyn term of
    complexity and the complexity of the function
    
    Args:
        :labels (list): list of strings giving node labels of tree
        :basis_functions (list): list of lists basis functions. basis_functions[0] are
            nullary, basis_functions[1] are unary and basis_functions[2] are
            binary operators
        :verbose (bool, default=True): Whether to print results (True) or not (False)
    
    Returns:
        :aifeyn (float): the contribution to description length from describing tree
        :complexity (int): the number of nodes in the function
    """

    # Convert the string to a sympy function
    s = generator.labels_to_shape(labels, basis_functions)
    success, _, tree = generator.check_tree(s)
    fstr = generator.node_to_string(0, tree, labels)
    max_param = simplifier.get_max_param([fstr], verbose=verbose)
    
    # Get the functional complexity
    param_list = ['a%i'%j for j in range(max_param)]
    aifeyn = generator.aifeyn_complexity(labels, param_list)
    if verbose:
        print('Function:', aifeyn)

    return aifeyn, len(labels)
    
    
def string_to_aifeyn(fun, basis_functions, maxvar=20, verbose=True,
    replace_floats=False):
    """
    Takes a string defining a function and returns the AIFeyn term of
    complexity and the complexity of the function

    Args:
        :fun (str): String representation of the function to be fitted
        :basis_functions (list): list of lists basis functions. basis_functions[0] are
            nullary, basis_functions[1] are unary and basis_functions[2] are
            binary operators
        :maxvar (int, default=20): The maximum number of variables which could appear
            in the function
        :verbose (bool, default=True): Whether to print results (True) or not (False)
        :replace_floats (bool, default=False): whether to replace any numbers found in
            the function with variables to optimise

    Returns:
        :aifeyn (float): the contribution to description length from describing tree
        :complexity (int): the number of nodes in the function
    """

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

    return tree_to_aifeyn(labels, basis_functions, verbose=verbose)
