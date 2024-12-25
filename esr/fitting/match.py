import numpy as np
import math
import sympy
from mpi4py import MPI
import warnings
import os
import sys
import numdifftools as nd
import itertools
import esr.fitting.test_all as test_all
import esr.fitting.test_all_Fisher as test_all_Fisher
from esr.fitting.sympy_symbols import *
import matplotlib.pyplot as plt

import esr.generation.simplifier as simplifier
import jax.numpy as jnp
import scipy
from jax import vmap
import jax

warnings.filterwarnings("ignore")

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def get_sigma_from_integral(p, Sigma, fcn_i, negloglike_all, nparams, max_fun_params, fop, number_points=10**3):
        # j is the index of the parameter in the list of all parameters
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
            
        # def get_boundary(res):
        #     return 10**res.x[0]
        
        def get_boundary(res, log=True):
            if log:
                return 10**res.x[0]
            else:
                return res.x[0]
        
        # def get_boundary(res_plus, res_minus, theta):
        #     if res_plus.success == False:
        #         boundary = 10**res_minus.x[0]
        #     elif res_minus.success == False:
        #         boundary = 10**res_plus.x[0]
        #     else:
        #         boundary_right = 10**res_plus.x[0]
        #         boundary_left = 10**res_minus.x[0]
        #         boundary = max(boundary_right, boundary_left)
        #     return boundary
        
        def get_integral(theta, boundary, param, fop, negloglike, number_points=5*10**2):
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

        # param = p[:nparams]
        param = np.append(p[:nparams], p[max_fun_params:])

        arg_integral = np.append(np.argwhere(Sigma <= 0), np.argwhere(Sigma == np.inf))
        arg_integral = np.append(arg_integral, np.argwhere(np.isnan(Sigma)))

        #Plot likelihood
        # if fcn_i == 'a0*x + x':
        #     Delta_plot = 0.1
        #     x_range = np.linspace(param - Delta_plot, param + Delta_plot, 10**2)

        #     nll = []
        #     for x_value in x_range:
        #         negloglike = fop(x_value)
        #         nll = np.append(nll, negloglike)

        #     plt.plot(x_range, np.exp(-nll + jnp.min(nll)))
        #     # plt.plot(theta_ML, np.exp(-chi2_fcn(theta_ML, xvar, yvar, yerr) + jnp.min(nll)), 'ro')
        #     plt.show()

        # xx = param + 10**(-2)
        # yy = fop(xx)
        # print(yy, flush=True)

        loss_template = get_loss()
        likelihood_fcn = wrap_loss(loss_template)
        # print('fcn_i', fcn_i, flush=True)
        # print('param', param, flush=True)
        # print('fop', fop(param), flush=True)
        
        for j in arg_integral:
                theta = param[j]
                # print('ESTO', param, theta)

                initial_guesses = [-20, -10, -5, 0.5, 1, 5, np.log10(np.abs(theta))]
                initial_guesses = np.sort(initial_guesses)

                # initial_guesses_log_space_small = [-20, -10, -5]
                # initial_guesses_normal_space = np.sort([0, np.abs(theta)])
                # initial_guesses_log_space_large = [1, 5, 10]

                # initial_guesses = [(val, True) for val in initial_guesses_log_space_small] + \
                #   [(val, False) for val in initial_guesses_normal_space] + \
                #   [(val, True) for val in initial_guesses_log_space_large]

                # initial_guesses.sort(key=lambda x: 10**x[0] if x[1] else x[0])

                for factor in [-1, 1]:
                    for initial_guess in initial_guesses:
                        is_log = True
                        # print('initial_guess', initial_guess, flush=True)
                        boundary_found = False
                        # print('initial_guess', initial_guess, flush=True)

                        # print('factor', factor, flush=True)
                        # print('theta', theta, flush=True)
                        # print('param', param, flush=True)
                        # print('j', j, flush=True)
                        # print(fop(param), flush=True)

                        res = scipy.optimize.minimize(likelihood_fcn, initial_guess, args=(param, j, factor), method='Nelder-Mead', tol=1e-8)
                        # print('res_minus', res)

                        log = is_log 

                        if test_success(res)== True:
                            if factor == -1:
                                boundary_left = get_boundary(res, log)
                                print('boundary_left', boundary_left, flush=True)
                            else:
                                boundary_right = get_boundary(res, log)
                                print('boundary_right', boundary_right, flush=True)
                            boundary_found = True
                            break

                    if not boundary_found:
                        break

                if not boundary_found:
                    print(f"Couldn't find integral limits for factor {factor}, function {fcn_i}, parameter {theta}", flush=True)
                    Sigma[j] = np.inf
                    continue

                #do integral
                boundary = np.max([boundary_left, boundary_right])
            
                #do integral
                integral, a_range = get_integral(theta, boundary, param, fop, negloglike_all, number_points=number_points)
                                        
                #get the 68% confidence interval from the integral
                arg_min = np.argmin(abs(0.68 - integral/integral[-1]))
                param68 = a_range[arg_min + 1]
                sigma = np.abs(theta - param68)
                # print('sigma', sigma, flush=True)
                Sigma[j] = sigma
        # print('SIGMA', Sigma)

        return Sigma

def main(comp, likelihood, tmax=5, print_frequency=1000, try_integration=False):
    """Apply results of fitting the unique functions to all functions and save to file
    
    Args:
        :comp (int): complexity of functions to consider
        :likelihood (fitting.likelihood object): object containing data, likelihood functions and file paths
        :tmax (float, default=5.): maximum time in seconds to run any one part of simplification procedure for a given function
        :print_frequency (int, default=1000): the status of the fits will be printed every ``print_frequency`` number of iterations
        :try_integration (bool, default=False): when likelihood requires integral, whether to try to analytically integrate (True) or just numerically integrate (False)
        
    Returns:
        None
        
    """
    def get_eq_numpy(nparams, fcn_i, try_integration, tmax=5.):
        fcn_i, eq = likelihood.run_sympify(fcn_i, tmax=tmax, try_integration=try_integration)
        # if nparams == 0:
        #     eq_numpy = sympy.lambdify([x], eq, modules=["jax"])
        # elif nparams == 1:
        #     eq_numpy = sympy.lambdify([x, a0], eq, modules=["jax"])
        # else:
        #     all_a = ' '.join([f'a{i}' for i in range(nparams)])
        #     all_a = list(sympy.symbols(all_a, real=True))
        #     eq_numpy = sympy.lambdify([x] + all_a, eq, modules=["jax"])

        if nparams == 0:
            if likelihood.physicalize:
                rho0, rs = sympy.symbols("rho0 rs", real=True)
                eq_numpy = sympy.lambdify([x, rho0, rs], eq, modules=["jax"])
            else:
                eq_numpy = sympy.lambdify([x], eq, modules=["jax"])
        elif nparams > 1:
            all_a = ' '.join([f'a{i}' for i in range(nparams)])
            all_a = list(sympy.symbols(all_a, real=True))
            if likelihood.physicalize:
                all_a += sympy.symbols("rho0 rs", real=True)
            eq_numpy = sympy.lambdify([x] + all_a, eq, modules=["jax"])
        else:
            if likelihood.physicalize:
                rho0, rs = sympy.symbols("rho0 rs", real=True)
                eq_numpy = sympy.lambdify([x, a0, rho0, rs], eq, modules=["jax"])
            else:
                eq_numpy = sympy.lambdify([x, a0], eq, modules=["jax"])

        return eq_numpy


    if likelihood.is_mse:
        raise ValueError('Cannot use MSE with description length')


    # Data
    xvar = likelihood.xvar
    yvar = likelihood.yvar
    yerr = likelihood.yerr


    def get_fop(chi2_fcn, total_param):
        if total_param > 0:
            def fop(x):
                return chi2_fcn(x, xvar, yvar, yerr)
        else:
            def fop(x):
                return chi2_fcn([x], xvar, yvar, yerr)
        return fop
        
    if rank == 0:
        print('\nMatching', flush=True)
    
    invsubs_file = likelihood.fn_dir + "/compl_%i/inv_subs_%i.txt"%(comp,comp)
    match_file = likelihood.fn_dir + "/compl_%i/matches_%i.txt"%(comp,comp)
    
    fcn_list_proc, data_start, data_end = test_all.get_functions(comp, likelihood, unique=False)
    negloglike, params_meas, Nconv, Niter, times = test_all_Fisher.load_loglike(comp, likelihood, data_start, data_end, split=False)
    # print(Nconv, Niter, times, flush=True)
    max_param = params_meas.shape[1]
    # max_param =7

    all_inv_subs_proc = simplifier.load_subs(invsubs_file, max_param)[data_start:data_end]
    matches_proc = np.atleast_1d(np.loadtxt(match_file).astype(int))[data_start:data_end]

    all_fish = np.loadtxt(likelihood.out_dir + '/derivs_comp'+str(comp)+'.dat')   # 2D array of shape (# unique fcns, 10)
    all_fish = np.atleast_2d(all_fish)


    codelen = np.zeros(len(fcn_list_proc))              # Both of these are also just for this proc
    negloglike_all = np.zeros(len(fcn_list_proc))
    index_arr = np.zeros(len(fcn_list_proc))
    params = np.zeros([len(fcn_list_proc), max_param])
    Nconv_all = np.zeros(len(fcn_list_proc))
    Niter_all = np.zeros(len(fcn_list_proc))
    times_all = np.zeros(len(fcn_list_proc))
    Deltas = np.zeros([len(fcn_list_proc), max_param])

    for i in range(len(fcn_list_proc)):                 # The part of all eqs analysed by this proc
        if i%print_frequency==0 and rank==0:
            print(i, len(fcn_list_proc))


        fcn_i = fcn_list_proc[i].replace('\'', '')

        # print('funcion', fcn_i, flush=True)

        # print('ALL_EQS', fcn_i, flush=True)

        # if fcn_i == 'Abs(a0)/x':
        #     print('HERE')
        
        #number of parameters from the function
        nparams = simplifier.count_params([fcn_i], max_param)[0]

        if likelihood.physicalize:
            extra_params = 2
        else:
            extra_params = 0

        #total number of parameters
        total_param = nparams + extra_params
        max_fun_params = 4
        max_param = max_fun_params + extra_params

        index = matches_proc[i]          # Index in total unique eqs file, common to all procs

        index_arr[i] = index

        negloglike_all[i] = negloglike[index]           # Assign the likelihood of this variant to the that of the unique eq
        Nconv_all[i] = Nconv[index]
        Niter_all[i] = Niter[index]
        times_all[i] = times[index]

        fcn_i, eq = likelihood.run_sympify(fcn_i, tmax=tmax)

        #CHANGE THIS - take this out actually
        try:
            eq_numpy = get_eq_numpy(nparams, fcn_i, try_integration, tmax)
            loss_template = likelihood.get_loss(eq_numpy, value = 'evaluate')
            chi2_fcn =likelihood.get_wrapped_like(loss_template)
        except:
            print('Error with function:', fcn_i)
            codelen[i] = np.inf
            continue
        # try:

        #     if nparams == 0:
        #         eq_numpy = sympy.lambdify([x], eq, modules=["jax"])
        #     elif nparams > 1:
        #         all_a = ' '.join([f'a{i}' for i in range(nparams)])
        #         all_a = list(sympy.symbols(all_a, real=True))
        #         eq_numpy = sympy.lambdify([x] + all_a, eq, modules=["jax"])
        #     else:
        #         eq_numpy = sympy.lambdify([x, a0], eq, modules=["jax"])
        # except:
        #     print('Error with function:', fcn_i)
        #     codelen[i] = np.inf
        #     continue

        
        xvar = likelihood.xvar
        yvar = likelihood.yvar
        yerr = likelihood.yerr

        if np.isnan(negloglike[index]) or np.isinf(negloglike[index]):          # Element of the unique eqs file, common to all procs
            codelen[i] = np.nan
            continue

        if total_param==0:
            continue
        else:
            # k = nparams
            # measured = params_meas[index,:nparams].copy()
            k = max_param
            measured = params_meas[index,:k].copy()
        
        fish_measured = all_fish[index,:]               # Access from the unique eqs all_fish array, common to all procs
        try:
            if nparams > 0:
                print(measured, fish_measured)
                p, fish = simplifier.convert_params(measured[:nparams], fish_measured, all_inv_subs_proc[i], n=max_param)
                Sigma = 1/np.sqrt(fish)

                fish = np.zeros((len(measured), len(measured)))
                fish[np.triu_indices(len(measured))] = fish_measured
                fish = np.where(fish, fish, fish.T)
                Sigma_extra = 1/np.sqrt(np.diag(fish[max_fun_params:, max_fun_params:]))
                Sigma = np.array(list(Sigma) + list(Sigma_extra))
                p = np.append(p, measured[nparams:])

                #p and SIgma have different lengths

                #add the thing here
                # print('first Sigma', Sigma)
                if isinstance(p, float):
                    p=[p]
                p = np.atleast_1d(p)

            else:
                p = measured
                fish = np.zeros((len(measured), len(measured)))
                fish[np.triu_indices(len(measured))] = fish_measured
                fish = np.where(fish, fish, fish.T)
                Sigma = 1/np.sqrt(np.diag(fish[max_fun_params:, max_fun_params:]))
                Sigma = np.array(Sigma)

            if isinstance(p, float):
                p=[p]
            p = np.atleast_1d(p)

        except Exception as ex:
            codelen[i] = np.inf
            continue

        
        # if np.sum(fish<=0)>0:
        #     codelen[i] = np.inf
        #     print('HERE')
        #     continue

        #If Sigma is not defined we need to compute it integrating
        if (np.sum(Sigma <= 0.) > 0.) or (np.sum(np.isnan(Sigma)) > 0) or (np.sum(np.isinf(Sigma)) > 0):#  or (np.sum(Nsteps<1) > 0):
            # print('here')
            # try:
            #     eq_numpy = get_eq_numpy(nparams, fcn_i, try_integration, tmax)
            #     loss_template = likelihood.get_loss(eq_numpy,  value = 'evaluate')
            #     chi2_fcn =likelihood.get_wrapped_like(loss_template)
            
            # except Exception:
            #     print("BAD:", fcn_i, negloglike[index], np.isfinite(negloglike[index]))
            
            # print('Integrating', fcn_i, flush=True)
            # print('nparams', nparams, flush=True)
            fop = get_fop(chi2_fcn, total_param=total_param)
            negloglike_fcn = negloglike_all[i]
            # print('negloglike_fcn', negloglike_fcn)
            params_no_zeroes = np.append(p[:nparams], p[max_fun_params:])
            # print('params_no_zeroes', params_no_zeroes)
            # print('fop', fop(params_no_zeroes), flush=True)


            #here p and Sigma have the shape of max_params
            # print('SIgmassss', nparams, flush=True)
            # print('p', p, flush=True)
            Sigma = get_sigma_from_integral(p, Sigma, fcn_i, negloglike_fcn, nparams, max_fun_params, fop, number_points=10**3)
            # print('Sigma', Sigma, flush=True)
            # print(p, flush=True)
            if np.isnan(Sigma).any():
                Sigma[np.isnan(Sigma)] = np.inf

        fish = 1/Sigma**2
        # fish = np.concatenate((fish[:nparams], np.zeros(max_param - nparams)))
        fish = np.concatenate((fish[:nparams], np.zeros(max_param - nparams - extra_params), fish[nparams:]))
        
        try:
            # p = np.append(p, np.zeros(max_param - len(p)))
            # Sigma = np.concatenate((Sigma[:nparams], np.zeros(max_param - nparams)))
            Sigma = np.concatenate((Sigma[:nparams], np.zeros(max_param - nparams - extra_params), Sigma[nparams:]))
            
            #delta has the same shape as p
            Delta = np.zeros(Sigma.shape)
            m = (Sigma != np.inf)
            # Delta = np.zeros(fish.shape)
            # m = (fish != 0)
            # Delta[m] = np.atleast_1d(np.sqrt(12./fish[m]))
            Delta[m] = np.atleast_1d(np.sqrt(12.)*Sigma[m])
            Delta[~m] = np.inf
            Nsteps = np.atleast_1d(np.abs(np.array(p)))
            m = (Delta != 0)
            Nsteps[m] /= Delta[m]
            Nsteps[~m] = np.nan

            #Nsteps has shape of params taking out zeroes
            # Nsteps = np.concatenate(Nsteps[:nparams], Nsteps[max_fun_params:])

            # print("I'M HERE", fcn_i, Delta, Nsteps)
        except:
            print('Error with function:', fcn_i)
            codelen[i] = np.inf
            continue
        
        negloglike_orig = np.copy(negloglike_all[i])
        # p = np.concatenate((p[:nparams], p[max_fun_params:])) #take out zeroes
        ptrue=np.copy(p)
        Delta_true = np.copy(Delta)
        # print(fcn_i, p, Delta)

        #Set delta=theta for all galaxy params with Nsteps<1 -- don't set to 0 because we want to keep the parameter in the model
        #to here p, Sigma and Nsteps have hape of max_param
        Nsteps_gal = Nsteps[nparams:]
        if np.sum(Nsteps_gal<1)>0:
            Delta[nparams:][Nsteps_gal<1] = abs(p[nparams:][Nsteps_gal<1])
            fish[np.argwhere(Nsteps_gal<1)] = 12./(p[np.argwhere(Nsteps_gal<1)]**2)
            Nsteps[nparams:][Nsteps_gal<1] = 1
        
        if np.sum(Nsteps<1)>0:         # should reevaluate -log(L) with the param(s) set to 0, but doesn't matter unless the fcn is a very good one
            # print('Nsteps despues de esto', Nsteps)
            try:
                p[Nsteps<1] = 0.         # Set any parameter to 0 that doesn't have at least one precision step, and recompute -log(L).
                # print('p', p, flush=True)
            except (IndexError, TypeError):
                p=0.
            
            # try:            # It's possible that after putting params to 0 the likelihood is botched, in which case give it nan
            #     # fcn_i, eq = likelihood.run_sympify(fcn_i, tmax=tmax, try_integration=try_integration)
            #     # if k==1:
            #     #     eq_numpy = sympy.lambdify([x, a0], eq, modules=["jax"])
            #     #     fop = get_fop(chi2_fcn, total_param=nparams)
            #     #     negloglike_all[i] = fop(p)               # Modified here for this variant, but if this doesn't happen it stays the same as the unique eq
            #     #     # print('negloglike_all[i]', negloglike_all[i])
            #     # else:
            #     #     all_a = ' '.join([f'a{i}' for i in range(nparams)])
            #     #     all_a = list(sympy.symbols(all_a, real=True))
            #     #     eq_numpy = sympy.lambdify([x] + all_a, eq, modules=["jax"])
            #     #     fop = get_fop(chi2_fcn, total_param=nparams)
            #     #     negloglike_all[i] = fop(p)
                

            # except NameError:
            #     if try_integration:
            #         fcn_i, eq = likelihood.run_sympify(fcn_i, tmax=tmax, try_integration=False)
            #         if k==1:
            #             eq_numpy = sympy.lambdify([x, a0], eq, modules=["jax"])
            #             negloglike_all[i] = fop(p)               # Modified here for this variant, but if this doesn't happen it stays the same as the unique eq
            #         else:
            #             all_a = ' '.join([f'a{i}' for i in range(nparams)])
            #             all_a = list(sympy.symbols(all_a, real=True))
            #             eq_numpy = sympy.lambdify([x] + all_a, eq, modules=["jax"])
            #             fop = get_fop(chi2_fcn, total_param=nparams)
            #             negloglike_all[i] = fop(p)
            #     else:
            #         negloglike_all[i] = np.nan

            # except:
            #     negloglike_all[i] = np.nan
                
            # fop = get_fop(chi2_fcn, total_param=nparams)
            fop = get_fop(chi2_fcn, total_param=total_param)
            params_no_zeroes = np.append(p[:nparams], p[max_fun_params:])
            negloglike_all[i] = fop(params_no_zeroes)
                
            if np.isfinite(negloglike_all[i]):
                k -= np.sum(Nsteps<1)
                # kept_mask = Nsteps>=1

                kept_mask = np.ones(len(ptrue), dtype=bool)
                kept_mask[np.argwhere(Nsteps<1)] = 0
                #shape of p, Delta and fish, kept_mask is max_param

                nparams = nparams - np.sum(Nsteps<1)
                max_fun_params = max_fun_params - np.sum(Nsteps<1)
                # print('p aqui en el if', p, flush=True)
                #Add something here

            else:
                # Let's see if setting any of the parameters to zero is ok
                try_idx = np.arange(max_param)[Nsteps < 1]
                for r in reversed(range(1, len(try_idx))):
                    for idx in itertools.combinations(try_idx, r):
                        p = np.copy(ptrue)
                        for idx_ in idx:
                            p[idx_] = 0.
                        # if k==1:
                        #     fop = get_fop(chi2_fcn, total_param=nparams)
                        #     negloglike_all[i] = fop(p)               # Modified here for this variant, but if this doesn't happen it stays the same as the unique eq
                        # else:
                        #     # fop = get_fop(chi2_fcn, total_param=len(p))
                        #     negloglike_all[i] = fop(p)
                            
                        p_params = np.append(p[:nparams], p[max_fun_params:])
                        negloglike_all[i] = fop(p_params)
                        if np.isfinite(negloglike_all[i]):
                            nparams = nparams - len(idx)
                            max_fun_params = max_fun_params - len(idx)

                            Delta[np.argwhere(Nsteps<1)] = abs(p[np.argwhere(Nsteps<1)])
                            break
                kept_mask = np.ones(len(p), dtype=bool)
                if np.isfinite(negloglike_all[i]):
                    k -= len(idx)
                    kept_mask[idx] = 0
                # elif not np.isfinite(negloglike_all[i]) and not np.isnan(negloglike_all[i]): # infinite nll
                elif not np.isfinite(negloglike_all[i]):
                    p = ptrue
                    # fish= fish[:nparams]
                    fish[Nsteps<1] = 12./(p[Nsteps<1]**2) # set uncertainty=parameter in this case
                    Delta[np.argwhere(Nsteps<1)] = abs(p[np.argwhere(Nsteps<1)])
                    # codelen[i] = -k/2.*math.log(3.) + np.sum( 0.5*np.log(fish) + np.log(abs(np.array(p))) )
                    # p = np.append(p, np.zeros(max_param - len(p)))

                    # Delta_codelen = Delta[:nparams]
                    # p_codelen = p[:nparams]
                    Delta_codelen = np.append(Delta[:nparams], Delta[max_fun_params:])
                    p_codelen = np.append(p[:nparams], p[max_fun_params:])
                    codelen[i] = k*math.log(2.) + np.sum(np.log(abs(np.array(p_codelen))/Delta_codelen))
                    negloglike_all[i] = negloglike_orig
                    # print(i, fcn_i, codelen[i], negloglike_all[i], flush=True)
                    try:        # If p was an array, we can make a list out of it
                        list_p = list(p)
                        # params[i,:] = np.pad(p, (0, max_param-len(p)))
                        # Deltas[i,:] = np.pad(Delta, (0, max_param-len(Delta)))
                        params[i,:] = p
                        Deltas[i,:] = Delta
                    except:     # p is either a number or nothing
                        if p:   # p is a number
                            params[i,:] = 0
                            params[i,0] = p
                            Deltas[i,:] = 0
                            Deltas[i,0] = Delta
                        else:
                            params[i,:] = np.zeros(max_param)
                            Deltas[i,:] = np.zeros(max_param)
                    
                    assert len(params[i,:])==max_param
                    continue

            if k<0:
                print("This shouldn't have happened", flush=True)
                quit()
            elif k==0:                  # If we have no parameters left then the parameter codelength is 0 so we can move on
                continue
            
            # print('HOLA', fish, nparams, kept_mask, flush=True)
            #fish = fish[:nparams][kept_mask]      # Only consider these parameters in the codelen
            # Delta = Delta[:nparams][kept_mask]
            fish = fish[kept_mask]
            p = p[kept_mask]
            Delta = Delta[kept_mask]
            
        else:
            kept_mask = np.ones(len(p), dtype=bool)
        
        # print(fcn_i, p, Delta)
        
        try:
            # p = np.append(p, np.zeros(max_param - len(p)))
                # codelen[i] = -k/2.*math.log(3.) + np.sum( 0.5*np.log(fish) + np.log(abs(np.array(p))) )
            # Delta = Delta[:nparams]
            # Delta_codelen = Delta[:nparams][kept_mask]
            # print(np.sum(np.log(abs(np.array(p))/Delta)))

            p2 = np.append(p[:nparams], p[max_fun_params:])
            Delta2 = np.append(Delta[:nparams], Delta[max_fun_params:])
            codelen[i] = k*math.log(2.) + np.sum(np.log(abs(np.array(p2))/Delta2))
        except:
            codelen[i] = np.nan
        
        p = ptrue
        p[~kept_mask]=0.
        Delta = Delta_true
        Delta[~kept_mask]=0.
        # Delta = np.append(Delta[:nparams], Delta[max_fun_params:])
        # Delta = Delta[:nparams]
        # Delta[~kept_mask]=0.
        # p and Delta have the shape of the free parameters in the func
            
        try:        # If p was an array, we can make a list out of it
            list_p = list(p)
            # sys.exit()
            params[i,:] = p
            Deltas[i,:] = Delta
            # params[i,:] = np.pad(p, (0, max_param-len(p)))
            # Deltas[i,:] = np.pad(Delta, (0, max_param-len(Delta)))
        except:     # p is either a number or nothing
            if p:   # p is a number
                params[i,:] = 0
                params[i,0] = p
                Deltas[i,:] = 0
                Deltas[i,0] = Delta
            else:
                params[i,:] = np.zeros(max_param)
                Deltas[i,:] = np.zeros(max_param)
        
        assert len(params[i,:])==max_param

        # print('HERE', params[i,:], Deltas[i,:])

        print(i, fcn_i, codelen[i], negloglike_all[i], flush=True)
        

    # print(codelen, flush=True)
        
    for i in range(len(params)):          # Loop over all unique fcns to find variant with min codelength
        for j in range(len(params[i])):
            if Deltas[i,j] == 0 and params[i,j] != 0:
                print(fcn_list_proc[i], params[i], Deltas[i])
    
    fcn_list_proc = np.array(fcn_list_proc)
    print(fcn_list_proc[np.argwhere(codelen == np.inf)])
    print(fcn_list_proc[np.argwhere(codelen == - np.inf)])
    # print(params)
    # print(Deltas)
    # print(codelen)
    # out_arr = np.transpose(np.vstack([negloglike_all, codelen, index_arr] + [params[:,i] for i in range(max_param)]))

    out_arr = np.vstack([negloglike_all, codelen, index_arr] + [params[:,i] for i in range(max_param)] + [Deltas[:,i] for i in range(max_param)] +  [Nconv_all, Niter_all, times_all])
    out_arr = np.transpose(out_arr)

    np.savetxt(likelihood.temp_dir + '/codelen_matches_'+str(comp)+'_'+str(rank)+'.dat', out_arr, fmt='%.7e')        # Save the data for this proc in Partial

    comm.Barrier()

    if rank == 0:
        string = 'cat `find ' + likelihood.temp_dir + '/ -name "codelen_matches_'+str(comp)+'_*.dat" | sort -V` > ' + likelihood.out_dir + '/codelen_matches_comp'+str(comp)+'.dat'
        os.system(string)
        string = 'rm ' + likelihood.temp_dir + '/codelen_matches_'+str(comp)+'_*.dat'
        os.system(string)
        
    comm.Barrier()
        
    return

