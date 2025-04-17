import numpy as np
import sympy
import warnings
import os
import sys
from mpi4py import MPI
from scipy.optimize import minimize
import itertools
import jax.numpy as jnp
from scipy.optimize import minimize as minimize_scipy
from scipy.optimize import basinhopping
from itertools import product
import scipy.stats
from scipy.stats import qmc
import time
import matplotlib.pyplot as plt

from esr.fitting.sympy_symbols import *
import esr.generation.simplifier as simplifier
from esr.esd import ExcessSurfaceDensity
#import nlopt


warnings.filterwarnings("ignore")

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def chi2_fcn(x, likelihood, eq_numpy, integrated, signs):
    """Compute chi2 for a function
    
    Args:
        :x (list): parameters to use for function
        :likelihood (fitting.likelihood object): object containing data and likelihood function
        :eq_numpy (numpy function): function to pass to likelihood object to make prediction of y(x)
        :integrated (bool): whether eq_numpy has already been integrated
        :signs (list): each entry specifies whether than parameter should be optimised logarithmically. If None, then do nothing, if '+' then optimise 10**x[i] and if '-' then optimise -10**x[i]
        
    Returns:
        :negloglike (float): - log(likelihood) for this function and parameters
    
    """
    if signs is None:
        p = x
    else:
        p = [None] * len(signs)
        for i in range(len(signs)):
            if signs[i] == None:
                p[i] = x[i]
            elif signs[i] == '+':
                p[i] = 10 ** x[i]
            elif signs[i] == '-':
                p[i] = - 10 ** x[i]
            else:
                raise ValueError
    return likelihood.negloglike(p,eq_numpy, integrated=integrated)


def initial_guess(nparam, pmin, pmax, likelihood, lhs = False):
        pmin = -10
        pmax = 10
        def get_lhs_sample(ndim, bounds):
        # Create a LatinHypercube sampler object
            sampler = qmc.LatinHypercube(d=ndim)

            # Sample a single point within the defined bounds
            sample = sampler.random(n=1)[0]

            # Scale the sample values to match the specified bounds
            scaled_sample = []
            for i in range(ndim):
                low, high = bounds[i]
                scaled_sample.append(low + (sample[i] * (high - low)))

            return np.array(scaled_sample)

        
        def generate_inpt(nparam, pmin, pmax, lhs = False):
            #Function params - derived either from LHS or uniform distributions
            if lhs:
                bounds = [[pmin,pmax] for _ in range(nparam)]
                inpt = get_lhs_sample(len(bounds), bounds)
            else:
                inpt = [np.random.uniform(pmin,pmax) for _ in range(nparam)]
            return inpt

        inpt = generate_inpt(nparam, pmin, pmax, lhs = lhs)

        if likelihood.physicalize:
            #add guess for rho0 and rs
            inpt = np.append(inpt, np.random.uniform(0,pmax))
            inpt = np.append(inpt, np.random.uniform(0,pmax))

        return inpt
    
def get_functions(comp, likelihood, unique=True):
    """Load all functions for a given complexity to use and distribute among ranks
    
    Args:
        :comp (int): complexity of functions to consider
        :likelihood (fitting.likelihood object): object containing data, functions to convert SR expressions to variable of data and file path
        :unique (bool, default=True): whether to load just the unique functions (True) or all functions (False)
        
    Returns:
        :fcn_list (list): list of strings representing functions to be used by given rank
        :data_start (int): first index of function used by rank
        :data_end (int): last index of function used by rank
        
    """

    if unique:
        unifn_file = likelihood.fn_dir + "/compl_%i/unique_equations_%i.txt"%(comp,comp)
    else:
        unifn_file = likelihood.fn_dir + "/compl_%i/all_equations_%i.txt"%(comp,comp)
    
    if comp>=8:
        sys.setrecursionlimit(2000 + 500 * (comp - 8))

    if rank == 0:
        for dirname in [likelihood.base_out_dir, likelihood.out_dir, likelihood.temp_dir]:
            if not os.path.isdir(dirname):
                print('Making dir:', dirname)
                os.mkdir(dirname)
    comm.Barrier()

    if rank==0:
        print("Number of cores:", size, flush=True)

    with open(unifn_file, "r") as f:
        fcn_list = f.readlines()

    nLs = int(np.ceil(len(fcn_list) / float(size)))       # Number of lines per file for given thread

    while nLs*(size-1) > len(fcn_list):
        if rank==0:
            print("Correcting for many cores.", flush=True)
        nLs -= 1

    if rank==0:
        print("Total number of functions: ", len(fcn_list), flush=True)
        print("Number of test points per proc: ", nLs, flush=True)

    data_start = rank*nLs
    data_end = (rank+1)*nLs

    if rank==size-1:
        data_end = len(fcn_list)
    
    return fcn_list[data_start:data_end], data_start, data_end

def combine_params(global_params, local_params, global_index):
    params = np.append(global_params, local_params)
    return params 

def local_optimisation(inpt_global, chi2_fcn, xvar, yvar, yerr, signs, global_index, nlocal, likelihood):
    #need to add more than one iteration
    Niter = 10
    Nconv = 5
    pmin = -10
    pmax = 10

    #divide the data for the different clusters
    points_per_cluster = 9
    nclusters = len(xvar)//points_per_cluster
    xvar = np.array_split(xvar, nclusters)
    yvar = np.array_split(yvar, nclusters)
    yerr = np.array_split(yerr, nclusters)

    total_loss = 0
    local_params_per_cluster = []
    count_lowest_per_cluster = []

    for i in range(nclusters):
        # print(i)
        xvar_i = xvar[i]
        yvar_i = yvar[i]
        yerr_i = yerr[i]

        # take out nans
        no_nans = np.isnan(yvar_i) == False
        xvar_i = xvar_i[no_nans]
        yvar_i = yvar_i[no_nans]
        yerr_i = yerr_i[no_nans]

        chi2_min = np.inf
        count_lowest = 0
        best_params = None

        # if nlocal == 0:
        #     local_params_per_cluster.append([])
        #     count_lowest_per_cluster.append(0)

        #     chi2 = chi2_fcn(inpt_global, xvar_i, yvar_i, yerr_i, signs)[0]
        #     total_loss += chi2
        #     # print(i, chi2)
        #     continue

        for j in range(Niter):
            # print('Iteration local:', j, flush=True)
            inpt_local = initial_guess(nlocal, pmin, pmax, likelihood, lhs=True)
            # Perform local optimization (global params fixed)
            res = minimize(chi2_fcn, inpt_local, jac= True, args=(xvar_i, yvar_i, yerr_i, signs, inpt_global, global_index),
                            method='BFGS', options=dict(gtol=1e-3))
            
            # print(res['x'], flush=True)
            
            if res['fun']-chi2_min < -2.:
                count_lowest=0

            # If within 0.5 of lowest, say converged to that value
            if abs(res['fun']-chi2_min) < 0.5:
                count_lowest += 1
            
            if res['fun'] <= chi2_min:
                chi2_min = res['fun']
                best_params = res.x.copy()

            # Converged the required number of times, so a success
            if count_lowest==Nconv:
                break

        if not np.isfinite(chi2_min):
            count_lowest_per_cluster = np.zeros(nclusters)
            local_params_per_cluster = np.zeros(nclusters*nlocal)
            total_loss += np.inf
            # print('\tFailed to find parameters for function')
            # print('nlocal:', nlocal)
            # return np.zeros(nclusters*nlocal), np.inf, 0
            break

        else:
            count_lowest_per_cluster.append(count_lowest)
            local_params_per_cluster.append(best_params)
            total_loss += chi2_min


    print(len(local_params_per_cluster))
        # print(i, chi2_min, count_lowest)

    return local_params_per_cluster, total_loss, count_lowest_per_cluster

def global_loss(global_params, chi2_fcn, xvar, yvar, yerr, signs, global_index, nlocal, likelihood, state):
    # print('HERE')
    local_params_per_cluster, total_loss, count_lowest_per_cluster = local_optimisation(global_params, chi2_fcn, xvar, yvar, yerr, signs, global_index, nlocal, likelihood)
    
    if total_loss < state['best_total_loss']:
        state['best_total_loss'] = total_loss
        state['best_local_params'] = local_params_per_cluster
        state['best_local_count'] = count_lowest_per_cluster

    # print('local', len(local_params_per_cluster))
    
    
    # print('total_loss', total_loss, local_params_per_cluster)
    # global_loss.best_local_params = local_params_per_cluster
    # global_loss.best_local_count = count_lowest_per_cluster
    return total_loss 

def mixed_optimise_fun(fcn_i, likelihood, global_index, tmax, pmin, pmax, n_clusters, comp=0, try_integration=False, log_opt=False, max_param=4, test_success=False, ignore_previous_eqns=True, method='BFGS', Nconv=400, Niter=400):

    xvar, yvar, yerr= likelihood.xvar, likelihood.yvar, likelihood.yerr
    

    params = np.zeros(max_param)

    nparams = simplifier.count_params([fcn_i], max_param)[0]
    n_fun_params = nparams

    # (1.5) Get the global and local indices
    # global_index = [0]
    nglobal = len(global_index)
    nlocal = nparams - nglobal
    n_extra = 0

    if likelihood.physicalize:
        n_extra = 2
        nlocal = nlocal + n_extra
        nparams = nparams + n_extra
        max_param += n_extra

    # print('n_global:s', n_global, 'n_local:', n_local, 'n_extra:', n_extra, 'nparams:', nparams, flush=True)
    # sys.exit(0)

    # if comp>1 and ignore_previous_eqns:
    #     previous_fns_file = likelihood.fn_dir + "/compl_"+str(comp)+"/previous_eqns_"+str(comp)+".txt"
    #     with open(previous_fns_file, "r") as f:
    #         previous_fns = f.readlines()
    #     if fcn_i in previous_fns:
    #         return np.inf, params, 0, 0, False

    # if nparam > 0:
    #     if (Nconv <= 0) or (Niter <= 0) or (Nconv > Niter):
    #         raise ValueError("Nconv and/or Niter have unacceptable values")

    fcn_i, eq = likelihood.run_sympify(fcn_i, tmax=tmax, try_integration=try_integration)

    # all_a = ' '.join([f'a{i}' for i in range(nparam)])
    # all_a = list(sympy.symbols(all_a, real=True))
    # eq_numpy = sympy.lambdify([x] + all_a, eq, modules=["jax"])

    # if nparam > 1:
    #     all_a = ' '.join([f'a{i}' for i in range(nparam)])
    #     all_a = list(sympy.symbols(all_a, real=True))
    #     eq_numpy = sympy.lambdify([x] + all_a, eq, modules=["jax"])
    # else:
    #     eq_numpy = sympy.lambdify([x, a0], eq, modules=["jax"])

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
    
    #add these params to the function params later
    Niter = 5
    Nconv = 3

    # Reset chi2
    chi2_min = np.inf


    # nglobal = len(global_index)
    # nlocal = nparams - nglobal
    # print('nglobal:', nglobal, 'nparam:', nparam, flush=True)

    loss_template = likelihood.get_loss(eq_numpy)
    chi2_fcn = likelihood.get_wrapped_like(loss_template)

    mult_arr = np.ones(max_param)
    count_lowest = 0
    inf_count = 0
    print('n_clusters:', n_clusters, flush=True)

    for j in range(Niter):

        state = {
            'best_total_loss': np.inf,
            'best_local_params': np.zeros(nlocal*n_clusters),
            'best_local_count': 0
        }

        print('Iteration:', j, flush=True)

        inpt_global = initial_guess(nglobal, pmin, pmax, likelihood, lhs = True)
        signs = None
        method = 'Nelder-Mead'

        if nlocal > 0:
            res = minimize_scipy(global_loss, inpt_global,  args=(chi2_fcn, xvar, yvar, yerr, signs, global_index, nlocal, likelihood, state), options=dict(gtol = 1e-3), method=method)
            # The optimized global parameters
            global_optimized = res.x
            best_local_params = state['best_local_params']
            print('Best local params:', len(best_local_params), flush=True)
            local_count_lowest = state['best_local_count']
            # best_local_params = global_loss.best_local_params
            # local_count_lowest = global_loss.best_local_count
            res.local_count_lowest = local_count_lowest
            # print(res['success'])

            #combine the global and local parameters
            res.x = combine_params(global_optimized, best_local_params, global_index)
            print(res.fun, res.x, flush=True)
            # print(best_local_params)
        elif nlocal == 0:
            #take out nans
            no_nans = np.isnan(yvar) == False
            xvar = xvar[no_nans]
            yvar = yvar[no_nans]
            yerr = yerr[no_nans]
            res = minimize(chi2_fcn, inpt_global, jac= True, args=(xvar, yvar, yerr, signs), method=method, options=dict(gtol=1e-3))
            res.local_count_lowest = 0
        # print(nlocal)
        # print('res:', res.x, flush=True)
            

        #CONVERGENCE CHECKS
        # if test_success and (not res['success']):
        #     continue

        if np.isinf(res['fun']):
            inf_count += 1

        # Failure if first 50 all give inf
        # if inf_count==50 and np.isinf(chi2_min):
        #     break

        # Reset count if log-like improves by 2
        if res['fun']-chi2_min < -2.:
            count_lowest=0

        # If within 0.5 of lowest, say converged to that value
        if abs(res['fun']-chi2_min) < 0.5:
            count_lowest += 1

        if res['fun'] < chi2_min:
            best = res
            chi2_min = res['fun']


        # Converged the required number of times, so a success
        if count_lowest==Nconv:
            break

    if not np.isfinite(chi2_min) and method== 'Nelder-Mead':
        print('\tFailed to find parameters for function:', fcn_i)
        params = res.x
        return np.inf, params, 0, 0, False

    chi2_i = chi2_min
    params = best['x']

    # print('Best:', chi2_i, params, count_lowest, j, best['success'], flush=True)
    print('Local count lowest:', best.local_count_lowest, flush=True)

    return chi2_i, params, count_lowest, j, best['success']
    
    
def optimise_fun(fcn_i, likelihood, tmax, pmin, pmax, comp=0, try_integration=False, log_opt=False, max_param=4, Niter_params=[40,60], Nconv_params=[-5,20], test_success=False, ignore_previous_eqns=True, method='BFGS', Nconv=400, Niter=400):
    """Optimise the parameters of a function to fit data
    
    The list of parameters, P, passed as Niter_params and Nconv_params compute these values, N, to be
    N = P[0] + P[1] * nparam + P[2] * nparam ** 2 + ...
    where nparam is the number of parameters of the function. The order of the polynomial is determined by
    the length of P, so P can be arbirary in length.
    
    Args:
        :fcn_i (str): string representing function we wish to fit to data
        :likelihood (fitting.likelihood object): object containing data and likelihood function
        :tmax (float): maximum time in seconds to run any one part of simplification procedure for a given function
        :pmin (float): minimum value for each parameter to consider when generating initial guess
        :pmax (float): maximum value for each parameter to consider when generating initial guess
        :comp (float, default=0): Complexity. Deafault of 0 because it is not provided when fitting a single function
        :try_integration (bool, default=False): when likelihood requires integral, whether to try to analytically integrate (True) or just numerically integrate (False)
        :log_opt (bool, default=False): whether to optimise 1 and 2 parameter cases in log space
        :max_param (int, default=4): The maximum number of parameters considered. This sets the shapes of arrays used.
        :Niter_params (list, default=[40, 60]): Parameters determining maximum number of parameter optimisation iterations to attempt.
        :Nconv_params (list, default=[-5, 20]): If we find Nconv solutions for the parameters which are within a logL of 0.5 of the best, we say we have converged and stop optimising parameters. These parameters determine Nconv.
        :test_sucess (bool, default=False): Whether to test whether the optimisation was successful using scipy's criteria
        :ignore_previous_eqns (bool, default=True): If we have seen an equation at lower complexity, whether to ignore the equation in this routine.
        
    Returns:
        :chi2_i (float): the minimum value of -log(likelihood) (corresponding to the maximum likelihood)
        :params (list): the maximum likelihood values of the parameters
    
    """

    # print('Optimising:', fcn_i, flush=True)

    print(Niter)

    xvar, yvar, yerr= likelihood.xvar, likelihood.yvar, likelihood.yerr
    
    nparam = simplifier.count_params([fcn_i], max_param)[0]

    # if likelihood.physicalize:
    #     max_param += 2
    
    params = np.zeros(max_param)

    if comp>1 and ignore_previous_eqns:
        previous_fns_file = likelihood.fn_dir + "/compl_"+str(comp)+"/previous_eqns_"+str(comp)+".txt"
        with open(previous_fns_file, "r") as f:
            previous_fns = f.readlines()
        # discard repeat of lower complexity (e.g. [inv, inv, ...])
        if fcn_i in previous_fns:
            # print('Ignoring:', fcn_i, flush=True)
            return np.inf, params, 0, 0, False
    # Niter = int(np.sum(nparam ** np.arange(len(Niter_params)) * np.array(Niter_params))) + 100
    # Nconv = int(np.sum(nparam ** np.arange(len(Nconv_params)) * np.array(Nconv_params)))

    # Niter = 400
    # Nconv = 400
    # print(Niter, Nconv, flush=True)
    if nparam > 0:
        if (Nconv <= 0) or (Niter <= 0) or (Nconv > Niter):
            raise ValueError("Nconv and/or Niter have unacceptable values")

    try:
        fcn_i, eq = likelihood.run_sympify(fcn_i, tmax=tmax, try_integration=try_integration)

        if ("a0" in fcn_i)==False and not likelihood.physicalize:
            eq_numpy = sympy.lambdify(x, eq, modules=["jax"])

            loss_template = likelihood.get_loss(eq_numpy)
            chi2_fcn = likelihood.get_wrapped_like(loss_template)
            chi2_i =  chi2_fcn([], xvar, yvar, yerr, None)

            # print('No parameters:', fcn_i, chi2_i, flush=True)

            return chi2_i[0], params, 0, 0, False
        
        elif ("a0" in fcn_i)==False and likelihood.physicalize:
            rho0, rs = sympy.symbols("rho0 rs", real=True)
            eq_numpy = sympy.lambdify([x, rho0, rs], eq, modules=["jax"])


        flag_three = False

        mult_arr = np.ones(max_param)
        count_lowest = 0
        inf_count = 0
        exception_count = 0

                
        if nparam > 1:
            all_a = ' '.join([f'a{i}' for i in range(nparam)])
            all_a = list(sympy.symbols(all_a, real=True))
            if likelihood.physicalize:
                all_a += sympy.symbols("rho0 rs", real=True)
                eq_numpy = sympy.lambdify([x] + all_a, eq, modules=["jax"])
            else:
                eq_numpy = sympy.lambdify([x] + all_a, eq, modules=["jax"])
        elif nparam == 1:
            if likelihood.physicalize:
                rho0, rs = sympy.symbols("rho0 rs", real=True)
                eq_numpy = sympy.lambdify([x, a0, rho0, rs], eq, modules=["jax"])
                # print('here', eq_numpy(10.,1,1,1))
            else:
                eq_numpy = sympy.lambdify([x, a0], eq, modules=["jax"])

        # import jax
        
        # # grad_density_x = jax.grad(eq_with_params)(10.)
        # grad_density = jax.grad(eq_numpy, argnums=0)(10., 3., 4.)

        # print('here too', eq_numpy(10.,1,1))
        # print('grad_density', grad_density)

        # jax.debug.print("hola {x}", x = grad_density)

        # sys.exit()


        bad_fun = True
        if  likelihood.physicalize:
            ntotal = nparam + 2

        else:
            ntotal = nparam

        
        for p in itertools.product([1, -1], repeat=ntotal):
            try:
                if not (np.sum(np.isnan(eq_numpy(xvar,*p)))>0):
                    bad_fun = False
                    break
            except Exception as e: #probably not an issue cause functions 
                print(e)
                bad_fun = True
        
        if bad_fun:
            # Don't bother trying to optimise bc this fcn is clearly really bad
            chi2_i = np.inf
            print('Bad function:', fcn_i, flush=True)
            return chi2_i, params, 0, 0, False
            
        # Reset chi2
        chi2_min = np.inf
            
        if nparam > 2:
            flag_three = True
        
        #OPT
        loss_template = likelihood.get_loss(eq_numpy)
        chi2_fcn = likelihood.get_wrapped_like(loss_template)

        grad_template = likelihood.get_loss(eq_numpy, value = 'grad')
        grad_fcn = likelihood.get_wrapped_like(grad_template)

        # params = [3.54185284e-03, 6.57679865e+00]
        # chi2 = chi2_fcn(params, xvar, yvar, yerr, signs=None)
        # print(chi2)
        # sys.exit(0)


        # PLOT GRID
        plot_grid = False
        if plot_grid:
            from matplotlib.colors import LogNorm

            loss_template_eval = likelihood.get_loss(eq_numpy, value= 'evaluate')
            chi2_fcn_mcmc = likelihood.get_wrapped_like(loss_template_eval)
            
            #Two params
            #GRID
            p1_range = np.linspace(-2, 30, 100)
            p0_range = np.linspace(-10, 15, 100)

            X, Y = np.meshgrid(p0_range,p1_range)

            likelihood_grid = np.zeros((len(p0_range), len(p1_range)))
            for i, p0 in enumerate(p0_range):
                for j, p1 in enumerate(p1_range):
                    # p = [p0, p1, 70.72968204, 3.91752186, 0.61884448, 1.10634005]
                    # p = [p0, p1, likelihood.inc_true, likelihood.distance_true, likelihood.upsilon_disk_true, likelihood.upsilon_gas_true]
                    p = [p0, p1]
                    negloglike = chi2_fcn_mcmc(p, likelihood.xvar, likelihood.yvar, likelihood.yerr, signs=None)
                    likelihood_grid[j, i] = np.log10(negloglike)
                    # likelihood_grid[j, i] = negloglike

            # Plot the heatmap
            plt.figure(figsize=(8, 6))
            plt.contourf(X, Y, likelihood_grid, levels=50)
            plt.colorbar(label='Log(Negative Log-Likelihood)')
            x_range = np.linspace(-8, 0, 100)
            plt.plot(x_range, - likelihood.xvar[-1]*x_range - likelihood.xvar[-1])
            plt.xlabel('p0')
            plt.ylabel('p1')
            plt.title('Likelihood Grid')
            plt.grid(False)
            plt.show()
            sys.exit(0)

            #Likelihood slice
            # p1 = -5
            # p0_range = np.linspace(0, 8, 100)
            # negloglikes = []
            # esds = []
            # for p0 in p0_range:
            #     p = [p0, p1]
            #     p_10 = [-10**p0, -10**p1]
            #     esd = likelihood.get_pred(p_10, likelihood.xvar, eq_numpy)
            #     # print(p0, esd)
            #     esd = esd[-1]
            #     negloglike = chi2_fcn_mcmc(p, likelihood.xvar, likelihood.yvar, likelihood.yerr, signs=[-1, 1])
            #     # print(esd)
            #     negloglike = np.log10(negloglike)
            #     negloglikes.append(negloglike)
            #     esds.append(esd)
            # plt.plot(p0_range, negloglikes)
            # # plt.plot(p0_range, esds)
            # plt.show()


            #One param
            # p0_range = np.linspace(0, 5, 1000)
            # negloglikes = []
            # esds = []
            # for p0 in p0_range:
            #     p = [p0]
            #     p_10 = [10**p0]
            #     esd = likelihood.get_pred(p_10, likelihood.xvar, eq_numpy)
            #     negloglike = chi2_fcn_mcmc(p, likelihood.xvar, likelihood.yvar, likelihood.yerr, signs=[1])
            #     negloglike = np.log10(negloglike)
            #     negloglikes.append(negloglike)
            #     esds.append(np.sum(esd))
            # plt.plot(p0_range, negloglikes)
            # # plt.plot(p0_range, esds)
            # plt.show()
            # sys.exit(0)


            # esds = []
            # a = [131.39093196]
            # for num_points in [120, 240, 480, 960, 1000, 1920, 3840]:

            #     esd = ExcessSurfaceDensity.calculate(xvar, eq_numpy, params=a, num_points=num_points)
            #     esd = np.sum(esd)
            #     esds.append(esd)


            # plt.plot([120, 240, 480, 960, 1000, 1920, 3840], esds)
            # plt.xlabel('Number of points')
            # plt.ylabel('Sum(ESD)')
            # plt.show()
            # sys.exit()


        for j in range(Niter):
            inpt = initial_guess(nparam, pmin, pmax, likelihood, lhs = True)
            # print(inpt)
            if log_opt:
                all_sign_combinations = list(product([1, -1], repeat=nparam))

                res = {'fun': np.inf, 'success': False}
                res_fun = np.inf
                mult_arr = []
                for signs in all_sign_combinations:
                    res_iteration = minimize_scipy(chi2_fcn, inpt, jac=True,  args=(xvar, yvar, yerr, signs), options=dict(gtol = 1e-3), method=method)
                    # print(j, res_iteration['success'], res_iteration['fun'], res_iteration['x'], flush=True)
                    choose = jnp.argmin(jnp.array([[res_iteration['fun']], [res_fun]])) #check wether it's succesful before choosing
                    # if choose ==0 and res_iteration.success:
                    if choose ==0:
                        res = res_iteration
                        res_fun = res_iteration['fun']
                        mult_arr = signs
                    # print(res.success, res.fun)

            else:
                signs = None
                flag_three = True
                # method = 'Nelder-Mead'
                res = minimize_scipy(chi2_fcn, inpt, jac=True,  args=(xvar, yvar, yerr, signs), options=dict(gtol = 1e-3), method=method)
                # print(j, res['success'], res['fun'], res['x'], flush=True)
                # res = minimize_scipy(chi2_fcn, inpt, jac=False,  args=(xvar, yvar, yerr, signs), options=dict(gtol = 1e-3), method='Nelder-Mead')
                # res = optimise_with_nlopt(chi2_fcn, inpt, xvar, yvar, yerr, signs, nparam, method)
                # print(j, res['success'], res['fun'], res['x'], flush=True)            
                    
            if test_success and (not res['success']):
                continue

            if np.isinf(res['fun']):
                inf_count += 1

            # Failure if first 50 all give inf
            
            if inf_count==50 and np.isinf(chi2_min):
                break

            # Reset count if log-like improves by 2
            if res['fun']-chi2_min < -2.:
                count_lowest=0

            # If within 0.5 of lowest, say converged to that value
            if abs(res['fun']-chi2_min) < 0.5:
                count_lowest += 1
                # print('Converged:', count_lowest, flush=True)
                # res['success'] = (res['success'] or success_iteration_before)

            # success_iteration_before = res['success']

            if res['fun'] < chi2_min:
                best = res
                mult_arr_best = mult_arr
                chi2_min = res['fun']

            # Converged the required number of times, so a success
            if count_lowest==Nconv:
                break
        
        if chi2_min < 1.e100:
            # Optimisation happened. Print something
            if flag_three:
                params = np.zeros(max_param)
                params[:nparam] = np.array(best['x'][:nparam])
                if likelihood.physicalize:
                    params[-2:] = np.array(best['x'][nparam:])
                # params = np.pad(np.array(best['x']), (0, max_param-len(best['x'])))
            else:
                # Params put in linear space and sign added back in
                params = np.zeros(max_param)
                params[:nparam] = 10.**np.array(best['x'][:nparam]) * mult_arr_best[:nparam]

                if likelihood.physicalize:
                    params[-2:] = 10.**np.array(best['x'][nparam:]) * mult_arr_best[nparam:]
                # params = np.pad(10.**np.array(best['x']), (0, max_param-len(best['x']))) * mult_arr_best
        elif not np.isfinite(chi2_min) and method== 'Nelder-Mead':
            print('\tFailed to find parameters for function:', fcn_i)
                    
        # This is after all the iterations, so it's the best we have; reduced chi2
        chi2_i = chi2_min

    except NameError:
        print(NameError)
        # Occurs if function produced not implemented in numpy
        raise NameError

    except simplifier.TimeoutException:
        print('TIMED OUT:', fcn_i, flush=True)
        try:
            if chi2_min < 1.e100:
                if flag_three:
                    params = np.pad(np.array(best.x), (0, max_param-len(best.x)))
                else:
                    params = np.pad(10.**np.array(best.x), (0, max_param-len(best.x))) * mult_arr_best
                chi2_i = chi2_min
            else:
                chi2_i = np.nan
                params[:] = 0.
        except:
            chi2_i = np.nan
            params[:] = 0.

    # except Exception as e:
    #     print('Exception:', e, flush=True)
    #     return np.nan, params, 0, 0

    # maybe check success myself
    # success when gradient is 0 at the minimum
    # best_sol = np.array([-28.6353216 , -40.52691897])
    # print(grad_fcn(best['x'], xvar, yvar, yerr, None))
    # grad = jnp.linalg.norm(grad_fcn(best['x'], xvar, yvar, yerr, None))
    # if grad < 1e-4:
    #     res.success = True
    #     print('Success', grad, flush=True)
    # else:
    #     res.success = False
    #     print('Fail', grad, flush=True)

    # print('chi2:', chi2_i, best['success'], flush=True)

    # success = best['success']
    # print('Final success:', res['success'], flush=True)


    return chi2_i, params, count_lowest, j, res['success']
    
    
def main(comp, likelihood, tmax=120, pmin=0, pmax=3, print_frequency=50, try_integration=False, log_opt=False, Niter_params=[40,60], Nconv_params=[-5,20], ignore_previous_eqns=True, method = 'BFGS'):
    """Optimise all functions for a given complexity and save results to file.
    
    This can optimise in log-space, with separate +ve and -ve branch (except when there are >=3 params in which case it does it in linear)
    
    The list of parameters, P, passed as Niter_params and Nconv_params compute these values, N, to be
    N = P[0] + P[1] * nparam + P[2] * nparam ** 2 + ...
    where nparam is the number of parameters of the function. The order of the polynomial is determined by
    the length of P, so P can be arbirary in length.
    
    Args:
        :comp (int): complexity of functions to consider
        :likelihood (fitting.likelihood object): object containing data, likelihood functions and file paths
        :tmax (float, default=5.): maximum time in seconds to run any one part of simplification procedure for a given function
        :pmin (float, default=0.): minimum value for each parameter to considered when generating initial guess
        :pmax (float, default=3.): maximum value for each parameter to considered when generating initial guess
        :print_frequency (int, default=50): the status of the fits will be printed every ``print_frequency`` number of iterations
        :try_integration (bool, default=False): when likelihood requires integral, whether to try to analytically integrate (True) or just numerically integrate (False)
        :log_opt (bool, default=False): whether to optimise 1 and 2 parameter cases in log space
        :Niter_params (list, default=[40, 60]): Parameters determining maximum number of parameter optimisation iterations to attempt.
        :Nconv_params (list, default=[-5, 20]): If we find Nconv solutions for the parameters which are within a logL of 0.5 of the best, we say we have converged and stop optimising parameters. These parameters determine Nconv.
        :ignore_previous_eqns (bool, default=True): If we have seen an equation at lower complexity, whether to ignore the equation in this routine.
        
    Returns:
        None
    
    """
    # print(likelihood.yvar)
    if rank == 0:
        print('\nRunning fits', flush=True)

    fcn_list_proc, _, _ = get_functions(comp, likelihood)

    # ignore_previous_eqns = False
    if rank==0 and ignore_previous_eqns:
        previous_unifn_list = []
        if comp>1: 
            for compl in range(1,comp):
                unifn_file_i = likelihood.fn_dir + "/compl_%i/unique_equations_%i.txt"%(compl,compl)
                with open(unifn_file_i, "r") as f:
                    fcn_list_i = f.readlines()
                previous_unifn_list += fcn_list_i
        previous_unifn_list = np.array(previous_unifn_list)
        np.savetxt(likelihood.fn_dir + "/compl_"+str(comp)+"/previous_eqns_"+str(comp)+".txt", previous_unifn_list, fmt='%s')

    comm.Barrier()

    
    # Set max param >=4 for backwards compatibility
    max_param = int(max(4, np.floor((comp - 1) / 2)))

    if likelihood.physicalize:
        max_param += 2

    chi2 = np.zeros(len(fcn_list_proc))     # This is now only for this proc
    params = np.zeros([len(fcn_list_proc), max_param])
    N_conv = np.zeros(len(fcn_list_proc))
    N_iter = np.zeros(len(fcn_list_proc))
    times = np.zeros(len(fcn_list_proc))
    success_rate = np.zeros(len(fcn_list_proc))

    print('method:', method, ', log_opt:', log_opt, flush=True)


    for i in range(len(fcn_list_proc)):           # Consider all possible complexities
        if rank == 0 and ((i == 0) or ((i+1) % print_frequency == 0)):
            print(f'{i+1} of {len(fcn_list_proc)}', flush=True)
        
        start = time.time()
        nparam = simplifier.count_params([fcn_list_proc[i]], max_param)[0]
        
            
        try:
            #Do BFGS
            Niter = 400
            Nconvs = [40, 50, 60, 65]
            Nconv = Nconvs[nparam - 1]
            chi2[i], params[i,:], count_lowest, j, success = optimise_fun(fcn_list_proc[i], 
                                            likelihood, 
                                            tmax, 
                                            pmin, 
                                            pmax, 
                                            comp=comp,
                                            try_integration=try_integration,
                                            log_opt=log_opt,
                                            max_param=max_param,
                                            Niter_params=Niter_params,
                                            Nconv_params=Nconv_params,
                                            ignore_previous_eqns=ignore_previous_eqns,
                                            method='BFGS',
                                            Niter=Niter,
                                            Nconv=Nconv)
            
            method_used = 'BFGS'
            # print(chi2[i], fcn_list_proc[i])
            
            # if not success:
                # print(chi2[i], 'BFGS')
            
            # print(chi2[i], params[i,:], count_lowest, j, success, 'BFGS', flush=True)
            
            if (count_lowest < Nconv or not success) and nparam>0:
            #Do Nelder-Mead
                try:
                    with simplifier.time_limit(tmax):
                        Niter = 300
                        Nconvs = [15, 20, 25, 30]
                        Nconv = Nconvs[nparam - 1]
                        chi2NM, paramsNM, count_lowestNM, jNM, successNM = optimise_fun(fcn_list_proc[i], 
                                                        likelihood, 
                                                        tmax, 
                                                        pmin, 
                                                        pmax, 
                                                        comp=comp,
                                                        try_integration=try_integration,
                                                        log_opt=log_opt,
                                                        max_param=max_param,
                                                        Niter_params=Niter_params,
                                                        Nconv_params=Nconv_params,
                                                        ignore_previous_eqns=ignore_previous_eqns,
                                                        method='Nelder-Mead',
                                                        Niter=Niter,
                                                        Nconv=Nconv)
                        
                        #Choose the best of the two
                        choose = jnp.argmin(jnp.array([chi2[i], chi2NM ]))
                        if choose ==1:
                            chi2[i] = chi2NM
                            params[i,:] = paramsNM
                            method_used = 'Nelder-Mead'
                            j = jNM
                            count_lowest = count_lowestNM
                            success = successNM
                        else:
                            method_used = 'BFGS'
                    
                    # print(chi2NM, count_lowestNM, jNM, successNM, 'Nelder-Mead')
                except Exception as e:
                    print(e, flush=True)
                                                                        
                    # print(chi2[i], params[i,:])
        except NameError:
                if try_integration:
                    chi2[i], params[i,:], count_lowest, j, success = optimise_fun(fcn_list_proc[i], 
                                                likelihood, 
                                                tmax, 
                                                pmin, 
                                                pmax, 
                                                comp=comp,
                                                try_integration=False,
                                                log_opt=log_opt,
                                                max_param=max_param,
                                                Niter_params=Niter_params,
                                                Nconv_params=Nconv_params,
                                                ignore_previous_eqns=ignore_previous_eqns)
                else:
                    raise NameError
        except Exception as e:
            print('overall exception:', e, flush=True) 
            chi2[i] = np.nan
            params[i,:] = 0.
            count_lowest = 0
            j = 0
            success = False

        time_taken = time.time() - start

        N_conv[i] = count_lowest
        N_iter[i] = j
        times[i] = time_taken
        success_rate[i] = success


        # num_false_comprehension = len([element for element in success_rate if not element])
        # print(f"Number of False elements (list comprehension): {num_false_comprehension}")  
        

        print(i, fcn_list_proc[i], chi2[i], params[i,:], count_lowest, j, method_used, flush=True)
    # print(N_conv, N_iter, times, flush=True)

    # out_arr = np.transpose(np.vstack([chi2] + [params[:,i] for i in range(max_param)]))

    out_arr = np.vstack([chi2] + [params[:, i] for i in range(max_param)] + [N_conv, N_iter, times])
    out_arr = np.transpose(out_arr)

    # Save the data for this proc in Partial
    np.savetxt(likelihood.temp_dir + '/chi2_comp'+str(comp)+'weights_'+str(rank)+'.dat', out_arr, fmt='%.7e')
    
    comm.Barrier()

    if rank == 0:
        string = 'cat `find ' + likelihood.temp_dir + '/ -name "chi2_comp'+str(comp)+'weights_*.dat" | sort -V` > ' + likelihood.out_dir + '/negloglike_comp'+str(comp) + '.dat'
        os.system(string)
        string = 'rm ' + likelihood.temp_dir + '/chi2_comp'+str(comp)+'weights_*.dat'
        os.system(string)
        
    comm.Barrier()

    return

