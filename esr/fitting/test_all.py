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

from esr.fitting.sympy_symbols import *
import esr.generation.simplifier as simplifier
import nlopt


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


def initial_guess(nparam, pmin, pmax, lhs = False):
        pmin = -1
        pmax = 1
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

        return inpt

def optimise_with_nlopt(chi2_fcn, x0, xvar, yvar, yerr, signs, nparam, method):
    def objective_function(params, grad):
        chi2 = chi2_fcn(params, xvar, yvar, yerr, signs)
        # print(chi2)
        if grad.size > 0:
            grad[:] = chi2[1]
            print(grad)
            print(chi2[0])
        return float(chi2[0])
    
    if method == 'Nelder-Mead':
        opt = nlopt.opt(nlopt.LN_NELDERMEAD, int(nparam))
    else:
        opt = nlopt.opt(nlopt.LN_SBPLX, int(nparam))
    opt.set_min_objective(objective_function)
    opt.set_xtol_rel(1e-4)
    opt.set_ftol_rel(1e-4)

    x = opt.optimize(x0)
    minf = opt.last_optimum_value()
    retcode = opt.last_optimize_result()
    success = retcode > 0

    res = {'x': x, 'fun': minf, 'retcode': retcode, 'success': success}


    return res

    
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

def optimise_with_base_hoping(fcn_i, likelihood, tmax, pmin, pmax, comp=0, try_integration=False, log_opt=False, max_param=4, Niter_params=[40,60], Nconv_params=[-5,20], test_success=False, ignore_previous_eqns=True, method='BFGS'):
    xvar, yvar, yerr= likelihood.xvar, likelihood.yvar, likelihood.yerr
    
    nparam = simplifier.count_params([fcn_i], max_param)[0]
    params = np.zeros(max_param)

    if comp>1 and ignore_previous_eqns:
        previous_fns_file = likelihood.fn_dir + "/compl_"+str(comp)+"/previous_eqns_"+str(comp)+".txt"
        with open(previous_fns_file, "r") as f:
            previous_fns = f.readlines()
        # discard repeat of lower complexity (e.g. [inv, inv, ...])
        if fcn_i in previous_fns:
            # print('Ignoring:', fcn_i, flush=True)
            return np.inf, params, 0, 0

    Niter = int(np.sum(nparam ** np.arange(len(Niter_params)) * np.array(Niter_params))) + 100

    # try:
    fcn_i, eq = likelihood.run_sympify(fcn_i, tmax=tmax, try_integration=try_integration)

    if ("a0" in fcn_i)==False:
        eq_numpy = sympy.lambdify(x, eq, modules=["jax"])
        loss_template = likelihood.get_loss(eq_numpy)
        chi2_fcn = likelihood.get_wrapped_like(loss_template)
        chi2_i =  chi2_fcn([], xvar, yvar, yerr, None)

        return chi2_i[0], params, 0, 0

    flag_three = False

    if nparam > 1:
        all_a = ' '.join([f'a{i}' for i in range(nparam)])
        all_a = list(sympy.symbols(all_a, real=True))
        eq_numpy = sympy.lambdify([x] + all_a, eq, modules=["numpy"])
    else:
        eq_numpy = sympy.lambdify([x, a0], eq, modules=["numpy"])


    bad_fun = True
    for p in itertools.product([1, -1], repeat=nparam):
        if not (np.sum(np.isnan(eq_numpy(xvar,*p)))>0):
            bad_fun = False
            break
    
    if bad_fun:
        # Don't bother trying to optimise bc this fcn is clearly really bad
        chi2_i = np.inf
        # print('Bad function:', fcn_i, flush=True)
        return chi2_i, params, 0, 0
        
    # Reset chi2
    chi2_min = np.inf
        
    if nparam > 2:
        flag_three = True
    
    #OPT
    loss_template = likelihood.get_loss(eq_numpy)
    chi2_fcn = likelihood.get_wrapped_like(loss_template)

    grad_template = likelihood.get_loss(eq_numpy, value = 'grad')
    grad_fcn = likelihood.get_wrapped_like(grad_template)


    #Basehoping
    inpt = initial_guess(nparam, pmin, pmax, lhs = True)
    signs = None
    minimizer_kwargs = {
        "method": "Nelder-Mead",  # You can choose a different method if needed
        "args": (xvar, yvar, yerr, signs),
        "jac": True,  # If your function returns the gradient as well
        "options": {"fatol": 1e-3}
    }

    res = basinhopping(chi2_fcn, inpt, minimizer_kwargs=minimizer_kwargs, niter=300)

    params = np.pad(np.array(res['x']), (0, max_param-len(res['x'])))
    # Output the result
    # print("Global minimum: x = ", res.x, ", f(x) = ", res.fun)

    return res.fun, params, 0, 0, True
    
    
    
def optimise_fun(fcn_i, likelihood, tmax, pmin, pmax, comp=0, try_integration=False, log_opt=False, max_param=4, Niter_params=[40,60], Nconv_params=[-5,20], test_success=False, ignore_previous_eqns=True, method='BFGS'):
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


    xvar, yvar, yerr= likelihood.xvar, likelihood.yvar, likelihood.yerr
    
    nparam = simplifier.count_params([fcn_i], max_param)[0]
    params = np.zeros(max_param)

    if comp>1 and ignore_previous_eqns:
        previous_fns_file = likelihood.fn_dir + "/compl_"+str(comp)+"/previous_eqns_"+str(comp)+".txt"
        with open(previous_fns_file, "r") as f:
            previous_fns = f.readlines()
        # discard repeat of lower complexity (e.g. [inv, inv, ...])
        if fcn_i in previous_fns:
            # print('Ignoring:', fcn_i, flush=True)
            return np.inf, params, 0, 0, False
    Niter = int(np.sum(nparam ** np.arange(len(Niter_params)) * np.array(Niter_params))) + 100
    Nconv = int(np.sum(nparam ** np.arange(len(Nconv_params)) * np.array(Nconv_params)))

    # Niter = 400
    # Nconv = 400
    # print(Niter, Nconv, flush=True)
    if nparam > 0:
        if (Nconv <= 0) or (Niter <= 0) or (Nconv > Niter):
            raise ValueError("Nconv and/or Niter have unacceptable values")

    try:
        fcn_i, eq = likelihood.run_sympify(fcn_i, tmax=tmax, try_integration=try_integration)

        if ("a0" in fcn_i)==False:
            eq_numpy = sympy.lambdify(x, eq, modules=["jax"])
            loss_template = likelihood.get_loss(eq_numpy)
            chi2_fcn = likelihood.get_wrapped_like(loss_template)
            chi2_i =  chi2_fcn([], xvar, yvar, yerr, None)

            return chi2_i[0], params, 0, 0, False

        flag_three = False

        mult_arr = np.ones(max_param)
        count_lowest = 0
        inf_count = 0
        exception_count = 0

                
        if nparam > 1:
            all_a = ' '.join([f'a{i}' for i in range(nparam)])
            all_a = list(sympy.symbols(all_a, real=True))
            eq_numpy = sympy.lambdify([x] + all_a, eq, modules=["numpy"])
        else:
            eq_numpy = sympy.lambdify([x, a0], eq, modules=["numpy"])


        bad_fun = True
        for p in itertools.product([1, -1], repeat=nparam):
            if not (np.sum(np.isnan(eq_numpy(xvar,*p)))>0):
                bad_fun = False
                break
        
        if bad_fun:
            # Don't bother trying to optimise bc this fcn is clearly really bad
            chi2_i = np.inf
            # print('Bad function:', fcn_i, flush=True)
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


        for j in range(Niter):
            inpt = initial_guess(nparam, pmin, pmax, lhs = True)
            if log_opt:
                all_sign_combinations = list(product([1, -1], repeat=nparam))

                res = {'fun': np.inf, 'success': False}
                res_fun = np.inf
                mult_arr = []
                for signs in all_sign_combinations:
                    res_iteration = minimize_scipy(chi2_fcn, inpt, jac=True,  args=(xvar, yvar, yerr, None), options=dict(gtol = 1e-3), method=method)
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
                # res = optimise_with_nlopt(chi2_fcn, inpt, xvar, yvar, yerr, signs, nparam, method)
                # print(j, res['success'], res['fun'], res['x'], flush=True)            
                    
            if test_success and (not res['success']):
                continue

            if np.isinf(res['fun']):
                inf_count += 1

            # Failure if first 50 all give inf
            if inf_count==30 and np.isinf(chi2_min):
                break

            # Reset count if log-like improves by 2
            if res['fun']-chi2_min < -2.:
                count_lowest=0

            # If within 0.5 of lowest, say converged to that value
            if abs(res['fun']-chi2_min) < 0.5:
                count_lowest += 1

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
                params = np.pad(np.array(best['x']), (0, max_param-len(best['x'])))
            else:
                # Params put in linear space and sign added back in
                params = np.pad(10.**np.array(best['x']), (0, max_param-len(best['x']))) * mult_arr_best
        elif not np.isfinite(chi2_min):
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

    except Exception as e:
        return np.nan, params, 0, 0

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

    return chi2_i, params, count_lowest, j, res['success']
    
    
def main(comp, likelihood, tmax=300, pmin=0, pmax=3, print_frequency=50, try_integration=False, log_opt=False, Niter_params=[40,60], Nconv_params=[-5,20], ignore_previous_eqns=True, method = 'BFGS'):
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
    
    if rank == 0:
        print('\nRunning fits', flush=True)

    fcn_list_proc, _, _ = get_functions(comp, likelihood)

    
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
        try:
            with simplifier.time_limit(tmax):
                start = time.time()
                try:
                        #Do BFGS
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
                                                        method='BFGS')
                        
                        #Do Nelder-Mead
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
                                                        method='Nelder-Mead')
                        
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
        except:
            chi2[i] = np.nan
            params[i,:] = 0.

        time_taken = time.time() - start

        N_conv[i] = count_lowest
        N_iter[i] = j
        times[i] = time_taken
        success_rate[i] = success


        # num_false_comprehension = len([element for element in success_rate if not element])
        # print(f"Number of False elements (list comprehension): {num_false_comprehension}")  
        

        print(i, fcn_list_proc[i], chi2[i], params[i,:], count_lowest, j, method_used, flush=True)

    # out_arr = np.transpose(np.vstack([chi2] + [params[:,i] for i in range(max_param)]))

    out_arr = np.vstack([chi2] + [params[:, i] for i in range(max_param)] + [N_conv, N_iter, times])
    out_arr = np.transpose(out_arr)

    # Save the data for this proc in Partial
    np.savetxt(likelihood.temp_dir + '/chi2_comp'+str(comp)+'weights_'+str(rank)+'.dat', out_arr, fmt='%.7e')
    
    comm.Barrier()

    if rank == 0:
        string = 'cat `find ' + likelihood.temp_dir + '/ -name "chi2_comp'+str(comp)+'weights_*.dat" | sort -V` > ' + likelihood.out_dir + '/negloglike_comp'+str(comp)+ '_' + method + '.dat'
        os.system(string)
        string = 'rm ' + likelihood.temp_dir + '/chi2_comp'+str(comp)+'weights_*.dat'
        os.system(string)
        
    comm.Barrier()

    return

