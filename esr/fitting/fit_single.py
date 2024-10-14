import sys
import numpy as np

from esr.fitting.test_all import optimise_fun
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



def get_sigma_from_integral(p, Sigma, fcn_i, negloglike_all, nparams, max_fun_params, fop, number_points=10**3):
        #j is the index of the parameter we are looking at
        #i is the index of the data point we are looking at
        def fraction_likelihood(x, p, j, right):
            if right:
                factor = 1
            else:
                factor = -1

            # if k == 1:  
            #     return np.abs(fop([x]) - negloglike_all[i] - np.log(1e4))  
            # else:
            params = np.copy(p)
            # print('param', params, flush=True)
            params[j] = params[j] + factor*10**(x)
            # print('after', params, flush=True)
            # print(params, flush=True)
            # print('fop', fop(params))
            # print(negloglike_all[i])
            # print(fop(params) - negloglike_all)
            return np.abs(fop(params) - negloglike_all - np.log(1e3))
        
        def get_boundary(res_plus, res_minus, theta):
            if res_plus.success == False:
                boundary = 10**res_minus.x[0]
            elif res_minus.success == False:
                boundary = 10**res_plus.x[0]
            else:
                boundary_right = 10**res_plus.x[0]
                boundary_left = 10**res_minus.x[0]
                boundary = max(boundary_right, boundary_left)
            return boundary
        
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
            # p = np.copy(param)
            # p[j] = param[j] + factor*10**(res.x[0])
            # print(param)
            # print(p)
            # likelihood_diff =  fop(p) - negloglike_all
            # print('likelihood_diff', likelihood_diff, flush=True)
            # print(np.abs(fop(p) - negloglike_all - np.log(1e4)))
            if res.success == False or res.fun > 0.1:
            # if res.success == False or likelihood_diff <= np.log(5*1e2):
                return False
            else:
                return True

        param = p[:nparams]
        # initial_guess_array = [10**(-6), 10**(-5), 10**(-4), 10**(-3), 10**(-2), 10**(-1), 1, 10, 10**2, 10**3]
        # # neglog_initial_guess = (fop(test_initial_guess) for test_initial_guess in initial_guess_array)
        # neglog_initial_guess = jnp.array([fop(np.array([test_initial_guess])) for test_initial_guess in initial_guess_array])
        # print('neglog_initial_guess', neglog_initial_guess, flush=True)
        # initial_guess = initial_guess_array[jnp.isfinite(neglog_initial_guess)[-1]]
        # print('initial_guess', initial_guess, flush=True)

        # initial_guesses = [np.ones(nparams), np.ones(nparams)*(-1)]
        initial_guesses = [np.zeros(nparams), np.zeros(nparams)]
        # initial_guesses = [np.(param), np.log10(param)*(-1)]

        # xx = param + 10**(-2)
        # yy = fop(xx)
        # print(yy, flush=True)
        # param = [27.86277272,  1.87778243]
        # for j in [1]:
        for j in range(nparams):
                theta = param[j]

                #Plot likelihood
                Delta_plot = 0.1
                x_range = np.linspace(theta - Delta_plot, theta + Delta_plot, 10**2)
                params_range = np.tile(param, (len(x_range), 1))
                params_range[:, j] = x_range

                nll = []
                for params in params_range:
                    negloglike = fop(params)
                    nll = np.append(nll, negloglike)        
                plt.plot(x_range, np.exp(-nll + jnp.min(nll)))
                plt.yscale('log')
                # plt.plot(theta_ML, np.exp(-chi2_fcn(theta_ML, xvar, yvar, yerr) + jnp.min(nll)), 'ro')
                plt.show()
                sys.exit()


                for initial_guess in initial_guesses:
                    # print('initial_guess', initial_guess, flush=True)
                    res_minus = scipy.optimize.minimize(fraction_likelihood, initial_guess[j], args=(param, j, False), method='Nelder-Mead', tol=1e-8)
                    res_plus = scipy.optimize.minimize(fraction_likelihood, initial_guess[j], args=(param, j, True), method = 'Nelder-Mead', tol=1e-8)
                    # print('res_minus', res_minus, 'res_plus', res_plus, flush=True)
                    # print(theta - 10**res_minus.x[0], theta + 10**res_plus.x[0], flush=True)
                    # print(fop([1.4182408815858817e+49]))
                    # print(fop([1.4182408815858817e+49])- negloglike_all)
                    # print(np.log(1e3))
                    # print(fop([27.86277272,  1.4182408815858817e+49]) - negloglike_all)

                    # print(test_success(res_minus), test_success(res_plus), flush=True)

                    if test_success(res_minus)== True or test_success(res_plus)==True: 
                        # print("HERE")
                        break        

                if test_success(res_minus)== False and test_success(res_plus)==False:
                    print("Couldn't find integral limits", fcn_i, flush=True)
                    sigma = np.inf
                    continue

                boundary = get_boundary(res_plus, res_minus, theta)
                print('boundary', boundary, flush=True)
            
                #do integral
                integral, a_range = get_integral(theta, boundary, param, fop, negloglike_all, number_points=number_points)
                                        
                #get the 68% confidence interval from the integral
                # print(integral/integral[-1])
                # print(theta+boundary)
                arg_min = np.argmin(abs(0.68 - integral/integral[-1]))
                param68 = a_range[arg_min + 1]
                # print(param68)
                sigma = np.abs(theta - param68)
                Sigma[j] = sigma
        # print(Sigma)

        return Sigma

def codelength(params, Sigma, nparams):
    k = nparams
    Delta = np.atleast_1d(np.sqrt(12.)*Sigma)
    if np.isinf(Delta).any():
        Delta[Delta == np.inf] = params[Delta == np.inf]
    codelen = k*math.log(2.) + np.sum(np.log(abs(np.array(params))/Delta))
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
        Sigma = get_sigma_from_integral(params, np.zeros(nparams), fstr, negloglike, nparams, max_param, get_fop(chi2_fcn, nparams), number_points=10**3)
        codelen = codelength(params, Sigma, nparams)
        return codelen



def single_function(labels, basis_functions, likelihood, method, pmin=0, pmax=5, tmax=5,
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
    print(fstr)
    # (2) Fit this function to the data
    Niter = 300
    Nconv = 20
    chi2, params, count_lowest, j, success = optimise_fun(fstr,
                            likelihood,
                            tmax,
                            pmin,
                            pmax,
                            try_integration=try_integration,
                            max_param=max_param,
                            Niter_params=[Niter],
                            Nconv_params=[Nconv],
                            log_opt=log_opt,
                            method=method)
    
    print('best:', chi2, params,count_lowest, j, success)
    

                            
    if likelihood.is_mse:
        print('Not computing DL as using MSE')
        DL = np.nan
        negloglike = chi2
    else:
        # (3) Obtain the Fisher matrix for this function
        fcn, eq = likelihood.run_sympify(fstr,
                                                tmax=tmax,
                                                try_integration=try_integration)
        params_convert, negloglike_convert, deriv, codelen = convert_params(
            fcn, eq, params, likelihood, chi2, max_param=max_param)


        #if convert params fails (there is a cutoff or other porblems), calculate codelen from integral
        if np.isnan(codelen) or np.isinf(codelen):
            negloglike = chi2

            if nparams == 0:
                eq_numpy = sympy.lambdify([x], eq, modules=["jax"])
            elif nparams == 1:
                eq_numpy = sympy.lambdify([x, a0], eq, modules=["jax"])
            else:
                all_a = ' '.join([f'a{i}' for i in range(nparams)])
                all_a = list(sympy.symbols(all_a, real=True))
                eq_numpy = sympy.lambdify([x] + all_a, eq, modules=["jax"])
            codelen = get_codelen(likelihood, params, nparams, max_param, fstr, eq_numpy, negloglike)
        else:
            params = params_convert
            chi2 = negloglike_convert

        if verbose:
            print('\ntheta_ML:', params)
            print('Residuals:', chi2)
            print('Parameter:', codelen)

        # (4) Get the functional complexity
        param_list = ['a%i'%j for j in range(max_param)]
        aifeyn = generator.aifeyn_complexity(labels, param_list)
        if verbose:
            print('Function:', aifeyn)

        # (5) Combine to get description length
        DL = chi2 + codelen + aifeyn
        if verbose:
            print('\nDescription length:', DL)
            
    if return_params:
        return chi2, DL, params

    return chi2, DL, params

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
    print(labels)
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
            method=method
    )
    
    if return_params:
        return res[0], res[1], labels, res[2]
    
    return res[0], res[1], labels
    
    
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
