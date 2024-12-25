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
from WL_likelihood import WLLikelihood



def single_function(labels, basis_functions, likelihood, method, pmin=0, pmax=5, tmax=5,
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
    print(fstr)
    # (2) Fit this function to the data
    Niter = 300
    Nconv = 20
    chi2, params, count_lowest, j, success = mixed_optimise_fun(fstr,
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

    return chi2, params, count_lowest, j, success
    # sys.exit()
    

                            
    # if likelihood.is_mse:
    #     print('Not computing DL as using MSE')
    #     DL = np.nan
    #     negloglike = chi2
    # else:
    #     # (3) Obtain the Fisher matrix for this function
    #     fcn, eq = likelihood.run_sympify(fstr,
    #                                             tmax=tmax,
    #                                             try_integration=try_integration)
    #     params_convert, negloglike_convert, deriv, codelen = convert_params(
    #         fcn, eq, params, likelihood, chi2, max_param=max_param)


    #     #if convert params fails (there is a cutoff or other porblems), calculate codelen from integral
    #     if np.isnan(codelen) or np.isinf(codelen):
    #         negloglike = chi2

    #         if nparams == 0:
    #             eq_numpy = sympy.lambdify([x], eq, modules=["jax"])
    #         elif nparams == 1:
    #             eq_numpy = sympy.lambdify([x, a0], eq, modules=["jax"])
    #         else:
    #             all_a = ' '.join([f'a{i}' for i in range(nparams)])
    #             all_a = list(sympy.symbols(all_a, real=True))
    #             eq_numpy = sympy.lambdify([x] + all_a, eq, modules=["jax"])
    #         codelen = get_codelen(likelihood, params, nparams, max_param, fstr, eq_numpy, negloglike)
    #     else:
    #         params = params_convert
    #         chi2 = negloglike_convert

    #     if verbose:
    #         print('\ntheta_ML:', params)
    #         print('Residuals:', chi2)
    #         print('Parameter:', codelen)

    #     # (4) Get the functional complexity
    #     param_list = ['a%i'%j for j in range(max_param)]
    #     aifeyn = generator.aifeyn_complexity(labels, param_list)
    #     if verbose:
    #         print('Function:', aifeyn)

    #     # (5) Combine to get description length
    #     DL = chi2 + codelen + aifeyn
    #     if verbose:
    #         print('\nDescription length:', DL)
            
    # if return_params:
    #     return chi2, DL, params

    # return chi2, DL, params


if __name__ == '__main__':
    # (1) Load the data
    data_file = 'XXL/cluster_91.txt'
    run_name = 'WL_cluster_91'
    likelihood = WLLikelihood(data_file, run_name)

    # (2) Define the function
    basis_functions = [["x", "a"],  # type0
                ["inv", "abs", "log", "exp"],  # type1
                ["+", "*", "-", "/", "pow"]]  # type2
    
    fn = 'a0/x^2 + a1/x'
    
    # print(run_name)

    #Likelihood
    likelihood = WLLikelihood(data_file, run_name, data_dir=None, fn_set = 'core_maths') 

    chi2, DL, labels, params = single_function(fn, basis_functions, likelihood, method = 'Nelder-Mead', try_integration = False, verbose=True, log_opt=True, return_params=True)