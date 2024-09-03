import time
from mpi4py import MPI

import sys
import numpy as np

from esr.fitting.test_all import optimise_fun
from esr.fitting.test_all_Fisher import convert_params

import esr.generation.generator as generator
import esr.generation.simplifier as simplifier

from prettytable import PrettyTable

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

#add optimise function


def run_fit_single(labels, basis_functions, likelihood, method, pmin=0, pmax=5, tmax=5,
    try_integration=False, verbose=False, Niter=30, Nconv=5, log_opt=False,
    return_params=False):

    s = generator.labels_to_shape(labels, basis_functions)
    success, _, tree = generator.check_tree(s)
    fstr = generator.node_to_string(0, tree, labels)
    max_param = simplifier.get_max_param([fstr], verbose=verbose)
    fstr, fsym = simplifier.initial_sympify(
        [fstr], max_param, parallel=False, verbose=verbose)
    fstr = fstr[0]
    fsym = fsym[fstr]
    # (2) Fit this function to the data
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
        params, negloglike, deriv, codelen = convert_params(
            fcn, eq, params, likelihood, chi2, max_param=max_param)
        if verbose:
            print('\ntheta_ML:', params)
            print('Residuals:', negloglike, chi2)
            print('Parameter:', codelen)

        #Add the integral if Fisher doesn't work

        # (4) Get the functional complexity
        param_list = ['a%i'%j for j in range(max_param)]
        aifeyn = generator.aifeyn_complexity(labels, param_list)
        if verbose:
            print('Function:', aifeyn)

        # (5) Combine to get description length
        DL = negloglike + codelen + aifeyn
        if verbose:
            print('\nDescription length:', DL)
            
    if return_params:
        return negloglike, DL, params

    return negloglike, DL, params


def get_fucntional_complexity(labels, verbose=False, max_param=4):
        param_list = ['a%i'%j for j in range(max_param)]
        aifeyn = generator.aifeyn_complexity(labels, param_list)
        if verbose:
            print('Function:', aifeyn)

        return aifeyn

def get_description_length(negloglike, codelen, aifeyn, verbose=False):
        DL = negloglike + codelen + aifeyn
        if verbose:
            print('\nDescription length:', DL)

        return negloglike, DL, params

def calculate_codelen(likelihood, fstr, tmax=5, try_integration=False, verbose=False, return_params=False, max_param=4):

    if likelihood.is_mse:
        print('Not computing DL as using MSE')
        DL = np.nan
        negloglike = chi2
    else:
        fcn, eq = likelihood.run_sympify(fstr,
                                                tmax=tmax,
                                                try_integration=try_integration)
        params, negloglike, deriv, codelen = convert_params(
            fcn, eq, params, likelihood, chi2, max_param=max_param)
        if verbose:
            print('\ntheta_ML:', params)
            print('Residuals:', negloglike, chi2)
            print('Parameter:', codelen)

        #Add the integral if Fisher doesn't work
                

    return negloglike, codelen, params

#Read the function
with open('esr/fitting/output/output_WL/combine_DL_fcn_comp6.dat', "r") as f:         # All
    fcn_list = f.read().splitlines()

# Read data from last file
data = np.genfromtxt('esr/fitting/output/output_WL/combine_DL_comp6_2.dat')

DL = data[:,0]
params = data[:,1:4]
negloglike = data[:,-6]
codelen = data[:,-5]
aifeyn = data[:,-4]
Nconv = data[:,-3]
Niter = data[:,-2]
time_taken = data[:,-1]

xarr = np.linspace(0, len(fcn_list)-1, len(fcn_list)).astype(int)


#with alpha
# vmin = 1e-20 
# DL_min = np.amin(DL[np.isfinite(DL)])
# alpha = DL_min - DL
# alpha = np.exp(alpha)
# m = (alpha > vmin)

#with negloglike
dif_min = 20
negloglike_min = np.amin(negloglike[np.isfinite(DL)])

alpha = negloglike - negloglike_min
m = (alpha < dif_min) & (alpha > 0)

fcn_list = [d for i, d in enumerate(fcn_list) if m[i]]
params = params[m,:]
alpha = alpha[m]
Nconv = Nconv[m]

print(len(fcn_list))
print('fcn_list:', fcn_list)
# print('params:', params)
print('alpha:', alpha)
print('Nconv:', Nconv)


# indices_sort = np.argsort(DL)

# params_sort = params[indices_sort,:]
# fcn_min_sort = [fcn_list[i] for i in indices_sort]

# DL_sort = DL[indices_sort]
# negloglike_sort = negloglike[indices_sort]
# codelen_sort = codelen[indices_sort]
# aifeyn_sort = aifeyn[indices_sort]
# Nconv_sort = Nconv[indices_sort]
# Niter_sort = Niter[indices_sort]
# time_sort = time_taken[indices_sort]

# Nfuncs = 20

# Prel_DL = np.zeros(len(negloglike_sort))+np.inf
# negloglike_list = []                    # Store all unique negloglikes
# for i in range(len(negloglike_sort)):
#     if negloglike_sort[i] in negloglike_list:       # Never happens for 0th fcn bc negloglike would have to be nan
#         continue                                        # Prel_DL stays at inf for this duplicate function, so Prel -> 0
#     negloglike_list += [negloglike_sort[i]]
#     Prel_DL[i] = DL_sort[i] - DL_sort[0]                # Always gives 0 for the 0th function, so this gets the highest Prel

# Prel = np.exp(-Prel_DL)             # Don't want to use every fcn here bc they could be inf or nan, but the best 1000 should be fine
# Prel[~np.isfinite(Prel) | np.isnan(Prel)] = 0.0
# Prel /= np.sum(Prel)                # Relative probability of fcn, normalised over the top 1000 functions just of this complexity

# ptab = PrettyTable()
# names = ["Rank", "Function", "L(D)", "Prel", "-logL", "Codelen", "AIFeyn"] + [f"a{i}" for i in range(params.shape[1])]
# #Time and other things
# names += ["Nconv", "Niter", "Time"]
# ptab.field_names = names

# for i in range(len(DL_sort)):
    
#     # Only happens for non-duplicates; all Prels should be non-zero
#     if i < Nfuncs:
#         # Combine all data into a single list
#         row_data = [i+1, fcn_min_sort[i], '%.2f'%DL_sort[i], '%.2e'%Prel[i], '%.2f'%negloglike_sort[i], '%.2f'%codelen_sort[i], '%.2e'%aifeyn_sort[i]]
#         row_data += ['%.2e'%params_sort[i,j] for j in range(params.shape[1])]
#         row_data += ['%.2f'%Nconv_sort[i], '%.2f'%Niter_sort[i], '%.2f'%time_sort[i]]

#         # Add the row to the table
#         ptab.add_row(row_data)

# print(ptab)


sys.exit(0)


for i in len(fcn_list):
    if Nconv[i] < 15: #Cambiar esto a los numeros de verdad
        run_name = 'WL'
        data_file = 'esr/dark_matter_data2.txt'
        method = "Nelder-Mead"
        log_opt = False

        Niter = 500
        Nconv = 10

        start = time.time()
        negloglike, DL, params = run_fit_single(data_file, run_name, fcn_list[i], log_opt, method,  Niter, Nconv)
        end = time.time()

        #things
        # s = generator.labels_to_shape(labels, basis_functions)
        # success, _, tree = generator.check_tree(s)
        # fstr = generator.node_to_string(0, tree, labels)
        # max_param = simplifier.get_max_param([fstr], verbose=verbose)
        # fstr, fsym = simplifier.initial_sympify(
        #     [fstr], max_param, parallel=False, verbose=verbose)
        # fstr = fstr[0]
        # fsym = fsym[fstr]
        # print(fstr)

        # #get best fit params
        # chi2, params, algo = run_fit_single(data_file, run_name, fcn_list[i], log_opt, method)

        # #get description length
        # negloglike, codelen, params = calculate_codelen(likelihood, fcn_list[i], tmax=5, try_integration=False, verbose=False, return_params=False, max_param=4)
        # aifeyn = get_fucntional_complexity(labels, verbose=False, max_param=4)
        # negloglike, DL, params = get_description_length(negloglike, codelen, aifeyn, verbose=False)



