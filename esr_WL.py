import numpy as np
import sys
from mpi4py import MPI
from sympy import *
import sympy
import time
import matplotlib.pyplot as plt

import esr.fitting.test_all
import esr.fitting.test_all_Fisher
import esr.fitting.match
import esr.fitting.combine_DL
import esr.fitting.plot
from esr.fitting.likelihood import Likelihood
from esr.generation.simplifier import time_limit
from esr.fitting.sympy_symbols import *
import esr.plotting.plot
import esr.generation.simplifier as simplifier
from esr.fitting.fit_single import fit_from_string
from esr.fitting.WL_likelihood import WLLikelihood
import esr.fitting.plot_Hull
import esr.fitting.compare_methods
from esr.esd import ExcessSurfaceDensity
import esr.fitting.combine_comp
import esr.fitting.combine_galaxies


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def plot_single(fcn, measured, likelihood, ax1, max_param=4, tmax=5, try_integration=False, xscale='linear', yscale='linear'):
        fcn_i = fcn.replace('\'', '')

        # fig  = plt.figure(figsize=(7,5))
        # ax1  = fig.add_axes([0.10,0.10,0.70,0.85])

        
        k = simplifier.count_params([fcn_i], max_param)[0]


        fcn_i, eq= likelihood.run_sympify(fcn_i, tmax=tmax, try_integration=try_integration)
        
        # if k == 0:
        #     eq_numpy = sympy.lambdify([x], eq, modules=["numpy"])
        # elif k > 1:
        #     all_a = ' '.join([f'a{i}' for i in range(k)])
        #     all_a = list(sympy.symbols(all_a, real=True))
        #     eq_numpy = sympy.lambdify([x] + all_a, eq, modules=["numpy"])
        # else:
        #     eq_numpy = sympy.lambdify([x, a0], eq, modules=["numpy"])

        if likelihood.physicalize:
            n_fun_params = k
            nparams  = n_fun_params + 2
            n_extra = 2
        else:
            n_fun_params = k
            n_extra = 0
        # print(eq)
        # sys.exit()
        print('nparams', n_extra)
        if n_fun_params == 0 and n_extra == 0:
            eq_numpy = sympy.lambdify([x], eq, modules=["jax"])
        elif n_fun_params == 0 and n_extra > 0:
                rho0, rs = sympy.symbols("rho0 rs", real=True)
                eq_numpy = sympy.lambdify([x, rho0, rs], eq, modules=["jax"])
        elif n_fun_params > 1:
            all_a = ' '.join([f'a{i}' for i in range(n_fun_params)])
            all_a = list(sympy.symbols(all_a, real=True))
            if n_extra>0:
                print('HERE')
                all_a += sympy.symbols("rho0 rs", real=True)
            eq_numpy = sympy.lambdify([x] + all_a, eq, modules=["jax"])
        elif n_fun_params == 1:
            if n_extra>0:
                # print('HERE')
                rho0, rs = sympy.symbols("rho0 rs", real=True)
                eq_numpy = sympy.lambdify([x, a0, rho0, rs], eq, modules=["jax"])
            else:
                eq_numpy = sympy.lambdify([x, a0], eq, modules=["jax"])
        ypred = eq_numpy(likelihood.xvar, *measured)

    
        # esd = ExcessSurfaceDensity.calculate(likelihood.xvar, eq_numpy, params=measured)
        x_array = np.linspace(0, likelihood.xvar.max(), 1000)
        esd = ExcessSurfaceDensity.calculate(x_array, eq_numpy, params=measured)
        y_pred2 = eq_numpy(x_array, *measured)

        # Plot the ESD on the first figure
        ax1.plot(x_array, esd, label=fcn_i)
        ax1.errorbar(likelihood.xvar, likelihood.yvar, yerr=likelihood.yerr, fmt='.')
        ax1.set_xscale(xscale)
        ax1.set_yscale(yscale)
        plt.show()

        #Plot density
        x_array = np.linspace(0,60, 1000)
        y_pred = eq_numpy(x_array, *measured)
        fig2 = plt.figure(figsize=(7,5))
        ax2  = fig2.add_axes([0.10,0.10,0.70,0.85])

        ax2.plot(x_array, y_pred)

        # if np.isscalar(y_pred2):
        #     ax2.plot(x_array, [y_pred2]*len(x_array))
        # else:
        #     ax2.plot(x_array, y_pred2)
        ax2.set_xscale(xscale)
        ax2.set_yscale(yscale)
        # plt.show()

def run_fit_single(data_file, run_name, fn, log_opt, method, ax):
        basis_functions = [["x", "a"],  # type0
                ["inv", "abs", "log", "exp", "re", "im"],  # type1
                ["+", "*", "-", "/", "pow"]]  # type2
        
        # print(run_name)

        #Likelihood
        likelihood = WLLikelihood(data_file, run_name, data_dir=None, fn_set = 'core_maths', physicalize=False) 

        print('fn:', fn)

        chi2, DL, labels, params, delta = fit_from_string(fn,
                                                        basis_functions,
                                                        likelihood,
                                                        method = method,
                                                        try_integration = False,
                                                        verbose=True,
                                                        log_opt=log_opt,
                                                        return_params=True)
        
        print('params', params)
        plot_single(fn, params, likelihood, ax)

        #calculate the mass and concentration
        # fcn_i, eq= likelihood.run_sympify(fn)
        # k = simplifier.count_params([fcn_i], 4)[0]
        # if k == 0:
        #     eq_numpy = sympy.lambdify([x], eq, modules=["jax"])
        # elif k > 1:
        #     all_a = ' '.join([f'a{i}' for i in range(k)])
        #     all_a = list(sympy.symbols(all_a, real=True))
        #     eq_numpy = sympy.lambdify([x] + all_a, eq, modules=["jax"])
        # else:
        #     eq_numpy = sympy.lambdify([x, a0], eq, modules=["jax"])

        # M200, r200, error_M200 = likelihood.M_delta(200, eq_numpy, params, delta)

        return chi2, params, DL


def fit_galaxy(data_file, run_name, comp, try_integration=False, method="Nelder-Mead", log_opt=True):
    if log_opt:
        log = '_log'
    else:
        log = ''

    #Likelihood
    likelihood = WLLikelihood(data_file, run_name + log, data_dir=None, fn_set = 'core_maths', physicalize=False)   
    
    # run esr
    esr.fitting.test_all.main(comp, likelihood, try_integration=try_integration, log_opt=log_opt, method=method, ignore_previous_eqns=False)
    # esr.fitting.test_all_Fisher.main(comp, likelihood, tmax=5, try_integration=try_integration)
    # esr.fitting.match.main(comp, likelihood, tmax=5, try_integration=try_integration)
    # esr.fitting.combine_DL.main(comp, likelihood)
    # esr.fitting.plot.main(comp, likelihood, tmax=5, try_integration=try_integration, xscale='log', yscale='log')

     
    
###########################################################################
                                 # MAIN #
###########################################################################
    
#------------------------------------------------------------
# run code for a single set of data
#------------------------------------------------------------

# comp = 4
# try_integration = False
# method = "Nelder-Mead"
# log_opt = True
# # data_file = 'esr/dark_matter_data2.txt'
# name = 148
# data_file = 'XXL/' + str(name) + '.txt'
# run_name = 'WL_' + str(name) + '_katz'

# if rank == 0:
#     print('method:', method, ', log_opt:', log_opt, "cluster:", name, flush=True)

# start = time.time()
# fit_galaxy(data_file, run_name, comp, try_integration=try_integration, method=method, log_opt=log_opt)
# end = time.time()

# if rank == 0:
#     print('Total time taken:', end - start, flush=True)


# ------------------------------------------------------------
# run code for one fucntion
# ------------------------------------------------------------

method = "Nelder-Mead"
log_opt = False
# name = 76
# name = 178
# name = 171
# name = 127
name = 111
# name = 161
data_file = 'XXL/' + str(name) + '.txt'
run_name = 'WL_' + str(name)

# dtafile all clusters
# data_file = 'XXL/combined_data.txt'
# run_name = 'WL'

fn1 =  'pow(Abs(a0*x - pow(x,x)),a1)'
fn2 =  'a1/(-x + pow(Abs(a0),x))'
fn3 = '(a0 + a1/x)/x'
fn4 = 'a0*(x + 1/x)/x'
fn5 = 'a0*(a1*x + 1/x)'

fn_list = [fn1, fn2, fn3, fn4, fn5]

# fn_list = ['a0/(1 - a1) - x/(1 - a1)']

fn_list = ['pow(Abs(a0),(pow(x,x)))']
# fn_list = ['a0/(a1 + x)']
# fn_list = ['1/(a1 + pow(Abs(a0),x))']
# fn_list = ['(a0*x - x)*pow(Abs(a1),x)']
fn_list = ['x*pow((1/Abs(a0)*x),x)/a1']
fn_list = ['a1*pow(1/(Abs(a0)*x),x) ']
# fn_list = ['pow(x,((x)**(-2)))']
# fn_list = ['x*(x + 1/a1)*pow(Abs(a0),x)']
fn_list = ['x*(a0 + x)*pow(Abs(a1),x)']
fn_list = ['(a1 + x**2)*pow(Abs(a0),x)']
# fn_list = ['pow((x*Abs(a0)),a1) - 1/x']
# fn_list = ['a0*(a1 + x**(-3))']
fn_list = ['x**2/(a0*pow(Abs(a1),x))']
# fn_list = ['a0*pow(Abs(a1),(x**2))'] 
fn_list = ['x/(a0*(-x + pow(Abs(a1),x)))']
fn_list = ['a0*x**(-3)']


if rank == 0:
    print('method:', method, ', log_opt:', log_opt, flush=True)

# Create a figure and axis for the plot
fig, ax = plt.subplots(figsize=(7, 5))

for fn in fn_list:
    start = time.time()
    chi2, params, algo = run_fit_single(data_file, run_name, fn, log_opt, method, ax)
    print(chi2, params)
    end = time.time()
    print('Total time taken:', end - start, flush=True)
plt.legend()
plt.show()

if rank == 0:
    print('Total time taken:', end - start, flush=True)
    
#------------------------------------------------------------
#run pareto plot
#------------------------------------------------------------
    
# from esr.plotting.plot import pareto_plot

# name = 27
# dirname = 'esr/fitting/output_glamdring/output/output_WL_' + str(name)
# savefile = 'pareto_' + str(name) + '.png'
# # pareto_plot(dirname, savefile)


#------------------------------------------------------------
# combine all comp for same cluster
#------------------------------------------------------------

# name = 27
# dirname = 'esr/fitting/output_glamdring/output/output_WL_' + str(name)
# savename = 'combine_all_comp_' + str(name) + '.dat'
# esr.fitting.combine_comp.main(dirname, savename)

#------------------------------------------------------------
# combine all comp for all clusters
#------------------------------------------------------------

# dirname = 'esr/fitting/output_glamdring/output/'
# fcn_dir = 'esr//function_library/core_maths/'
# esr.fitting.combine_galaxies.main(dirname, fcn_dir)


#------------------------------------------------------------
# esr.fitting.compare_methods.create_comparison_table(comp, likelihood)
# esr.fitting.plot_Hull.main(comp, likelihood, tmax=5, try_integration=try_integration, xscale='linear', yscale='linear')
# esr.fitting.combine_galaxies.main(comp, likelihood)