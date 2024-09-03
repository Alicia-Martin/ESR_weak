import numpy as np
import os
import sys
from mpi4py import MPI
import pickle
import scipy
from sympy import *
import sympy
import time
import matplotlib.pyplot as plt
import jax.numpy as jnp

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


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def plot_single(fcn, measured, likelihood, max_param=4, tmax=5, try_integration=False, xscale='linear', yscale='linear'):
        fcn_i = fcn.replace('\'', '')

        fig  = plt.figure(figsize=(7,5))
        ax1  = fig.add_axes([0.10,0.10,0.70,0.85])

        
        k = simplifier.count_params([fcn_i], max_param)[0]


        fcn_i, eq= likelihood.run_sympify(fcn_i, tmax=tmax, try_integration=try_integration)
        
        if k == 0:
            eq_numpy = sympy.lambdify([x], eq, modules=["numpy"])
        elif k > 1:
            all_a = ' '.join([f'a{i}' for i in range(k)])
            all_a = list(sympy.symbols(all_a, real=True))
            eq_numpy = sympy.lambdify([x] + all_a, eq, modules=["numpy"])
        else:
            eq_numpy = sympy.lambdify([x, a0], eq, modules=["numpy"])
        ypred = eq_numpy(likelihood.xvar, *measured)

    
        # esd = ExcessSurfaceDensity.calculate(likelihood.xvar, eq_numpy, params=measured)
        esd = ExcessSurfaceDensity.calculate(likelihood.xvar, eq_numpy, params=measured)


        # measured = [-10**(6), 1]
        # esd = ExcessSurfaceDensity.calculate(likelihood.xvar, eq_numpy, params=measured)
        # print(esd)

        # measured = [1, 1]
        # esd = ExcessSurfaceDensity.calculate(likelihood.xvar, eq_numpy, params=measured)
        # print(esd)


        # measured = [2, 1]
        # esd = ExcessSurfaceDensity.calculate(likelihood.xvar, eq_numpy, params=measured)
        # print(esd)
        # sys.exit(0)

        # Plot the ESD on the first figure
        ax1.plot(likelihood.xvar, esd)
        ax1.errorbar(likelihood.xvar, likelihood.yvar, yerr=likelihood.yerr, fmt='.')
        ax1.set_xscale(xscale)
        ax1.set_yscale(yscale)
        plt.show()

        fig2 = plt.figure(figsize=(7,5))
        ax2  = fig2.add_axes([0.10,0.10,0.70,0.85])
        ax2.plot(likelihood.xvar, ypred)
        ax2.set_xscale(xscale)
        ax2.set_yscale(yscale)
        plt.show()


        # ax1.plot(likelihood.xvar, esd)
        

        # ax1.errorbar(likelihood.xvar, likelihood.yvar, yerr=likelihood.yerr, fmt='.')

        # # ax1.set_xlabel(r'$r_{proj} (Mpc)$')
        # # ax1.set_ylabel(r'$ESD (10^{12} M_{sun}/Mpc^2)$')
        # # ax1.set_xscale(xscale)
        # # ax1.set_yscale(yscale)
        # # if xscale != 'log':
        # #     ax1.set_xlim(0, None)
        # # ax1.set_ylim(likelihood.yvar.min() * 0.9, likelihood.yvar.max() * 1.1)

        # # fig.tight_layout()
        # # fig.clf()
        # plt.show()

        # axfig2.plot(likelihood.xvar, ypred)
        # plt.show()
    # plt.close(fig)

    # fig2.tight_layout()
    # fig2.savefig(likelihood.fig_dir + '/density_plot_%i.png'%comp, dpi=300)
    # fig2.clf()
    # plt.close(fig2)

def run_fit_single(data_file, run_name, fn, log_opt, method):
        basis_functions = [["x", "a"],  # type0
                ["inv", "abs", "log", "exp"],  # type1
                ["+", "*", "-", "/", "pow"]]  # type2
        
        # print(run_name)

        #Likelihood
        likelihood = WLLikelihood(data_file, run_name, data_dir=None, fn_set = 'core_maths') 

        chi2, DL, labels, params = fit_from_string(fn,
                                                        basis_functions,
                                                        likelihood,
                                                        method = method,
                                                        try_integration = False,
                                                        verbose=True,
                                                        log_opt=log_opt,
                                                        return_params=True)
        
        # plot_single(fn, params, likelihood)
        return chi2, params, DL



def fit_galaxy(data_file, run_name, comp, try_integration=False, method="Nelder-Mead", log_opt=True):
    if log_opt:
        log = '_log'
    else:
        log = ''

    #Likelihood
    likelihood = WLLikelihood(data_file, run_name + log, data_dir=None, fn_set = 'core_maths')   
    
    # esr.fitting.compare_methods.create_comparison_table(comp, likelihood)
    # esr.fitting.plot_Hull.main(comp, likelihood, tmax=5, try_integration=try_integration, xscale='linear', yscale='linear')
    # esr.fitting.combine_comp.main(likelihood, tmax=5, try_integration=try_integration, xscale='log', yscale='log')

    
    # run esr
    # esr.fitting.test_all.main(comp, likelihood, try_integration=try_integration, log_opt=log_opt, method=method)
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
# method = "BFGS"
# log_opt = False
# # data_file = 'esr/dark_matter_data2.txt'
# name = 6
# data_file = 'XXL/' + str(name) + '.pickle'
# run_name = 'WL_' + str(name)

# if rank == 0:
#     print('method:', method, ', log_opt:', log_opt, flush=True)

# start = time.time()
# fit_galaxy(data_file, run_name, comp, try_integration=try_integration, method=method, log_opt=log_opt)
# end = time.time()

# if rank == 0:
#     print('Total time taken:', end - start, flush=True)


#------------------------------------------------------------
# run code for one fucntion
#------------------------------------------------------------

method = "BFGS"
log_opt = False
data_file = 'XXL/6.pickle'
run_name = 'WL'

# fn = 'a0/(x*(x + 1)^2)' #NFW

# fn = 'pow(Abs(a0),(pow(x,x)))'
# fn = 'pow(Abs(a0),(-1/x))'
# fn = '1/(x + pow(Abs(a0),x))'
# fn = ' pow(Abs(a0),(pow(Abs(a1),(1/x))))'
# fn = '1/(a0 + pow (Abs (a1) ,x) )'
# fn = 'a0/x**2'
# fn = 'x + 1/(a0 + x)'
# fn = 'pow(0,(1/a0))'
# fn = 'pow(Abs(a0),(-1/x))'
# fn = 'pow(Abs(a0 - x),-1.3600380)'
# fn = 'pow(Abs(a0),(1/(2*x)))'
# fn = '1/(x + pow(Abs(a0),x))'
# fn = '1/(x - Abs(a0))'
# fn = 'pow(Abs(a0),(1/x))/x'
# fn = 'x/pow(Abs(a0),x)'
# fn = 'a0/(a1 - x) + a2'
fn = 'a0'

if rank == 0:
    print('method:', method, ', log_opt:', log_opt, flush=True)

start = time.time()
chi2, params, algo = run_fit_single(data_file, run_name, fn, log_opt, method)
print(chi2, params)
end = time.time()



if rank == 0:
    print('Total time taken:', end - start, flush=True)