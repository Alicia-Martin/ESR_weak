import numpy as np
import math
from scipy.optimize import minimize
import sympy
from mpi4py import MPI
import warnings
import os
import sys
import itertools
# import numdifftools as nd
from scipy.stats import mode
import jax.numpy as jnp

import esr.fitting.test_all as test_all
from esr.fitting.sympy_symbols import *
import esr.generation.simplifier as simplifier

import numdifftools as nd


import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

use_relative_dx = True              # CHANGE

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
    
def load_loglike(comp, likelihood, data_start, data_end, split=True):
    """Load results of optimisation completed by test_all.py
    
    Args:
        :comp (int): complexity of functions to consider
        :likelihood (fitting.likelihood object): object containing data, likelihood functions and file paths
        :data_start (int): minimum index of results we want to load (only if split=True)
        :data_end (int): maximum index of results we want to load (only if split=True)
        :split (bool, deault=True): whether to return subset of results given by data_start and data_end (True) or all data (False)
        
    Returns:
        :negloglike (list): list of minimum log-likelihoods
        :params (np.ndarray): list of parameters at maximum likelihood points. Shape = (nfun, nparam).

    """
    if rank == 0:
        print(likelihood.out_dir + "/negloglike_comp"+str(comp)+".dat")
    data = np.genfromtxt(likelihood.out_dir + "/negloglike_comp"+str(comp)+".dat")
    negloglike = np.atleast_1d(data[:,0])
    params = np.atleast_2d(data[:,1:-3])
    Nconv = np.atleast_1d(data[:,-3])
    Niter = np.atleast_1d(data[:,-2])
    times = np.atleast_1d(data[:,-1])

    if split:
        negloglike = negloglike[data_start:data_end]               # Assuming same order of fcn and chi2 files
        params = params[data_start:data_end,:]
        Nconv = Nconv[data_start:data_end]
        Niter = Niter[data_start:data_end]
        times = times[data_start:data_end]
    return negloglike, params, Nconv, Niter, times


def convert_params(fcn_i, eq,theta_ML, likelihood, negloglike, max_param=4):
    """Compute Fisher, correct MLP and find parametric contirbution to description length for single function
    
    Args:
        :fcn_i (str): string representing function we wish to fit to data
        :eq (sympy object): sympy object for the function we wish to fit to data
        :integrated (bool): whether eq_numpy has already been integrated
        :theta_ML (list): the maximum likelihood values of the parameters
        :likelihood (fitting.likelihood object): object containing data, likelihood functions and file paths
        :negloglike (float): the minimum log-likelihood for this function
        :max_param (int, default=4): The maximum number of parameters considered. This sets the shapes of arrays used.
    
    Returns:
        :params (list): the corrected maximum likelihood values of the parameters
        :negloglike (float): the corrected minimum log-likelihood for this function
        :deriv (list): flattened version of the Hessian of -log(likelihood) at the maximum likelihood point
        :codelen (float): the parameteric contribution to the description length of this function
        
    """

    #max params is 6 if physicalize, 4 otherwise
    # print(max_param)
    nparam_fun = simplifier.count_params([fcn_i], max_param)[0]
    nparam_total = nparam_fun + 2 if likelihood.physicalize else nparam_fun

    # Data
    xvar = likelihood.xvar
    yvar = likelihood.yvar
    yerr = likelihood.yerr

    # print(fcn_i, nparam, flush=True)

    try:
        if nparam_fun == 0:
            if likelihood.physicalize:
                rho0, rs = sympy.symbols("rho0 rs", real=True)
                eq_numpy = sympy.lambdify([x, rho0, rs], eq, modules=["jax"])
            else:
                eq_numpy = sympy.lambdify([x], eq, modules=["jax"])
        elif nparam_fun > 1:
            all_a = ' '.join([f'a{i}' for i in range(nparam_fun)])
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
    except Exception:
        # print("BAD:", fcn_i, negloglike, np.isfinite(negloglike))
        Fisher_diag = np.nan
        deriv[:] = np.nan
        return params, negloglike, deriv, codelen

    #likelihood
    loss_template = likelihood.get_loss(eq_numpy, value = 'evaluate')
    chi2_fcn =likelihood.get_wrapped_like(loss_template)

    if nparam_total > 0:
        def fop(x):
            return chi2_fcn(jnp.array(x), xvar, yvar, yerr, check_nans =False)
    else:
        def fop(x):
            return chi2_fcn([x], xvar, yvar, yerr)

    params = np.zeros(max_param)
    deriv = np.full(int(max_param * (max_param + 1) / 2), np.nan)

    if nparam_total == 0:
        codelen = 0
        # print(fcn_i, 'no params', flush=True)
        return params, negloglike, deriv, codelen

    # try:
    #     if nparam > 1:
    #         all_a = ' '.join([f'a{i}' for i in range(nparam)])
    #         all_a = list(sympy.symbols(all_a, real=True))
    #         eq_numpy = sympy.lambdify([x] + all_a, eq, modules=["numpy"])
    #     else:
    #         eq_numpy = sympy.lambdify([x, a0], eq, modules=["numpy"])
    # except Exception:
    #     # print("BAD:", fcn_i, negloglike, np.isfinite(negloglike))
    #     Fisher_diag = np.nan
    #     deriv[:] = np.nan
    #     return params, negloglike, deriv, codelen
    
    # print(fcn_i)
    
    # Get Hessian
    def get_deriv(Hmat, nparam_fun, max_fun_param, max_param=4):

        Hmat_max = np.zeros((max_param,max_param))
        Hmat_max[:nparam_fun, :nparam_fun] = Hmat[:nparam_fun, :nparam_fun]


        Hmat_max[max_fun_param:, :nparam_fun] = Hmat[nparam_fun:, :nparam_fun]
        Hmat_max[:nparam_fun, max_fun_param:] = Hmat[:nparam_fun, nparam_fun:]
        Hmat_max[max_fun_param:,max_fun_param:] = Hmat[nparam_fun:, nparam_fun:]


        deriv = Hmat_max[np.triu_indices(max_param)]
        return deriv
    
    #CHANGE THIS
    # theta_ML = theta_ML[:nparam]
    max_fun_param = max_param + (nparam_fun - nparam_total)
    theta_ML = np.append(theta_ML[:nparam_fun], theta_ML[max_fun_param:])
    hessian_template = likelihood.get_loss(eq_numpy, value = 'hessian')
    Hmat = hessian_template(theta_ML, likelihood.xvar, likelihood.yvar, likelihood.yerr)
    
    #Other related quantities
    Fisher_diag = jnp.diag(Hmat)
    # Fisher_diag = jnp.array([jnp.nan]*len(Fisher_diag))
    # print('Fisher_diag aqui', fcn_i, Fisher_diag)
    Delta = np.sqrt(12./Fisher_diag)
    # Delta = theta_ML
    deriv = get_deriv(Hmat, nparam_fun, max_fun_param,  max_param=max_param)
    Nsteps = abs(np.array(theta_ML))/Delta

    # Hfun2 = nd.Hessian(fop, step=np.array([1e-1, 1e-20], dtype=np.float128), method='central')
    # Hmat2 = Hfun2(theta_ML)
    # Fisher_diag2 = jnp.diag(Hmat2)
    # Delta2 = np.sqrt(12./Fisher_diag2)
    # print('Delta2', Delta2)

    # print(fcn_i, Delta)


    loss_template = likelihood.get_loss(eq_numpy, value = 'evaluate')
    chi2_fcn = likelihood.get_wrapped_like(loss_template)


    # #Plot likelihood
    # if fcn_i == 'a0*pow(x,a1)':
    # for param_idx in range(len(theta_ML)):
    #     # Delta_plot = theta_ML[param_idx]*10
    #     Delta_plot = 2



    #     x_range = np.linspace(theta_ML[param_idx] - Delta_plot, theta_ML[param_idx] + Delta_plot, 10**3)
    #     # x_range = np.append(x_range, theta_ML[param_idx])
    #     loss_template = likelihood.get_loss(eq_numpy, value = 'evaluate')
    #     chi2_fcn = likelihood.get_wrapped_like(loss_template)

    #     grad_template = likelihood.get_loss(eq_numpy, value = 'grad')
    #     grad_fcn = likelihood.get_wrapped_like(grad_template)

    #     nll = []
    #     # print(x_range)
    #     # grads = []

    #     for i in x_range:
    #         params = np.copy(theta_ML)
    #         params[param_idx] = i

    #         negloglike = chi2_fcn(params, xvar, yvar, yerr)
    #         nll = np.append(nll, negloglike)

    #         # grad = grad_fcn(np.array([i]), xvar, yvar, yerr)
    #         # grads = np.append(grads, grad)

            
    #         # print(jnp.min(nll))
    #     # print(nll)
    #     plt.plot(x_range, np.exp(-nll + jnp.min(nll)))
    #     # plt.plot(theta_ML, np.exp(-chi2_fcn(theta_ML, xvar, yvar, yerr) + jnp.min(nll)), 'ro')
    #     plt.show()
    # sys.exit()

    # plt.plot(x_range, grads)
    # plt.plot(theta_ML,grad_fcn(theta_ML, xvar, yvar, yerr) , 'ro')
    # plt.show()
    
    # 2nd derivatives of -log(L) wrt params
    # Fisher_diag = np.array([Hmat[i,i] for i in range(nparam)])
    
    # Precision to known constants
    # Delta = np.sqrt(12./Fisher_diag)
    # Nsteps = abs(np.array(theta_ML))/Delta


    # Must indicate a bad fcn, so just need to make sure it doesn't have a good -log(L)
    if (np.sum(Fisher_diag <= 0.) > 0.) or (np.sum(np.isnan(Fisher_diag)) > 0) or (np.sum(np.isinf(Fisher_diag)) > 0):
        print("BAD:", fcn_i, negloglike, Fisher_diag)
        # print(deriv)
        # print('here')
        codelen = np.nan
        # print(Delta)
        return params, negloglike, deriv, codelen
    
    k = nparam_total
    # print(fcn_i, k)
    theta_ML_orig = np.copy(theta_ML)
    negloglike_orig = np.copy(negloglike)

    # See whether we can snap any parameters to zero
    # print('Nsteps', Nsteps)
    if np.sum(Nsteps<1)>0:
        # First try setting any parameter to 0 that doesn't have at least
        # one precision step, and recompute -log(L).
        theta_ML[Nsteps<1] = 0.
        negloglike = fop(theta_ML)

        # For the codelen, we effectively don't have the parameter that had Nsteps<1
        if np.isfinite(negloglike):
            k -= np.sum(Nsteps<1)
            kept_mask = Nsteps>=1
        else:
            # Let's see if setting any of the parameters to zero is ok
            try_idx = np.arange(nparam_total)[Nsteps < 1]
            for r in reversed(range(1, len(try_idx))):
                for idx in itertools.combinations(try_idx, r):
                    theta_ML = np.copy(theta_ML_orig)
                    for idx_ in idx:
                        theta_ML[idx_] = 0.
                    negloglike = fop(theta_ML)
                    if np.isfinite(negloglike):
                        break
            kept_mask = np.ones(len(theta_ML), dtype=bool)
            if np.isfinite(negloglike):
                k -= len(idx)
                kept_mask[idx] = 0
            else:
                theta_ML = theta_ML_orig
                negloglike = negloglike_orig
                k = nparam_total
            
        if k<0:
            print("This shouldn't have happened", flush=True)
            quit()
        elif k==0:
            codelen = 0
            # print('I am here', flush=True)
            # print('negloglike', negloglike)
            return params, negloglike, deriv, codelen
        
        Fisher_diag = Fisher_diag[kept_mask]     # Only consider these parameters in the codelen
        theta_ML = theta_ML[kept_mask]
    else:
        kept_mask = np.ones(len(theta_ML), dtype=bool)

    codelen = -k/2.*math.log(3.) + np.sum( 0.5*np.log(Fisher_diag) + np.log(abs(np.array(theta_ML))) )

    # New params after the setting to 0, padded to length max_param as always
    theta_ML = theta_ML_orig
    theta_ML[~kept_mask] = 0.

    # Check if the function has any cutoffs in the likelihood
    Delta[~kept_mask] = 0.
    cutoff_Delta = np.copy(Delta)
    cutoff_Delta[Nsteps < 1] = theta_ML[Nsteps < 1]
    if np.isinf(fop(theta_ML + Delta)) or np.isinf(fop(theta_ML - Delta)):
            codelen = np.nan
            deriv = np.nan*np.ones(deriv.shape)
            print(fcn_i, 'cutoffs', flush=True)
            return params, negloglike, deriv, codelen

    #Save params
    params = np.zeros(max_param)
    params[:nparam_fun] = theta_ML[:nparam_fun]
    params[max_fun_param:] = theta_ML[nparam_fun:]
    # params[:] = np.pad(theta_ML, (0, max_param-len(theta_ML)))

    # print('params', params)

    # print('codelen', codelen)
    # print(fcn_i)

    return params, negloglike, deriv, codelen

    
def main(comp, likelihood, tmax=5, print_frequency=50, try_integration=False):
    """Compute Fisher, correct MLP and find parametric contirbution to description length for all functions and save to file
    
    Args:
        :comp (int): complexity of functions to consider
        :likelihood (fitting.likelihood object): object containing data, likelihood functions and file paths
        :tmax (float, default=5.): maximum time in seconds to run any one part of simplification procedure for a given function
        :print_frequency (int, default=50): the status of the fits will be printed every ``print_frequency`` number of iterations
        :try_integration (bool, default=False): when likelihood requires integral, whether to try to analytically integrate (True) or just numerically integrate (False)
        
    Returns:
        None
    
    """
    
    if likelihood.is_mse:
        raise ValueError('Cannot use MSE with description length')
        
    if rank == 0:
        print('\nComputing Fisher', flush=True)

    if comp>=8:
        sys.setrecursionlimit(2000 + 500 * (comp - 8))

    fcn_list_proc, data_start, data_end = test_all.get_functions(comp, likelihood)
    negloglike, params_proc, Nconv_proc, Niter_proc, times_proc = load_loglike(comp, likelihood, data_start, data_end)
    max_param = params_proc.shape[1]

    codelen = np.zeros(len(fcn_list_proc))          # This is now only for this proc
    params = np.zeros([len(fcn_list_proc), max_param])
    deriv = np.zeros([len(fcn_list_proc), int(max_param * (max_param+1) / 2)])

    for i in range(len(fcn_list_proc)):           # Consider all possible complexities
        if rank == 0 and ((i == 0) or ((i+1) % print_frequency == 0)):
            print(f'{i+1} of {len(fcn_list_proc)}', flush=True)

        if np.isnan(negloglike[i]) or np.isinf(negloglike[i]):
            codelen[i]=np.nan
            continue

        theta_ML = params_proc[i,:]

        # print(fcn_list_proc[i])
        try:
            fcn_i = fcn_list_proc[i].replace('\n', '')
            fcn_i = fcn_list_proc[i].replace('\'', '')
            fcn_i, eq = likelihood.run_sympify(fcn_i, tmax=tmax, try_integration=try_integration)
            params[i,:], negloglike[i], deriv[i,:], codelen[i] = convert_params(fcn_i, eq, theta_ML, likelihood, negloglike[i], max_param=max_param)
        except NameError:
            # Occurs if function produced not implemented in numpy
            if try_integration:
                fcn_i = fcn_list_proc[i].replace('\n', '')
                fcn_i = fcn_list_proc[i].replace('\'', '')
                fcn_i, eq, integrated = likelihood.run_sympify(fcn_i, tmax=tmax, try_integration=False)
                params[i,:], negloglike[i], deriv[i,:], codelen[i] = convert_params(fcn_i, eq, integrated, theta_ML, likelihood, negloglike[i], max_param=max_param)
            else:
                params[i,:] = 0.
                deriv[i,:] = 0.
                codelen[i] = 0

        except:
            params[i,:] = 0.
            deriv[i,:] = 0.
            codelen[i] = 0

        # print(fcn_i, negloglike[i], codelen[i], flush=True)
        
    # out_arr = np.transpose(np.vstack([codelen, negloglike] + [params[:,i] for i in range(max_param)]))

    out_arr = np.vstack([codelen, negloglike] + [params[:, i] for i in range(max_param)] + [Nconv_proc, Niter_proc, times_proc])
    out_arr = np.transpose(out_arr)

    out_arr_deriv = np.transpose(np.vstack([deriv[:,0], deriv[:,1], deriv[:,2], deriv[:,3], deriv[:,4], deriv[:,5], deriv[:,6], deriv[:,7], deriv[:,8], deriv[:,9]]))
    out_arr_deriv = np.transpose(np.vstack([deriv[:,i] for i in range(deriv.shape[1])]))

    np.savetxt(likelihood.temp_dir + '/codelen_deriv_'+str(comp)+'_'+str(rank)+'.dat', out_arr, fmt='%.7e')
    np.savetxt(likelihood.temp_dir + '/derivs_'+str(comp)+'_'+str(rank)+'.dat', out_arr_deriv, fmt='%.7e')

    comm.Barrier()

    if rank == 0:
        string = 'cat `find ' + likelihood.temp_dir + '/ -name "codelen_deriv_'+str(comp)+'_*.dat" | sort -V` > ' + likelihood.out_dir + '/codelen_comp'+str(comp)+'_deriv.dat'
        os.system(string)
        string = 'rm ' + likelihood.temp_dir + '/codelen_deriv_'+str(comp)+'_*.dat'
        os.system(string)

        string = 'cat `find ' + likelihood.temp_dir + '/ -name "derivs_'+str(comp)+'_*.dat" | sort -V` > ' + likelihood.out_dir + '/derivs_comp'+str(comp)+'.dat'
        os.system(string)
        string = 'rm ' + likelihood.temp_dir + '/derivs_'+str(comp)+'_*.dat'
        os.system(string)
        
    comm.Barrier()

    return
