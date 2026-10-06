import sys
import matplotlib.pyplot as plt
import csv
from mpi4py import MPI
import warnings
import numpy as np
import sympy
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl
import os

from esr.fitting.sympy_symbols import *
import esr.generation.simplifier as simplifier
from esr.esd import ExcessSurfaceDensity

warnings.filterwarnings("ignore")

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def main(comp, likelihood, tmax=5, try_integration=False, xscale='linear', yscale='linear'):
    """Plot best 50 functions from all complexity levels against data and save plot to file
    
    Args:
        :comp (int): maximum complexity of functions to consider (will use all complexities up to this value)
        :likelihood (fitting.likelihood object): object containing data, functions to convert SR expressions to variable of data and output path
        :tmax (float, default=5.): maximum time in seconds to run any one part of simplification procedure for a given function
        :try_integration (bool, default=False): when likelihood requires integral, whether to try to analytically integrate (True) or just numerically integrate (False)
        :xscale (str), default='linear'): Scaling for x-axis
        :yscale (str), default='linear'): Scaling for y-axis
        
    Returns:
        None
    
    """
    if rank != 0:
        return
    
    print('\nMaking plots', flush=True)

    vmin = 1e-3
    vmax = 1
    tmax = 5

    if comp>=8:
        sys.setrecursionlimit(2000 + 500 * (comp - 8))

    if not os.path.isdir(likelihood.fig_dir):
        print('Making:', likelihood.fig_dir)
        os.mkdir(likelihood.fig_dir)

    # Collect data from all complexity levels
    all_data = []
    max_param = 0
    complexities_found = []
    
    for c in range(1, comp + 1):
        file_path = likelihood.out_dir + '/final_'+str(c)+'.dat'
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                reader = csv.reader(f, delimiter=';')
                comp_data = [row for row in reader]
                if len(comp_data) > 0:
                    all_data.extend(comp_data)
                    complexities_found.append(c)
                    # Track maximum number of parameters across all complexities
                    max_param = max(max_param, len(comp_data[0]) - 7)
                    print(f"Found {len(comp_data)} functions at complexity {c}")
        else:
            print(f"File {file_path} not found, skipping complexity {c}")
    
    print(f"Mixing results from complexities: {complexities_found}")
    print(f"Total functions found: {len(all_data)}")
    
    if len(all_data) == 0:
        print("No functions with finite DL found across all complexities, so will not make figure")
        return
        
    fcn_list = [d[1] for d in all_data]
    # Pad parameter arrays with zeros for functions with fewer parameters
    params = []
    for d in all_data:
        param_row = np.array(d[-max_param:], dtype=float) if len(d) >= 7 + max_param else np.zeros(max_param)
        if len(d) < 7 + max_param:
            actual_params = np.array(d[7:], dtype=float) if len(d) > 7 else np.array([])
            param_row[:len(actual_params)] = actual_params
        params.append(param_row)
    params = np.array(params)
    
    DL = np.array([d[2] for d in all_data], dtype=float)
    if not np.any(np.isfinite(DL)):
        print('All DL are infinite, so skipping plot')
        return
    DL_min = np.amin(DL[np.isfinite(DL)])
    alpha = DL_min - DL
    alpha = np.exp(alpha)
    m = (alpha > vmin)
    
    # Sort by DL (best functions first) before filtering
    finite_mask = np.isfinite(DL)
    if np.sum(finite_mask) == 0:
        print('No finite DL values found, so skipping plot')
        return
    
    # Sort indices by DL (ascending - best first)
    sort_indices = np.argsort(DL[finite_mask])
    finite_indices = np.where(finite_mask)[0][sort_indices]
    
    # Apply filtering and sorting
    filtered_indices = finite_indices[alpha[finite_indices] > vmin]
    
    fcn_list = [fcn_list[i] for i in filtered_indices]
    params = params[filtered_indices,:]
    alpha = alpha[filtered_indices]

    fig  = plt.figure(figsize=(7,5))
    ax1  = fig.add_axes([0.10,0.10,0.70,0.85])
    cmap = cm.hot_r
    norm = mpl.colors.LogNorm(vmin=vmin,vmax=vmax)

    fig2 = plt.figure(figsize=(7,5))
    ax2  = fig2.add_axes([0.10,0.10,0.70,0.85])

    for i in range(min(len(fcn_list),10)):

        fcn_i = fcn_list[i].replace('\'', '')
        
        k = simplifier.count_params([fcn_i], max_param)[0]
        measured = params[i,:k]

        print('%i of %i:'%(i+1,len(fcn_list)), fcn_i)
        
        # try:
        fcn_i, eq= likelihood.run_sympify(fcn_i, tmax=tmax, try_integration=try_integration)
        
        if k == 0:
            eq_numpy = sympy.lambdify([x], eq, modules=["numpy"])
        elif k > 1:
            all_a = ' '.join([f'a{i}' for i in range(k)])
            all_a = list(sympy.symbols(all_a, real=True))
            eq_numpy = sympy.lambdify([x] + all_a, eq, modules=["numpy"])
        else:
            eq_numpy = sympy.lambdify([x, a0], eq, modules=["numpy"])
        # ypred = likelihood.get_pred(measured, likelihood.xvar, eq_numpy)
        ypred = eq_numpy(likelihood.xvar, *measured)

        #Plot the ESD on the first figure        
        x_array = np.linspace(likelihood.xvar.min(), likelihood.xvar.max(), 1000)
        esd = ExcessSurfaceDensity.calculate(x_array, eq_numpy, params=measured)
        if np.isscalar(ypred):
            ax1.plot(x_array, [esd]*len(likelihood.xvar), color=cmap(norm(alpha[i])), zorder=len(fcn_list)-i)
        else:
            ax1.plot(x_array, esd, color=cmap(norm(alpha[i])), zorder=len(fcn_list)-i)

        #Plot density
        if np.isscalar(ypred):
            ax2.plot(x_array, [ypred]*len(x_array), color=cmap(norm(alpha[i])), zorder=len(fcn_list)-i, label = fcn_i)
        else:
            ax2.plot(x_array, ypred, color=cmap(norm(alpha[i])), zorder=len(fcn_list)-i, label = fcn_i)
        
    if hasattr(likelihood, 'yerr'):
        ax1.errorbar(likelihood.xvar, likelihood.yvar, yerr=likelihood.yerr, fmt='.', markersize=5, zorder=len(fcn_list)+1, capsize=1, elinewidth=1, color='k', alpha=1)
    else:
        ax1.plot(likelihood.xvar, likelihood.yvar, '.', color='k', ms=5, zorder=len(fcn_list)+1, alpha=1)
    ax1.set_xlabel(r'$r_{proj} (Mpc)$')
    ax1.set_ylabel(r'$ESD (10^{12} M_{sun}/Mpc^2)$')
    ax1.set_xscale(xscale)
    ax1.set_yscale(yscale)
    if xscale != 'log':
        ax1.set_xlim(0, None)
    ax1.set_ylim(likelihood.yvar.min() * 0.9, likelihood.yvar.max() * 1.1)

    ax2  = fig.add_axes([0.85,0.10,0.05,0.85])
    cb1  = mpl.colorbar.ColorbarBase(ax2,cmap=cmap,norm=norm,orientation='vertical')
    cb1.set_label(r'$\exp \left( MDL - DL \right)$')
    fig.tight_layout()
    fig.savefig(likelihood.fig_dir + '/plot_all_comps_up_to_%i.png'%comp, dpi=300)
    fig.clf()
    # plt.show()
    plt.close(fig)

    fig2.tight_layout()
    fig2.savefig(likelihood.fig_dir + '/density_plot_all_comps_up_to_%i.png'%comp, dpi=300)
    fig2.clf()
    plt.close(fig2)

    return