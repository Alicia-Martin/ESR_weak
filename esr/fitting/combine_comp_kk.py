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
from tabulate import tabulate

from esr.fitting.sympy_symbols import *
import esr.generation.simplifier as simplifier
from esr.esd import ExcessSurfaceDensity
from prettytable import PrettyTable

warnings.filterwarnings("ignore")

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def main(likelihood, tmax=5, try_integration=False, xscale='linear', yscale='linear'):
    """Plot best 50 functions at given complexity against data and save plot to file
    
    Args:
        :comp (int): complexity of functions to consider
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

    vmin = 1e-3
    vmax = 1
    tmax = 5

    comp_min = 1
    comp_max = 5

    data = []
    for comp in range(comp_min, comp_max+1):
        with open(likelihood.out_dir + '/final_'+str(comp)+'.dat', "r") as f:
            reader = csv.reader(f, delimiter=';')
            data_comp = [row for row in reader]
            for row in data_comp:
                row.append(comp)
            data += data_comp

            #add a column for complexity
    if len(data) == 0:
        print("No functions with finite DL found, so will not make figure")
        return
    
    max_param = len(data[0]) - 7

    fcn_list = [d[1] for d in data]
    DL = np.array([d[2] for d in data], dtype=float)
    Prel = np.array([d[3] for d in data], dtype=float)
    negloglike = np.array([d[4] for d in data], dtype=float)
    codelen = np.array([d[5] for d in data], dtype=float)
    ayfeyn = np.array([d[6] for d in data], dtype=float)
    comp = np.array([d[-1] for d in data], dtype=int)

    params = np.array([d[-max_param:(4 -max_param)] for d in data], dtype=float)
    DL = np.array([d[2] for d in data], dtype=float)
    if not np.any(np.isfinite(DL)):
        print('Add DL are infinite, so skipping plot')
        return

    # DL_min = np.amin(DL[np.isfinite(DL)])
    # alpha = DL_min - DL
    # alpha = np.exp(alpha)
    # m = (alpha > vmin)
    # fcn_list = [d for i, d in enumerate(fcn_list) if m[i]]
    # params = params[m,:]
    # alpha = alpha[m]

    # Table by complexity
        
    headers = ["Function", "DL", "Prel", "negloglike", "codelen", "ayfeyn", "a0", "a1", "a3", "a4", "Nconv", "Niter", "time (s)", "comp"]
    table_data = []


    for i in range(len(fcn_list)):
        # row = [fcn_list[i], DL[i], Prel[i],negloglike[i], codelen[i], ayfeyn[i]] + params[i].tolist() + [data[i][-3], data[i][-2], data[i][-1]] + [comp]
        row = [fcn_list[i], 
       '%.2f' % DL[i], 
       '%.2f' % Prel[i], 
       '%.2f' % negloglike[i], 
       '%.2f' % codelen[i], 
       '%.2f' % ayfeyn[i]] + ['%.2f' % p for p in params[i]] + [data[i][-4], data[i][-3], data[i][-2]] + [comp[i]]

        table_data.append(row)
    
    table_data.sort(key=lambda x: x[1])  # Sort by DL (second column)
    # table = tabulate(table_data, headers=headers, tablefmt="pretty")


    pretty_table = PrettyTable()
    pretty_table.field_names = headers

    for row in table_data:
        pretty_table.add_row(row)
    print(pretty_table)

    # Save the table to a file
        
    output_file = os.path.join(likelihood.out_dir, 'combine_final.txt')
    with open(output_file, "w") as f:
        f.write(pretty_table.get_string())


    # vmin = alpha.min()


    # fig  = plt.figure(figsize=(7,5))
    # ax1  = fig.add_axes([0.10,0.10,0.70,0.85])
    # cmap = cm.hot_r
    # norm = mpl.colors.LogNorm(vmin=vmin,vmax=vmax)

    # fig2 = plt.figure(figsize=(7,5))
    # axfig2  = fig2.add_axes([0.10,0.10,0.70,0.85])

    # fig3 = plt.figure(figsize=(7,5))
    # axfig3  = fig3.add_axes([0.10,0.10,0.70,0.85])

    # for i in range(min(len(fcn_list), 10)):
    #     print('i:', i)
    #     # print(DL[i], alpha[i])

    #     fcn_i = fcn_list[i].replace('\'', '')
        
    #     k = simplifier.count_params([fcn_i], max_param)[0]
    #     measured = params[i,:k]

    #     print('%i of %i:'%(i+1,len(fcn_list)), fcn_i)
        
    #     # try:
    #     fcn_i, eq= likelihood.run_sympify(fcn_i, tmax=tmax, try_integration=try_integration)
        
    #     if k == 0:
    #         eq_numpy = sympy.lambdify([x], eq, modules=["numpy"])
    #     elif k > 1:
    #         all_a = ' '.join([f'a{i}' for i in range(k)])
    #         all_a = list(sympy.symbols(all_a, real=True))
    #         eq_numpy = sympy.lambdify([x] + all_a, eq, modules=["numpy"])
    #     else:
    #         eq_numpy = sympy.lambdify([x, a0], eq, modules=["numpy"])
    #     ypred = likelihood.get_pred(measured, likelihood.xvar, eq_numpy)

    #     x_range = np.linspace(likelihood.xvar.min(), likelihood.xvar.max(), 1000)
    #     density = eq_numpy(x_range, *measured)

    #     if i==len(fcn_list)-1:
    #         print('HERE')
    #         axfig2.plot(x_range, density, color='blue', label='NFW')
    #     else:
    #         axfig2.plot(x_range, density, color=cmap(norm(alpha[i])), zorder=len(fcn_list)-i)
    #     # axfig2.set_xscale('log')
    #     axfig2.set_yscale('log')
    #     # axfig2.set_ylim(10**6, 10**7)
    #     axfig2.set_xlabel(r'$r (Mpc)$')
    #     axfig2.set_ylabel(r'$\rho / 10^{12} M_{sun}/Mpc^3)$')


    #     if i==len(fcn_list)-1:
    #         print('HERE')
    #         ax1.plot(likelihood.xvar, ypred, color='blue', label='NFW')

    #     else:
    #         if np.isscalar(ypred):
    #             ax1.plot(likelihood.xvar, [ypred]*len(likelihood.xvar), color=cmap(norm(alpha[i])), zorder=len(fcn_list)-i)
    #         else:
    #             ax1.plot(likelihood.xvar, ypred, color=cmap(norm(alpha[i])), zorder=len(fcn_list)-i)
    # if hasattr(likelihood, 'yerr'):
    #     ax1.errorbar(likelihood.xvar, likelihood.yvar, yerr=likelihood.yerr, fmt='.', markersize=5, zorder=len(fcn_list)+1, capsize=1, elinewidth=1, color='k', alpha=1)
    # else:
    #     ax1.plot(likelihood.xvar, likelihood.yvar, '.', color='k', ms=5, zorder=len(fcn_list)+1, alpha=1)
    # ax1.set_xlabel(r'$r_{proj} / Mpc$')
    # ax1.set_ylabel(r'$ESD / 10^{12} M_{sun}/Mpc^2$')
    # # ax1.set_xscale(xscale)
    # ax1.set_yscale('log')
    # # ax1.set_xscale(xscale)
    # # ax1.set_yscale(yscale)
    # if xscale != 'log':
    #     ax1.set_xlim(0, None)
    # ax1.set_ylim(likelihood.yvar.min() * 0.9, likelihood.yvar.max() * 1.1)

    # ax2  = fig.add_axes([0.85,0.10,0.05,0.85])
    # cb1  = mpl.colorbar.ColorbarBase(ax2,cmap=cmap,norm=norm,orientation='vertical')
    # cb1.set_label(r'$\exp \left( MDL - DL \right)$')
    # fig.legend(loc='upper right', bbox_to_anchor=(0.8, 0.95))
    # fig.tight_layout()
    # # fig.savefig(likelihood.fig_dir + '/plot_all_comp.png', dpi=300)
    # fig.clf()
    # plt.close(fig)

    # fig2.tight_layout()
    # fig2.legend(loc='upper right')
    # # plt.show()
    # # fig2.savefig(likelihood.fig_dir + '/density_plot_all_comp.png', dpi=300)
    # fig2.clf()
    # plt.close(fig2)

    # axfig3.errorbar(likelihood.xvar, likelihood.yvar, yerr=likelihood.yerr, fmt='.', markersize=5, zorder=len(fcn_list)+1, capsize=1, elinewidth=1, color='k', alpha=1)
    # axfig3.set_xlabel(r'$r_{proj} / Mpc$')
    # axfig3.set_ylabel(r'$ESD / 10^{12} M_{sun}/Mpc^2$')
    # axfig3.set_yscale('log')
    # fig3.tight_layout()
    # fig3.savefig(likelihood.fig_dir + '/plot_data.png', dpi=300)
    # fig3.clf()
    # plt.close(fig3)

    return