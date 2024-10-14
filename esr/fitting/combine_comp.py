import csv
from mpi4py import MPI
import warnings
import numpy as np
import os
from prettytable import PrettyTable
import sys
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl
from esr.fitting.sympy_symbols import *
from esr.esd import ExcessSurfaceDensity
from esr.generation import simplifier
from esr.fitting.WL_likelihood import WLLikelihood

warnings.filterwarnings("ignore")

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm

def plot_combined(name, fcn_list, params, DL, dirname, xscale='log', yscale='log', NFW=False):
    print('\nMaking plots', flush=True)

    import matplotlib as mpl

    # Set global font sizes
    mpl.rcParams['axes.labelsize'] = 16  # Font size for axis labels
    mpl.rcParams['xtick.labelsize'] = 12  # Font size for x-axis tick labels
    mpl.rcParams['ytick.labelsize'] = 12  # Font size for y-axis tick labels
    mpl.rcParams['legend.fontsize'] = 12  # Font size for the legend
    mpl.rcParams['font.size'] = 14  # General font size

    vmin = 1e-2
    vmax = 1
    tmax = 5

    NFW= True

    # likelihood and data
    data_file = 'XXL/' + str(name) + '.txt'
    run_name = 'WL_' + str(name)
    likelihood = WLLikelihood(data_file, run_name, data_dir=None, fn_set='core_maths')

    # if NFW:
    fcn_NFW = ['a0 / (x * (a1 + x)**2)']
    params_NFW = np.array([[45.42799462, 0.36449588, 0, 0]])
    DL_NFW = 33.90233990017775

    fcn_list = fcn_NFW + fcn_list
    params = np.concatenate((params_NFW, params), axis=0)
    DL = np.hstack((DL_NFW, DL))

    DL_min = np.amin(DL[np.isfinite(DL)])
    alpha = DL_min - DL
    alpha = np.exp(alpha)

    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    number_fun = 4
    vmin = np.min(alpha[:number_fun])*10**-1
    print(alpha[:number_fun])
    print(vmin)

    # cmap = cm.hot_r
    cmap= cm.seismic
    norm = mpl.colors.LogNorm(vmin=vmin, vmax=vmax)
    #not log
    # norm = mpl.colors.Normalize(vmin=0,vmax=1)

    for i in range(min(len(fcn_list), 4)):
        fcn_i = fcn_list[i].replace('\'', '')
        max_param = 4
        
        k = simplifier.count_params([fcn_i], max_param)[0]
        measured = params[i, :k]

        print('%i of %i:' % (i + 1, len(fcn_list)), fcn_i, DL[i], alpha[i], measured)

        fcn_i, eq = likelihood.run_sympify(fcn_i, tmax=tmax)

        if k == 0:
            eq_numpy = sympy.lambdify([x], eq, modules=["numpy"])
        elif k > 1:
            all_a = ' '.join([f'a{i}' for i in range(k)])
            all_a = list(sympy.symbols(all_a, real=True))
            eq_numpy = sympy.lambdify([x] + all_a, eq, modules=["numpy"])
        else:
            eq_numpy = sympy.lambdify([x, a0], eq, modules=["numpy"])

        x_array = np.linspace(np.min(likelihood.xvar), likelihood.xvar.max(), 1000)

        esd = ExcessSurfaceDensity.calculate(x_array, eq_numpy, params=measured)
        ypred = eq_numpy(x_array, *measured)

        # Plot the ESD on the first subplot
        if NFW and i == 0:
            ax1.plot(x_array, esd, color='red', zorder=len(fcn_list)-i, label='NFW')
        else:
            ax1.plot(x_array, esd, color='k', zorder=len(fcn_list)-i)

        # Plot density on the second subplot
        if NFW and i == 0:
            ax2.plot(x_array, ypred, color='red', zorder=len(fcn_list)-i)
        elif i == 3:
            ax2.plot(x_array, ypred + 100, color='k', zorder=len(fcn_list)-i)
        else:
            ax2.plot(x_array, ypred, color='k', zorder=len(fcn_list)-i)

    # Adding likelihood data points to ax1 (ESD plot)
    if hasattr(likelihood, 'yerr'):
        ax1.errorbar(likelihood.xvar, likelihood.yvar, yerr=likelihood.yerr, fmt='.', markersize=5, zorder=len(fcn_list)+1, capsize=1, elinewidth=1, color='black', alpha=1)
    else:
        ax1.plot(likelihood.xvar, likelihood.yvar, '.', color='k', ms=5, zorder=len(fcn_list)+1, alpha=1)

    # Set scales and labels
    ax1.set_xscale(xscale)
    ax1.set_yscale(yscale)
    ax1.set_xlabel(r'$R (Mpc)$')
    ax1.set_ylabel(r'$ESD (10^{12} M_{\odot}/Mpc^2)$')
    ax1.set_ylim(likelihood.yvar.min() * 0.9, likelihood.yvar.max() * 1.1)
    ax1.legend(loc='upper right')

    ax2.set_xscale(xscale)
    ax2.set_yscale(yscale)
    ax2.set_xlabel(r'$r (Mpc)$')
    ax2.set_ylabel(r'$\rho (10^{12} M_{sun}/Mpc^3)$')
    # ax2.legend(loc='upper right')

    # Create a shared colorbar
    # fig.subplots_adjust(right=0.85)
    # cbar_ax = fig.add_axes([0.88, 0.1, 0.02, 0.8])
    # cb1 = mpl.colorbar.ColorbarBase(cbar_ax, cmap=cmap, norm=norm, orientation='vertical')
    # cb1.set_label(r'$\exp \left( MDL - L(D) \right)$')
    plt.show()
    # Save the figure
    fig.savefig(dirname + '/combined_plot_' + str(name) + '.png', dpi=300)
    print(dirname + '/combined_plot_' + str(name) + '.png')

    plt.close(fig)



def plot(name, fcn_list, params, DL, dirname, xscale='log', yscale='log', NFW=True):
    print('\nMaking plots', flush=True)

    import matplotlib as mpl

    # Set global font sizes
    mpl.rcParams['axes.labelsize'] = 12  # Font size for axis labels
    mpl.rcParams['xtick.labelsize'] = 11  # Font size for x-axis tick labels
    mpl.rcParams['ytick.labelsize'] = 11  # Font size for y-axis tick labels
    # mpl.rcParams['legend.fontsize'] = 12  # Font size for the legend
    mpl.rcParams['font.size'] = 12  # General font size

    vmin = 1e-3
    vmax = 1
    tmax = 5

    #likelihood and data
    data_file = 'XXL/' + str(name) + '.txt'
    run_name = 'WL_' + str(name)
    likelihood = WLLikelihood(data_file, run_name, data_dir=None, fn_set = 'core_maths')  

    NFW=True

    if NFW:
        # fcn_NFW = ['a0 / (x * (a1 + x)**2)']
        # params_NFW = np.array([[70.18463205, 0.79648191, 0, 0]])
        # DL_NFW = 123.3880824099034

        #91
        fcn_NFW = ['a0 / (x * (a1 + x)**2)']
        params_NFW = np.array([[45.42799462, 0.36449588, 0, 0]])
        DL_NFW = 33.90233990017775

        fcn_list = fcn_NFW + fcn_list
        params = np.concatenate((params_NFW, params), axis=0)
        DL = np.hstack((DL_NFW, DL))

    DL_min = np.amin(DL[np.isfinite(DL)])
    alpha = DL_min - DL
    alpha = np.exp(alpha)


    fig  = plt.figure(figsize=(7,5))
    ax1  = fig.add_axes([0.10,0.10,0.70,0.85])
    cmap = cm.hot_r
    #not log
    # norm = mpl.colors.Normalize(vmin=vmin,vmax=vmax)

    number_fun = 4
    vmin = np.min(alpha[:number_fun])*10**-1
    norm = mpl.colors.LogNorm(vmin=vmin,vmax=vmax)

    fig2 = plt.figure(figsize=(7,5))
    axfig2  = fig2.add_axes([0.10,0.10,0.70,0.85])


    for i in range(min(len(fcn_list),5)):

        fcn_i = fcn_list[i].replace('\'', '')
        max_param = 4
        
        k = simplifier.count_params([fcn_i], max_param)[0]
        measured = params[i,:k]

        print('%i of %i:'%(i+1,len(fcn_list)), fcn_i, DL[i], alpha[i])
        
        # try:
        fcn_i, eq= likelihood.run_sympify(fcn_i, tmax=tmax)
        
        if k == 0:
            eq_numpy = sympy.lambdify([x], eq, modules=["numpy"])
        elif k > 1:
            all_a = ' '.join([f'a{i}' for i in range(k)])
            all_a = list(sympy.symbols(all_a, real=True))
            eq_numpy = sympy.lambdify([x] + all_a, eq, modules=["numpy"])
        else:
            eq_numpy = sympy.lambdify([x, a0], eq, modules=["numpy"])

        #Plot the ESD on the first figure        
        # x_array = np.linspace(0, likelihood.xvar.max(), 1000)
        x_array = np.linspace(np.min(likelihood.xvar), likelihood.xvar.max(), 1000)

        esd = ExcessSurfaceDensity.calculate(x_array, eq_numpy, params=measured)
        ypred = eq_numpy(x_array, *measured)

        # if NFW and i == 0:
        #     ax1.plot(x_array, esd, color='blue', zorder=len(fcn_list)-i, label='NFW')
        # else:
        if np.isscalar(esd):
            ax1.plot(x_array, [esd]*len(likelihood.xvar), color=cmap(norm(alpha[i])), zorder=len(fcn_list)-i)
        else:
            ax1.plot(x_array, esd, color=cmap(norm(alpha[i])), zorder=len(fcn_list)-i)


        #plot density
        if NFW and i == 0:
            axfig2.plot(x_array, ypred, color='blue', zorder=len(fcn_list)-i, label='NFW')
        else:
            if np.isscalar(ypred):
                axfig2.plot(x_array, [ypred]*len(x_array), color=cmap(norm(alpha[i])), zorder=len(fcn_list)-i)
            else:
                axfig2.plot(x_array, ypred, color=cmap(norm(alpha[i])), zorder=len(fcn_list)-i)
        
    if hasattr(likelihood, 'yerr'):
        ax1.errorbar(likelihood.xvar, likelihood.yvar, yerr=likelihood.yerr, fmt='.', markersize=5, zorder=len(fcn_list)+1, capsize=1, elinewidth=1, color='darkblue', alpha=1)
    else:
        ax1.plot(likelihood.xvar, likelihood.yvar, '.', color='k', ms=5, zorder=len(fcn_list)+1, alpha=1)
    ax1.set_xlabel(r'$r_{proj} (Mpc)$')
    ax1.set_ylabel(r'$ESD (10^{12} M_{\odot}/Mpc^2)$')
    ax1.set_xscale(xscale)
    ax1.set_yscale(yscale)
    if xscale != 'log':
        ax1.set_xlim(0, None)
    ax1.set_ylim(likelihood.yvar.min() * 0.9, likelihood.yvar.max() * 1.1)

    ax2  = fig.add_axes([0.85,0.10,0.05,0.85])
    cb1  = mpl.colorbar.ColorbarBase(ax2,cmap=cmap,norm=norm,orientation='vertical')
    cb1.set_label(r'$\exp \left( MDL - DL \right)$')
    fig.tight_layout()
    # fig.legend(loc='upper right', bbox_to_anchor=(0.8, 0.95))
    fig.savefig(dirname + '/plot_cluster_report' + str(name) + '.png', dpi=300)
    print(dirname + '/plot_cluster_' + str(name) + '.png')
    fig.clf()
    # plt.show()
    plt.close(fig)

    # axfig2.set_xscale(xscale)
    # axfig2.set_yscale('log')
    # axfig2.set_xlabel(r'$r (Mpc)$')
    # axfig2.set_ylabel(r'$\rho (10^{12} M_{sun}/Mpc^3)$')
    # fig2.tight_layout()
    # fig2.legend(loc='upper right')
    # # plt.show()
    # fig2.savefig(dirname + '/density_plot_all_comp.png', dpi=300)
    # fig2.clf()
    # plt.close(fig2)


# def plot_best_funs_one_galaxy(name, best_funcs):
#     #load final and get the params
#     if rank != 0:
#         return
#     dirname = 'esr/fitting/output_glamdring/output/output_WL_' + str(name)
#     all_f = os.listdir(dirname)
#     all_f = [f for f in all_f if f.startswith('combine_DL_')]
#     all_f.sort()
#     all_funcs = [f for f in all_f if f.startswith('combine_DL_fun_comp_')]
#     # print(all_f)
#     all_comp = [int(f[len('final_'):-len('.dat')]) for f in all_f]
    
#     data = []
#     for i, fname in enumerate(all_f):
#         print(i, fname)

#         with open(dirname + '/' +  fname, "r") as f:
#             reader = csv.reader(f, delimiter=';')
#             data_comp = [row for row in reader]
#             # print(data_comp)
#             for row in data_comp:
#                 row.append(all_comp[i])
#             data.extend(data_comp)

        
#         # print(data)
            
    # max_param = len(data[0]) - 7
    # print('max_param', max_param)

    # fcn_list = 
    # DL = np.array([float(d[2]) for d in data])
    # Prel = np.array([float(d[3]) for d in data])
    # negloglike = np.array([float(d[4]) for d in data])
    # codelen = np.array([float(d[5]) for d in data])
    # ayfeyn = np.array([float(d[6]) for d in data])
    # comp = np.array([int(d[-1]) for d in data])

    # params = np.array([d[-max_param:(4 - max_param)] for d in data], dtype=float)

    # #find the index of best fucns in combine
    # funcs = []
    # params_funcs = []
    # DL_funcs = []


    # for fun in best_funcs[:10]:
    #     index = fcn_list.index(fun)
    #     print('index', index)
    #     print('fun', fun)
    #     print('params', params[index])
    #     print('DL', DL[index])
    #     print('Prel', Prel[index])
    #     print('negloglike', negloglike[index])
    #     print('codelen', codelen[index])
    #     print('ayfeyn', ayfeyn[index])
    #     print('comp', comp[index])

    #     #save the best funcs to later plot
    #     funcs.append(fun)
    #     params_funcs.append(params[index])
    #     DL_funcs.append(DL[index])

    # #plot the best funcs\
    # plot(name, funcs, params_funcs, DL_funcs, dirname, xscale='log', yscale='log', NFW=True)

# def plot_params():



def main(name, dirname, plot= False):
    """Plot best 50 functions at given complexity against data and save plot to file

    Args:
        :comp (int): complexity of functions to consider
        :likelihood (fitting.likelihood object): object containing data, functions to convert SR expressions to variable of data and output path
        :tmax (float, default=5.): maximum time in seconds to run any one part of simplification procedure for a given function
        :try_integration (bool, default=False): when likelihood requires integral, whether to try to analytically integrate (True) or just numerically integrate (False)
        :xscale (str, default='linear'): Scaling for x-axis
        :yscale (str, default='linear'): Scaling for y-axis

    Returns:
        None

    """

    savename = 'combine_all_comp_' + str(name) + '.txt'

    if rank != 0:
        return

    all_f = os.listdir(dirname)
    all_f = [f for f in all_f if f.startswith('final_')]
    all_f.sort()
    # print(all_f)
    all_comp = [int(f[len('final_'):-len('.dat')]) for f in all_f]
    
    data = []
    for i, fname in enumerate(all_f):
        print(i, fname)

        with open(dirname + '/' +  fname, "r") as f:
            reader = csv.reader(f, delimiter=';')
            data_comp = [row for row in reader]
            # print(data_comp)
            for row in data_comp:
                row.append(all_comp[i])
            data.extend(data_comp)
        # print(data)

    if not data:
        print("No functions with finite DL found, so will not make figure")
        return


    max_param = len(data[0]) - 7
    print('max_param', max_param)

    fcn_list = [d[1] for d in data]
    DL = np.array([float(d[2]) for d in data])
    Prel = np.array([float(d[3]) for d in data])
    negloglike = np.array([float(d[4]) for d in data])
    codelen = np.array([float(d[5]) for d in data])
    ayfeyn = np.array([float(d[6]) for d in data])
    comp = np.array([int(d[-1]) for d in data])

    params = np.array([d[-max_param:(4 - max_param)] for d in data], dtype=float)

    if not np.any(np.isfinite(DL)):
        print('Add DL are infinite, so skipping plot')
        return

    # Get indices that would sort DL in ascending order
    sorted_indices = np.argsort(DL)

    # Apply the sorted indices to all arrays
    fcn_list_sorted = [fcn_list[i] for i in sorted_indices]
    DL_sorted = DL[sorted_indices]
    # Prel_sorted = Prel[sorted_indices]
    negloglike_sorted = negloglike[sorted_indices]
    codelen_sorted = codelen[sorted_indices]
    ayfeyn_sorted = ayfeyn[sorted_indices]
    comp_sorted = comp[sorted_indices]
    params_sorted = params[sorted_indices]

    #Calculate Prel
    Prel_DL = DL_sorted - DL_sorted[0]                # Always gives 0 for the 0th function, so this gets the highest Prel

    Prel = np.exp(-Prel_DL)             # Don't want to use every fcn here bc they could be inf or nan, but the best 1000 should be fine
    Prel[~np.isfinite(Prel) | np.isnan(Prel)] = 0.0
    Prel /= np.sum(Prel)  

    # Table by complexity
    headers = ["Rank", "Function", "DL", "Prel", "negloglike", "codelen", "ayfeyn"] + [f'a{i}' for i in range(params_sorted.shape[1])] + ["comp"]
    table_data = []

    for i in range(len(fcn_list_sorted)):
        row = [
            i + 1,
            fcn_list_sorted[i],
            f'{DL_sorted[i]:.2f}',  # Use scientific notation for large numbers
            f'{Prel[i]:.2f}',
            f'{negloglike_sorted[i]:.2f}',
            f'{codelen_sorted[i]:.2f}',
            f'{ayfeyn_sorted[i]:.2f}'
        ] + [f'{p:.2f}' for p in params_sorted[i]] + [comp_sorted[i]]
        table_data.append(row)

    pretty_table = PrettyTable()
    pretty_table.field_names = headers

    for row in table_data:
        pretty_table.add_row(row)
    print(pretty_table)

    output_file = dirname + '/' +  savename
    print(f"Saving table to {output_file}")
    # with open(output_file, 'w') as f:
    #     f.write(str(pretty_table))

    # name_cluster = '76'
    # plot_best_funs_one_galaxy(name_cluster, fcn_list_sorted)

    if plot:
        plot_combined(name, fcn_list_sorted, params_sorted, DL_sorted, dirname, NFW=True)

    # Save the table to a file
    with open(output_file, "w") as f:
        f.write(pretty_table.get_string())
