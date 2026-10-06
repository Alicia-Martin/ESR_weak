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
import sympy
import pickle
import pandas as pd

from itertools import combinations

warnings.filterwarnings("ignore")

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm


def check_integrable(fcn):
    try:
        eq = sympy.sympify(fcn,
                        locals={"inv": inv,
                            "square": square,
                                "cube": cube,
                                "sqrt": sqrt,
                                "log": log,
                                "pow": pow,
                                "x": x,
                                "a0": a0,
                                "a1": a1,
                                "a2": a2})

        tmax = 60
        try:
            with simplifier.time_limit(tmax):
                integral  = sympy.integrate(eq, x)

                if integral == None:
                    return False
                elif integral.has(sympy.Integral):
                    return False
                else:
                    return True
        except simplifier.TimeoutException:
            return "timeout"
        except Exception:
            return False
    except Exception as e:
        print(fcn)
        print(e)
        return True

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


def get_functions_global_params(fcn_list_sorted, DL_sorted, p0, p1, p2, p3, u1, u2, u3, u4, comp, katz,  BIC = False):
    Delta_DL_local = DL_sorted - DL_sorted[5]

    params = np.array([p0, p1, p2, p3]).T
    uncertainties = np.array([u1, u2, u3, u4]).T 

    print('shape', np.array(params).shape)

    #params shape: (number of clusters, number of functions, 4 params)
    #uncertainties shape: (number of clusters, number of functions, 4 params)

    def calculate_Delta(params, uncertainties, BIC = False, max_combination_size=4):
        if BIC:
            n_global = 1
            n_clusters = 12
            n_points = n_clusters*9
            #BIC criteria
            Delta_nparams = n_global*n_clusters - n_global
            Delta_BIC = Delta_nparams*np.log(n_points) + 2*Delta_DL_local
            
            return Delta_BIC, 0

        else:
            Delta_MDL_values = []
            param_indices_values = []
            N = len(params) #number of clusters
            print('N', N)
            for num_params in range(1, min(max_combination_size, params.shape[2]) + 1):
                for param_indices in combinations(range(params.shape[2]), num_params):
                    global_params = params[:, :, param_indices]
                    global_uncertainties = uncertainties[:, :, param_indices]

                    Fisher_diag = 12./global_uncertainties**2

                    fisher_term = 1/2*np.log(Fisher_diag) + np.log(np.abs(global_params))
                    fisher_term[global_params == 0] = 0
                    fisher_term[global_uncertainties == 0] = 0
                    sum_over_global_params = np.sum(fisher_term, axis=2)
                    # print('Delta_MDL', sum_over_global_params)
                    # sys.exit()

                    #sum over all galaxies
                    # print('sum_over_global_params', np.sum(sum_over_global_params, axis=0))
                    Delta_MDL = np.sum(sum_over_global_params, axis=0) + len(param_indices)*(np.log(2) - N/2*np.log(3))
                    # print(len(param_indices)*(np.log(2) - N/2*np.log(3)))
                    # print(N)
                    # print('Delta_MDL', Delta_MDL)
                    # sys.exit()

                    Delta_MDL_values.append(Delta_MDL)
                    param_indices_values.append(param_indices)
            
            return Delta_MDL_values, param_indices_values
            

    max_Delta_global, indices = calculate_Delta(params, uncertainties) #index_combination, #number of fucntions

    func_to_save = []
    indices_to_save = []
    comps_to_save = []
    diff_to_sort = []
    num = 0
    #print('katz', katz)
    #print(len(fcn_list_sorted))
    for i in range(len(fcn_list_sorted)):
        for index_combination in range(len(max_Delta_global)):
            # print(num, fcn_list_sorted[i], max_Delta_global[index_combination][i], Delta_DL_local[i])
            delta_difference = max_Delta_global[index_combination][i] - Delta_DL_local[i]

            if delta_difference > 0:
            #if max_Delta_global[index_combination][i] > Delta_DL_local[i]:
                num_params = simplifier.count_params([fcn_list_sorted[i]], 4)[0]
                #print(indices)
                #print(index_combination)
                indices_to_print = indices[index_combination]
                #maybe take out funcs with divergences
                if np.any(indices_to_print > (num_params - 1)):
                    continue
                else:
                    num += 1
                    #print(num, fcn_list_sorted[i], indices_to_print, max_Delta_global[index_combination][i]- Delta_DL_local[i])

                # return True
                    
                    func_to_save.append(fcn_list_sorted[i])
                    indices_to_save.append(indices_to_print)
                    comps_to_save.append(comp[i])

                    diff_to_sort.append((delta_difference, fcn_list_sorted[i], indices_to_print, comp[i]))
        # if Delta[i] > Delta_DL:
        #     print(fcn_list_sorted[i], Delta[i], DL_sorted[i], params[i], uncertainties[i])
        #     return True
                    
        # Sort the list by the difference in descending order
        diff_to_sort_sorted = sorted(diff_to_sort, key=lambda x: x[0], reverse=True)

        num_files = 3
        output_dir = 'esr/fitting/output/combining_clusters'
        # Divide las funciones en partes iguales
        chunks = [[] for _ in range(num_files)]
        for i, item in enumerate(diff_to_sort_sorted):
            chunks[i % num_files].append(item)

        # Guarda cada chunk en un archivo separado
        for i, chunk in enumerate(chunks):
            if katz:
                filename = f'global_local_funcs_part_{i + 1}_katz.txt'
            else:
                filename = f'global_local_funcs_part_{i + 1}.txt'
            file_path = os.path.join(output_dir, filename)
            with open(file_path, 'w') as f:
                for delta_diff, func, final_indices, complexity in chunk:
                    f.write(f'{func}|{str(final_indices)}|{complexity}\n')

        # Save the sorted functions
        if katz:
            filename = f'global_local_funcs_katz_duplicate.txt'
        else:
            filename = f'global_local_funcs_duplicate.txt'
        with open(os.path.join(output_dir, filename), 'w') as f:
            for delta_diff, func, final_indices, complexity in diff_to_sort_sorted:
                f.write(f'{func}|{str(final_indices)}|{complexity}\n')
                    
        # #save funcs and indices
        # with open('esr/fitting/output_decreasing/output/combining_clusters/glolbal_local_funcs.txt', 'w') as f:
        #     for i in range(len(func_to_save)):
        #         # f.write(func_to_save[i] + ', ' + str(indices_to_save[i]) + ', ' + str(comps_to_save[i]) + '\n')
        #         f.write(f'{func_to_save[i]}|{str(indices_to_save[i])}|{comps_to_save[i]}\n')
def statistics_params(fcn_list_sorted, p0, p1):

    #check if fucntion has abs and around which parameter
    def check_abs(fcn):
        if 'Abs' in fcn:
            if 'Abs(a0)' in fcn and not 'Abs(a1)' in fcn:
                return 0
            elif 'Abs(a1)' in fcn and not 'Abs(a0)' in fcn:
                return 1
            elif 'Abs(a0)' and 'Abs(a1)' in fcn:
                return 2
        else:
            return -1

    for fcn in fcn_list_sorted:
        abs_params = check_abs(fcn_list_sorted)
        if abs_params == 0:
            p0 = np.abs(p0)
        elif abs_params == 1:
            p1 = np.abs(p1)
        elif abs_params == 2:
            p0 = np.abs(p0)
            p1 = np.abs(p1)

        p0_nozero = p0[p0 != 0]
        p1_nozero = p1[p1 != 0]

        mean_p0 = np.mean(p0_nozero)
        mean_p1 = np.mean(p1_nozero)
        std_p0 = np.std(p0_nozero)
        std_p1 = np.std(p1_nozero)

        zero_p0 = np.sum(p0 == 0)
        zero_p1 = np.sum(p1 == 0)

    return mean_p0, mean_p1, std_p0, std_p1, zero_p0, zero_p1

def main(name, dirname, plot= False):
    use_katz = False
    #if use_katz:
    #    extra_name = '_katz'
    #else:
    extra_name = ''
    if use_katz:
        savename = 'combine_all_comp_' + str(name) + extra_name +'_duplicate_katz.txt'
        savename_dat = 'combine_all_comp_' + str(name) + extra_name +  '_duplicate_katz.dat'
    else:
        savename = 'combine_all_comp_' + str(name) + extra_name +'_duplicate.txt'
        savename_dat = 'combine_all_comp_' + str(name) + extra_name +  '_duplicate.dat'
    print(savename)
    if rank != 0:
        return

    all_f = os.listdir(dirname)
    #if use_katz:
    #    all_f = [f for f in all_f if f.startswith('final_') and f.endswith(extra_name + '.dat') and not f.endswith('_no_katz.dat') and not f.endswith('rerun_katz.dat') and 'global' not in f]
    #else:
    all_f = [f for f in all_f if f.startswith('final_') and f.endswith(extra_name + '.dat') and 'katz' not in f and 'rerun' not in f and 'global' not in f and 'funcs' not in f and 'likelihood' not in f]
        #all_f = [f for f in all_f if f.startswith('final_') and f.endswith('.dat')]
    #all_f = [f for f in all_f if f.startswith('final_') and not f.endswith('_katz.dat')]
    #all_f.sort()
    all_f.sort(key=lambda f: int(f[len('final_'):-len(extra_name + '.dat')]))
    #all_f.append('final_10_rerun_100_2.dat')
    print(all_f)
    all_comp = [int(f[len('final_'):-len(extra_name + '.dat')]) for f in all_f]

    all_f.append('final_10_rerun_100_2.dat')
    all_f.append('final_9_rerun_100_2.dat')
    all_comp.append(10)
    all_comp.append(9)
    print(all_f)
    print(dirname)
    params_f = os.listdir(dirname)
    #if use_katz:
    #    params_f = [f for f in params_f if f.endswith(extra_name + '.pkl') and not f.endswith('_no_katz.pkl') and not f.endswith('rerun_katz.pkl')]
        #params_f.sort()
    #    params_f.sort(key=lambda f: int(f[len('params_comp'):-len(extra_name + '.pkl')]))
    #else:
    params_f = [f for f in params_f if f.endswith('.pkl') and 'katz' not in f and 'rerun' not in f and 'funcs' not in f and 'likelihood' not in f]
    #params_f = [f for f in params_f if f.endswith('.pkl') and f.endswith('rerun.pkl')]
    #params_f.sort()
    params_f.sort(key=lambda f: int(f[len('params_comp'):-len('.pkl')]))
    #params_f = params_f[-4:-1]
    print(params_f)
    params_f.append('params_comp10_rerun_100_2.pkl')
    params_f.append('params_comp9_rerun_100_2.pkl')
    data = []
    p0, p1, p2, p3 = np.array([]), np.array([]), np.array([]), np.array([])
    d0, d1, d2, d3 = np.array([]), np.array([]), np.array([]), np.array([])
    for i, fname in enumerate(all_f):
        print(i, fname)

        with open(dirname + '/' +  fname, "r") as f:
            reader = csv.reader(f, delimiter=';')
            data_comp = [row for row in reader]
            #print([d[-5:-1] for d in data_comp])
            # print(len(data_comp))
            # if i == 4:
            #     print(data_comp[4])
            # print(data_comp)
            for row in data_comp:
                row.append(all_comp[i])
            data.extend(data_comp)


        with open(dirname + '/' +  params_f[i], "rb") as f:
            params = pickle.load(f)
            # if i == 4:
            #     print(params['a0'][4])
            p0 = np.concatenate((p0, params['a0'])) if p0.size else params['a0']
            p1 = np.concatenate((p1, params['a1'])) if p1.size else params['a1']
            p2 = np.concatenate((p2, params['a2'])) if p2.size else params['a2']
            p3 = np.concatenate((p3, params['a3'])) if p3.size else params['a3']

            d0 = np.concatenate((d0, params['d0'])) if d0.size else params['d0']
            d1 = np.concatenate((d1, params['d1'])) if d1.size else params['d1']
            d2 = np.concatenate((d2, params['d2'])) if d2.size else params['d2']
            d3 = np.concatenate((d3, params['d3'])) if d3.size else params['d3']

            # uncertainties = pickle.load(f)

        # print(data)
    # print('p0', len(p0))
    # print('p0', p0[16,:])
    # # print('d0', d0)
    # sys.exit()

    if not data:
        print("No functions with finite DL found, so will not make figure")
        return


    max_param = len(data[0]) - 7
    print('max_param', max_param)

    fcn_list = [d[1] for d in data]
    DL = np.array([float(d[2]) for d in data])
    Prel = np.array([float(d[2]) for d in data])
    negloglike = np.array([float(d[4]) for d in data])
    codelen = np.array([float(d[5]) for d in data])
    ayfeyn = np.array([float(d[6]) for d in data])
    katz = np.array([float(d[7]) for d in data])
    divergence = np.array([d[-2] for d in data])
    comp = np.array([int(d[-1]) for d in data])
    if use_katz:
        DL = np.array(negloglike + codelen + katz)
        DL[DL == 0] = np.inf
    #print(comp)
    #print([d[-6:-2] for d in data])
    params = np.array([d[-6:-2] for d in data], dtype=float)

    if not np.any(np.isfinite(DL)):
        print('Add DL are infinite, so skipping plot')
        return

    # Get indices that would sort DL in ascending order
    sorted_indices = np.argsort(DL)

    # print(len(DL))

    # Apply the sorted indices to all arrays
    fcn_list_sorted = [fcn_list[i] for i in sorted_indices]
    DL_sorted = DL[sorted_indices]
    print(DL_sorted)
    # Prel_sorted = Prel[sorted_indices]
    negloglike_sorted = negloglike[sorted_indices]
    codelen_sorted = codelen[sorted_indices]
    ayfeyn_sorted = ayfeyn[sorted_indices]
    katz_sorted = katz[sorted_indices]
    comp_sorted = comp[sorted_indices]
    params_sorted = params[sorted_indices]

    # print('pp', p0)


    # for i in range(len(p0)):
    #     for j in range(len(p0[i])):
    #         if d0[i][j] == 0 and p0[i][j] != 0:
    #             print('p0', p0[i][j])
                # print(fcn_list_sorted[i])
    # print('p0', p0)
    # sys.exit()

    #sorted params and uncertainties
    p0 = p0[sorted_indices]
    p1 = p1[sorted_indices]
    p2 = p2[sorted_indices]
    p3 = p3[sorted_indices]

    d0 = d0[sorted_indices]
    d1 = d1[sorted_indices]
    d2 = d2[sorted_indices]
    d3 = d3[sorted_indices]

    #Calculate Prel
    Prel_DL = DL_sorted - DL_sorted[0]                # Always gives 0 for the 0th function, so this gets the highest Prel

    Prel = np.exp(-Prel_DL)             # Don't want to use every fcn here bc they could be inf or nan, but the best 1000 should be fine
    Prel[~np.isfinite(Prel) | np.isnan(Prel)] = 0.0
    Prel /= np.sum(Prel)  

    # Table by complexity
    #headers = ["Rank", "Function", "DL", "Prel", "negloglike", "codelen", "ayfeyn", "katz"] + ["Mean a0", "Std a0", "Mean a1", "Std a1","zero_0", "zero_1",  "divergence"] + ["comp"]
    #table_data = []

    #seen_fn = {}
    #seen_likelihood = set()

    #negloglike_list = []
    #duplicate_list = []
    #no_duplicates_fcn_list = []  # List of unique functions
    #comp_list = []  # List of complexities for unique entries
    #DL_list = []  # List of DL values for unique entries
    #rank_fun = 0
    #for i in range(len(fcn_list_sorted)):
        #fn = fcn_list_sorted[i]
        #fn_str = str(fn)
        #nll = round(negloglike_sorted[i], 2)
        #DL = DL_sorted[i]
        #if DL_sorted[i] == np.inf or np.isnan(DL_sorted[i]):
        #    continue
        #if nll in no_duplicate_list:
            # Handle duplicates
        #    duplicate_list.append(nll)
        #    i_orig = negloglike_list.index(negloglike_sorted[i])  # Index of the original
        #    print(
        #        "%s is a duplicate of %s. "
        #        "The original has complexity=%d, DL=%f, negloglike=%f "
        #        "while the duplicate has complexity=%d, DL=%f, negloglike=%f"
        #        % (
        #            fcn_list_sorted[i], no_duplicates_fcn_list[i_orig],
        #            comp_list[i_orig], DL_list[i_orig], negloglike_list[i_orig],
        #            comp_sorted[i], DL_sorted[i], negloglike_sorted[i]
        #        )
        #    )
        #    continue
        #if fcn_list_sorted[i] in seen_fn:
        #    prev_i = seen_fn[fcn_list_sorted[i]]
        #    prev_nll = negloglike_sorted[prev_i]

        #    if nll < prev_nll:
                # Replace the stored entry with the better one
        #        seen_fn[fn_str] = i
                # Remove the old one from final lists and replace it below
        #        idx_to_remove = no_duplicates_fcn_list.index(fcn_list_sorted[prev_i])
        #        del no_duplicates_fcn_list[idx_to_remove]
        #        del negloglike_list[idx_to_remove]
        #        del comp_list[idx_to_remove]
        #       del DL_list[idx_to_remove]
        #    else:
        #        continue  # Keep old one, skip this one
        
        #else:
        #    seen_fn[fcn_list_sorted[i]] = i
        #else:
            # Process unique entries
        #rank_fun += 1
        #negloglike_list.append(negloglike_sorted[i])
        #seen_fn.append(fcn_list_sorted[i])
        #no_duplicates_fcn_list.append(fcn_list_sorted[i])  # Register function as unique
        #comp_list.append(comp_sorted[i])  # Register complexity
        #DL_list.append(DL_sorted[i])  # Register DL value
        #mean_p0, mean_p1, std_p0, std_p1, zero_p0, zero_p1 = statistics_params(fcn_list_sorted,p0[i], p1[i])
        #zero_p0, zero_p1 = 0, 0
        #row = [
        #        rank_fun,
        #        fcn_list_sorted[i],
        #        f'{DL_sorted[i]:.2f}',  # Use scientific notation for large numbers
        #        f'{Prel[i]:.2f}',
        #        f'{negloglike_sorted[i]:.2f}',
        #        f'{codelen_sorted[i]:.2f}',
        #        f'{ayfeyn_sorted[i]:.2f}',
        #        f'{katz_sorted[i]:.2f}'
        #    ] + [f'{p:.2e}' for p in params_sorted[i]] + [zero_p0] + [zero_p1] + [divergence[i]] + [comp_sorted[i]]
        #table_data.append(row)

    # First pass: keep only best version of each function
    best_indices = {}  # Maps fn_str to index i with best (lowest) negloglike
    seen_likelihood = set()
    for i, fn in enumerate(fcn_list_sorted):
        fn_str = str(fn)
        nll = round(negloglike_sorted[i], 2)
        DL_table = DL_sorted[i]
    
        # Skip invalid DLs
        if DL_table == np.inf or np.isnan(DL_table):
            continue
        if nll in seen_likelihood:
            #print(nll, fn)
            continue
        #else:
            #print('NOT', nll, fn)
        if fn_str not in best_indices:
            best_indices[fn_str] = i
            seen_likelihood.add(nll)
        else:
            i_prev = best_indices[fn_str]
            prev_nll = round(negloglike_sorted[i_prev], 2)
            if nll < prev_nll:
                #print('HERE', fn_str)
                best_indices[fn_str] = i  # Replace with better one
                seen_likelihood.discard(prev_nll)
                seen_likelihood.add(nll)
            else:
                continue

    # Now build the table using only the best versions
    headers = [
        "Rank", "Function", "DL", "Prel", "negloglike", "codelen", "ayfeyn", "katz",
        "Mean a0", "Std a0", "Mean a1", "Std a1", "zero_0", "zero_1", "divergence", "comp"
    ]
    table_data = []
    all_params_data = []
    #rank_fun = 0
    best_i_list = list(best_indices.values())

    # Sort these indices by DL
    best_i_list_sorted_by_DL = sorted(best_i_list, key=lambda idx: DL_sorted[idx])
    #for i in best_i_list_sorted_by_DL:
    #for i in range(0, 100):
    for rank_fun, i in enumerate(best_i_list_sorted_by_DL[:100], 1):
        #rank_fun += 1
        fn = fcn_list_sorted[i]
        DL = DL_sorted[i]
        nll = negloglike_sorted[i]
        #integrable = check_integrable(fn)
        integrable = False
        #if not integrable:
            #continue
        # Placeholder or replace with actual statistics
        mean_p0, std_p0 = np.mean(p0[i]), np.std(p0[i])
        mean_p1, std_p1 = np.mean(p1[i]), np.std(p1[i])
        mean_p0, mean_p1, std_p0, std_p1, zero_p0, zero_p1 = statistics_params(fcn_list_sorted,p0[i], p1[i])
        #zero_p0, zero_p1 = 0, 0
        row = [
            rank_fun,
            fcn_list_sorted[i],
            f'{DL:.2f}',
            f'{Prel[i]:.2f}',
            f'{nll:.2f}',
            f'{codelen_sorted[i]:.2f}',
            f'{ayfeyn_sorted[i]:.2f}',
            f'{katz_sorted[i]:.2f}',
            f'{mean_p0:.2e}',
            f'{std_p0:.2e}',
            f'{mean_p1:.2e}',
            f'{std_p1:.2e}',
            zero_p0,
            zero_p1,
            integrable,
            comp_sorted[i]
        ]
        table_data.append(row)
        pretty_table = PrettyTable()
        pretty_table.field_names = headers

        for cluster_idx, (param0, param1, delta0, delta1) in enumerate(zip(p0[i], p1[i], d0[i], d1[i])):
            all_params_data.append([rank_fun, fn, cluster_idx, param0, param1, delta0, delta1])

    for row in table_data:
        pretty_table.add_row(row)
    print(pretty_table)

    output_file = dirname + '/' +  savename
    print(f"Saving table to {output_file}")

    #save table_data to a file
    output_file_dat = dirname + '/' +  savename_dat
    df = pd.DataFrame(table_data)

    # Save to a .dat file
    df.to_csv(output_file_dat, sep="\t", index=False)
    print(f"Saving table to {output_file_dat}")

    # Save all params for all clusters to a separate CSV
    all_params_file = dirname + "/all_params_comp_9.csv"
    print(f"Saving all parameters to {all_params_file}")
    with open(all_params_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Rank", "Function", "ClusterIndex", "a0", "a1", "d0", "d1"])
        writer.writerows(all_params_data)

    # with open(output_file, 'w') as f:
    #     f.write(str(pretty_table))

    # name_cluster = '76'
    # plot_best_funs_one_galaxy(name_cluster, fcn_list_sorted)

    if plot:
        plot_combined(name, fcn_list_sorted, params_sorted, DL_sorted, dirname, NFW=True)

    # Save the table to a file
    with open(output_file, "w") as f:
        f.write(pretty_table.get_string())


    #check for which fucntions we need to do global and local
    #get_functions_global_params(fcn_list_sorted, DL_sorted, p0, p1, p2, p3, d0, d1, d2, d3, comp_sorted, use_katz, BIC = False)











