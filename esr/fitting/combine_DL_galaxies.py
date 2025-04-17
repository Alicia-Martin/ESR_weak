import numpy as np
import csv
import sys
import esr.plotting.plot
from mpi4py import MPI
from prettytable import PrettyTable
from esr.fitting.WL_likelihood import WLLikelihood
import os
from esr.plotting.plot import pareto_plot
from esr.fitting import test_all
from esr.generation.simplifier import inverse_convert_params
import esr.generation.simplifier as simplifier
import sympy as sp
import pickle
import re
import matplotlib.pyplot as plt
from esr.fitting.sympy_symbols import *

def load_equations_and_matches(fcn_dir, comp):
    with open(fcn_dir + "/compl_%i/unique_equations_%i.txt" % (comp, comp)) as f:
        fcn = f.read().splitlines()
    invsubs_file = fcn_dir + "/compl_%i/inv_subs_%i.txt" % (comp, comp)
    max_param = 4
    all_invs_subs = simplifier.load_subs(invsubs_file, max_param)
    with open(fcn_dir + "/compl_%i/all_equations_%i.txt" % (comp, comp)) as f:
        all_eqs = f.read().splitlines()
    return fcn, all_invs_subs, all_eqs

def save_params_with_pickle(filepath, a0, a1, a2, a3, d0, d1, d2, d3):
    params = {
        "a0": a0,
        "a1": a1,
        "a2": a2,
        "a3": a3,
        "d0": d0,
        "d1": d1,
        "d2": d2,
        "d3": d3
    }
    
    with open(filepath, 'wb') as file:
        pickle.dump(params, file)
    # print(a0)
    print(f"Saved parameters to {filepath}")

def plot_params(a0, a1, k):
    if k==1:
        plt.plot(a0, marker='o', label='a0')  # Plot a0 if available
        plt.xlabel('galaxy cluster')
        plt.ylabel('a0')

    elif k==2:
        plt.scatter(a0, a1, marker='o', label='a1 vs a0')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('a0')
        plt.ylabel('a1')

    elif k==0:
        print('No parameters for this function')

def sanitize_filename(name):
    # Replace any invalid characters with an underscore or another safe character
    return re.sub(r'[<>:"/\\|?*]', '_', name)

def check_divergences(param0, param1, param2, param3, fcn):
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
        
    except Exception as e:
        print(fcn)
        print(e)
        return True

    for i in range(len(param0)):
        params = [param0[i], param1[i], param2[i], param3[i]]

        a_symbols = [a0, a1, a2, a3]

        try:
            substitutions = {a_symbols[i]: params[i] for i in range(len(params))}
            eq_substituted = eq.subs(substitutions)

        # print(params)

            domain=sympy.Interval.open(0, sympy.oo)
            continuous = sympy.calculus.util.continuous_domain(eq_substituted, x, domain=domain)


            if continuous == domain:
                # print("The equation is continuous over the entire domain.")
                is_finite = True
            else:
                # Points or regions missing from the continuous domain
                divergence_points = domain - continuous
                # print(params)
                # print(f"Divergences found at: {divergence_points}", continuous)
                return False
            
        except Exception as e:
            return True
        
    return is_finite

def combine_all_galaxies(dirname, aifeyn_file, names, all_fcn, comp):
            #run each cluster
        max_params = 4
        count_zeroes = np.zeros(len(all_fcn))
        clusters_with_inf_per_fcn = [[] for _ in range(len(all_fcn))]
        negloglike = None

        for index, name in enumerate(names):
            # print(name)
            try:
                out_dir = dirname + 'output_WL_' + name 
            except: 
                out_dir = dirname + 'output_WL_' + name +  '_all_fun'

            data = np.genfromtxt(out_dir + "/codelen_matches_comp"+str(comp)+".dat") # All
            negloglike_value = data[:,0]
            codelen_value = data[:,1]
            index = data[:,2]
            p = data[:,3:3 + max_params]
            delta = data[:,3 + max_params:-3]
            Nconv = data[:,-3]
            Niter = data[:,-2]
            time = data[:, -1]
            # print('NAME', name)
            # print('fcn', all_fcn[1565])
            # print('codelen', codelen_value[1565])
            # print('negloglike', negloglike_value[1565])
            # print('params', p[1565, :])
            # print('delta', delta[1565, :])

            # print('fcn', all_fcn[1595])
            # print('codelen', codelen_value[1595])
            # print('negloglike', negloglike_value[1595])
            # print('params', p[1595, :])
            # print('delta', delta[1595, :])

            aifeyn_value = np.genfromtxt(aifeyn_file) # All
            codelen_value = np.atleast_1d(codelen_value)
            index = np.atleast_1d(index)
            aifeyn = np.atleast_1d(aifeyn_value)
            Nconv = np.atleast_1d(Nconv)
            Niter = np.atleast_1d(Niter)
            time = np.atleast_1d(time)

            data_file = 'XXL/' + str(name) + '.txt'
            run_name = 'WL_' + str(name)

            #save all params
            param0 = p[:, 0].reshape(-1, 1)
            param1 = p[:, 1].reshape(-1, 1)
            param2 = p[:, 2].reshape(-1, 1)
            param3 = p[:, 3].reshape(-1, 1)

            delta0 = delta[:, 0].reshape(-1, 1)
            delta1 = delta[:, 1].reshape(-1, 1)
            delta2 = delta[:, 2].reshape(-1, 1)
            delta3 = delta[:, 3].reshape(-1, 1)

            # Check and count infinities
            is_negloglike_inf = negloglike_value == np.inf
            is_codelen_inf = codelen_value == np.inf

            for idx, (neg_inf, code_inf) in enumerate(zip(is_negloglike_inf, is_codelen_inf)):
                if neg_inf or code_inf:
                    print(idx)
                    clusters_with_inf_per_fcn[idx].append(name)

            # Update count_zeroes
            count_zeroes += is_negloglike_inf
            count_zeroes += is_codelen_inf


            if negloglike is None:
                negloglike = negloglike_value
                codelen = codelen_value
                p0, p1, p2, p3 = param0, param1, param2, param3
                d0, d1, d2, d3 = delta0, delta1, delta2, delta3

            else:
                negloglike += negloglike_value
                codelen += codelen_value

                p0 = np.concatenate((p0, param0), axis=1)
                p1 = np.concatenate((p1, param1), axis=1)
                p2 = np.concatenate((p2, param2), axis=1)
                p3 = np.concatenate((p3, param3), axis=1)

                d0 = np.concatenate((d0, delta0), axis=1)
                d1 = np.concatenate((d1, delta1), axis=1)
                d2 = np.concatenate((d2, delta2), axis=1)
                d3 = np.concatenate((d3, delta3), axis=1)

        # Count zeroes                
        for j in range(len(all_fcn)):
            if count_zeroes[j] != 0 and count_zeroes[j] != len(names):
                print(count_zeroes[j], all_fcn[j], '\n', clusters_with_inf_per_fcn[j])

        # Process your data and calculate L, then save to a CSV file
        #arregalr esto para no considerar params that are snapped to zero and also abs should be considered the same minus or plus
        p0_nozero = np.where(p0 != 0, p0, np.nan)
        p1_nozero = np.where(p1 != 0, p1, np.nan)

        mean_p0 = np.mean(p0_nozero, axis=1)
        mean_p1 = np.mean(p1_nozero, axis=1)
        std_p0 = np.std(p0_nozero, axis=1)
        std_p1 = np.std(p1_nozero, axis=1)

        # print('hola', len(negloglike))


        return negloglike, codelen, aifeyn, index, p0, p1, p2, p3, d0, d1, d2, d3, mean_p0, std_p0, mean_p1, std_p1
        
def find_lowest_DL(unique_fcn, all_fcn, index, xarr, negloglike, codelen, aifeyn, p0, p1, p2, p3, d0, d1, d2, d3, mean_p0, std_p0, mean_p1, std_p1):
        DL_min = np.zeros(len(unique_fcn))

        # print(p0[0,:].shape)
        # sys.exit()

        fcn_min = [None] * len(unique_fcn)
        negloglike_min = np.zeros(len(unique_fcn))
        codelen_min = np.zeros(len(unique_fcn))
        aifeyn_min = np.zeros(len(unique_fcn))
        index_all_equations = np.zeros(len(unique_fcn))

        # Arrays to store min p0, p1, etc.
        p0_min = np.zeros((len(unique_fcn), p0.shape[1]))  # Use p0.shape[1] for the second dimension
        p1_min = np.zeros((len(unique_fcn), p1.shape[1]))  # Same for p1, p2, p3, etc.
        p2_min = np.zeros((len(unique_fcn), p2.shape[1]))
        p3_min = np.zeros((len(unique_fcn), p3.shape[1]))
        d0_min = np.zeros((len(unique_fcn), d0.shape[1]))
        d1_min = np.zeros((len(unique_fcn), d1.shape[1]))
        d2_min = np.zeros((len(unique_fcn), d2.shape[1]))
        d3_min = np.zeros((len(unique_fcn), d3.shape[1]))
        mean_p0_min = np.zeros(len(unique_fcn))
        mean_p1_min = np.zeros(len(unique_fcn))
        std_p0_min = np.zeros(len(unique_fcn))
        std_p1_min = np.zeros(len(unique_fcn))

        
        for i in range(len(unique_fcn)):          # Loop over all unique fcns to find variant with min codelength
            # print(f'{i+1} of {len(unique_fcn)}', flush=True)
            mask = (index == xarr[i])
            # print(len(mask))
            # print(len(negloglike))
            negloglike_i, codelen_i, aifeyn_i = negloglike[mask], codelen[mask], aifeyn[mask]  # Arrays of all variants
            fcn_list_all_i = [all_fcn[j] for j in range(len(mask)) if mask[j]]  # All corresponding fcns
            p0_i, p1_i, p2_i, p3_i = p0[mask], p1[mask], p2[mask], p3[mask]
            d0_i, d1_i, d2_i, d3_i = d0[mask], d1[mask], d2[mask], d3[mask]
            mean_p0_i, std_p0_i = mean_p0[mask], std_p0[mask]
            mean_p1_i, std_p1_i = mean_p1[mask], std_p1[mask]
            DL = negloglike_i + codelen_i + aifeyn_i  # Calculate DL
            # if i==13:
            #     for item in range(len(fcn_list_all_i)):
            #         print(fcn_list_all_i[item], DL[item], negloglike_i[item], codelen_i[item], aifeyn_i[item])
            #     sys.exit()

            # If all DL values are NaN, skip this equation
            if np.sum(~np.isnan(DL)) == 0:
                DL_min[i] = np.nan
                continue

            # Find the minimum DL
            min_DL = np.nanmin(DL)
            min_idx_local = None

            # Prioritize the unique function over variants if DL matches
            for idx, fcn in enumerate(fcn_list_all_i):
                if DL[idx] == min_DL:
                    if fcn == unique_fcn[i]:  # Prioritize unique function
                        min_idx_local = idx
                        break
                    elif min_idx_local is None:
                        min_idx_local = idx

            # Store results
            DL_min[i] = min_DL
            p0_min[i], p1_min[i], p2_min[i], p3_min[i] = p0_i[min_idx_local], p1_i[min_idx_local], p2_i[min_idx_local], p3_i[min_idx_local]
            d0_min[i], d1_min[i], d2_min[i], d3_min[i] = d0_i[min_idx_local], d1_i[min_idx_local], d2_i[min_idx_local], d3_i[min_idx_local]
            mean_p0_min[i], std_p0_min[i] = mean_p0_i[min_idx_local], std_p0_i[min_idx_local]
            mean_p1_min[i], std_p1_min[i] = mean_p1_i[min_idx_local], std_p1_i[min_idx_local]
            fcn_min[i] = fcn_list_all_i[min_idx_local]

            # Find the minimum DL and corresponding params and function
            # DL_min[i] = np.nanmin(DL)
            # min_idx_local = np.nanargmin(DL) 
            # # print(p0_i[min_idx_local])
            # p0_min[i], p1_min[i], p2_min[i], p3_min[i] = p0_i[min_idx_local], p1_i[min_idx_local], p2_i[min_idx_local], p3_i[min_idx_local]
            # d0_min[i], d1_min[i], d2_min[i], d3_min[i] = d0_i[min_idx_local], d1_i[min_idx_local], d2_i[min_idx_local], d3_i[min_idx_local]
            # mean_p0_min[i], std_p0_min[i] = mean_p0_i[min_idx_local], std_p0_i[min_idx_local]
            # mean_p1_min[i], std_p1_min[i] = mean_p1_i[min_idx_local], std_p1_i[min_idx_local]
            # fcn_min[i] = fcn_list_all_i[min_idx_local] 
            # print(i, DL, aifeyn_i)

            # Map the local index back to the global index in the 'index' array
            global_idx_list = np.where(mask)[0]
            index_all_equations[i] = global_idx_list[min_idx_local] 

            negloglike_min[i] = negloglike_i[np.nanargmin(DL)]
            codelen_min[i] = codelen_i[np.nanargmin(DL)]
            aifeyn_min[i] = aifeyn_i[np.nanargmin(DL)]

        for i in range(len(DL_min)):
            if codelen_min[i] == np.inf:
                print('codelen is inf for', fcn_min[i], negloglike_min[i], codelen_min[i], aifeyn_min[i])


        return fcn_min, DL_min, negloglike_min, codelen_min, aifeyn_min, p0_min, p1_min, p2_min, p3_min, d0_min, d1_min, d2_min, d3_min, mean_p0_min, std_p0_min, mean_p1_min, std_p1_min

def main(dirname, fcn_dir, save_dir):

    max_params = 4

    with open('all_clusters.txt') as f:
        names = f.read().splitlines() 

    #choose a complexity
    for i in range(7,  8):
        print('COMP', i)
        #open the appropriate files 
        if i in [1, 2, 3, 4, 5, 6]:
            fcn_dir = 'esr/function_library/core_maths/'

        else:
            fcn_dir = 'esr/function_library/best_funcs'
            # fcn_dir = 'esr/function_library/core_maths/'

        unifn_file = fcn_dir + "/compl_%i/unique_equations_%i.txt"%(i,i)
        allfn_file = fcn_dir + "/compl_%i/all_equations_%i.txt"%(i,i)
        aifeyn_file = fcn_dir + "/compl_%i/aifeyn_%i.txt"%(i,i)

        # print(allfn_file)

        # fcn_dir = 'esr/function_library/core_maths/'

        with open(unifn_file, "r") as f:         # All
            unique_fcn = f.read().splitlines()

        with open(allfn_file, "r") as f:         # All
            all_fcn = f.read().splitlines()

        print(len(unique_fcn), len(all_fcn))  

        xarr = np.linspace(0, len(unique_fcn)-1, len(unique_fcn)).astype(int)           # Indices of all the unique fcns, which are what we're looping over 

        #combine all galaxies - this is L for every funcs win all funcs file
        negloglike, codelen, aifeyn, index, p0, p1, p2, p3, d0, d1, d2, d3, mean_p0, std_p0, mean_p1, std_p1 = combine_all_galaxies(dirname, aifeyn_file, names, all_fcn, i)

        #find variant with the lowest L
        fcn , L, negloglike, codelen, aifeyn, p0, p1, p2, p3, d0, d1, d2, d3, mean_p0, std_p0, mean_p1, std_p1 = find_lowest_DL(unique_fcn, all_fcn, index, xarr, negloglike, codelen, aifeyn, p0, p1, p2, p3, d0, d1, d2, d3, mean_p0, std_p0, mean_p1, std_p1 )

        sorted_data = sorted(zip(fcn, L, negloglike, codelen, aifeyn, mean_p0, std_p0, mean_p1, std_p1), key=lambda x: x[1])
        
        #sort data
        sorted_indices = np.argsort(L)
        functions_sorted = [fcn[i] for i in sorted_indices]
        negloglike_sorted = [negloglike[i] for i in sorted_indices]
        codelen_sorted = [codelen[i] for i in sorted_indices]
        aifeyn_sorted = [aifeyn[i] for i in sorted_indices]
        L_sorted = [L[i] for i in sorted_indices]
        mean_p0_sorted = [mean_p0[i] for i in sorted_indices]
        std_p0_sorted = [std_p0[i] for i in sorted_indices]
        mean_p1_sorted = [mean_p1[i] for i in sorted_indices]
        std_p1_sorted = [std_p1[i] for i in sorted_indices]

        sorted_p0 = p0[sorted_indices, :]
        sorted_p1 = p1[sorted_indices, :]
        sorted_p2 = p2[sorted_indices, :]
        sorted_p3 = p3[sorted_indices, :]

        sorted_d0 = d0[sorted_indices, :]
        sorted_d1 = d1[sorted_indices, :]
        sorted_d2 = d2[sorted_indices, :]
        sorted_d3 = d3[sorted_indices, :]

        # for fun_idx in range(len(functions_sorted)):
        #     if functions_sorted[fun_idx] == 'pow(x,a0)/a1':
        #         print('AQUI', sorted_p0[fun_idx, :])

        ptab = PrettyTable()
        ptab.field_names = ["#", "Function", "L(D)", "-logL", "Codelen", "AIFeyn", "Mean a0", "Std a0", "Mean a1", "Std a1", "Divergence"]

        #SAVE THE DATA TO A FILE
        with open(dirname +  save_dir + 'final_' + str(i) + '.dat', 'w') as f:
            writer = csv.writer(f, delimiter=';')
            # for n in range(min(400, len(sorted_data))):
            for n in range(len(sorted_data)):
                no_divergence = check_divergences(sorted_p0[n, :], sorted_p1[n, :], sorted_p2[n, :], sorted_p3[n, :], functions_sorted[n])
                divergence_flag = "---" if no_divergence else "YES"
                
                #save table
                data = [n, functions_sorted[n], L_sorted[n], negloglike_sorted[n], negloglike_sorted[n], codelen_sorted[n], aifeyn_sorted[n], mean_p0_sorted[n], std_p0_sorted[n], mean_p1_sorted[n], std_p1_sorted[n], divergence_flag]
                writer.writerow(data)
                
                #print a pretty table
                if negloglike_sorted[n] != 0 and n<200: # and divergence:
                    ptab.add_row([n, functions_sorted[n], '%.2f'%L_sorted[n], '%.2f'%negloglike_sorted[n], '%.2f'%codelen_sorted[n], '%.2e'%aifeyn_sorted[n], '%.2e'%mean_p0_sorted[n], '%.2e'%std_p0_sorted[n], '%.2e'%mean_p1_sorted[n], '%.2e'%std_p1_sorted[n], divergence_flag])
        print(ptab)

        with open(dirname + save_dir + 'pretty_all_clusters_comp' + str(i) + '.txt', 'w') as f:
            print('Saving to' + dirname + save_dir + 'pretty_all_clusters_comp' + str(i) + '.txt')
            f.write(str(ptab))

        #plot params
        for index, function in enumerate(functions_sorted[:5]):
            print(index, function)
            sanitized_function_name = sanitize_filename(function)
            k = simplifier.count_params([function], 4)[0]
            if k == 0:
                continue
            else:
                plot_params(sorted_p0[index, :], sorted_p1[index, :], k)
                #save  the plot 
                plt.title(function)   
                plt.savefig(dirname + save_dir + 'plots/comp' + str(i) + '/' + sanitized_function_name + '.png', dpi=300)
                plt.close()
             
        # save both params and delta to a file
        save_params_with_pickle(dirname + save_dir + 'params_comp' + str(i) + '.pkl', sorted_p0, sorted_p1, sorted_p2, sorted_p3, sorted_d0, sorted_d1, sorted_d2, sorted_d3)

        # save the best 200 functions to a file


        
