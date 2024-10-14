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

# @profile
def convert_to_unique_params(all_eqs, all_inv_subs, fcn, index, p):
        # Find codelen matches for that function to unique equations
        max_param = 4
        #take out \n form all_eqs
        all_eqs = [eq.replace('\n', '').strip() for eq in all_eqs]

        # Find the index of the equation in all_eqs
        p_uniques = np.zeros(p.shape)
        for i, eq in enumerate(fcn):
            # print('eq', eq)
            # print('unique_fcn', unique_fcn[i]) 
            p_gal = p[i]
            # print(p_gal)
            eq = eq.replace('\n', '')
            try:
                idx = all_eqs.index(eq)
                # idx = int(index[i])
                # idx = indices[i]
                # print(all_eqs[idx])
                # match = matches[idx]
                # print('match', match)
                # inv_subs = all_inv_subs[match]
                inv_subs = all_inv_subs[idx]
                # print(inv_subs)
                # print(all_inv_subs[idx])
                # print(inv_subs)
                p_unique = inverse_convert_params(p_gal, inv_subs)


                if np.any(np.isinf(p_unique)):
                    p_unique[p_unique == np.inf] = p_gal[p_unique == np.inf]

            except ValueError as e:
                # print(eq)
                p_unique = p_gal

            # print('p_unqiue',p_unique)
            # print('p_gal',p_gal)

            p_uniques[i] = p_unique

        return p_uniques


def load_equations_and_matches(fcn_dir, comp):
    with open(fcn_dir + "/compl_%i/unique_equations_%i.txt" % (comp, comp)) as f:
        fcn = f.read().splitlines()
    invsubs_file = fcn_dir + "/compl_%i/inv_subs_%i.txt" % (comp, comp)
    max_param = 4
    all_invs_subs = simplifier.load_subs(invsubs_file, max_param)
    # all_invs_subs = None
    with open(fcn_dir + "/compl_%i/all_equations_%i.txt" % (comp, comp)) as f:
        all_eqs = f.read().splitlines()
    return fcn, all_invs_subs, all_eqs

def save_params_with_pickle(filepath, a0, a1, a2, a3):
    params = {
        "a0": a0,
        "a1": a1,
        "a2": a2,
        "a3": a3
    }
    
    with open(filepath, 'wb') as file:
        pickle.dump(params, file)

def plot_params(a0, a1, k):
    if k==1:
        plt.plot(a0, marker='o', label='a0')  # Plot a0 if available
        plt.xlabel('galaxy cluster')
        plt.ylabel('a0')

    elif k==2:
        plt.scatter(a0, a1, marker='o', label='a1 vs a0')
        plt.xlabel('a0')
        plt.ylabel('a1')

    elif k==0:
        print('No parameters for this function')

def sanitize_filename(name):
    # Replace any invalid characters with an underscore or another safe character
    return re.sub(r'[<>:"/\\|?*]', '_', name)
        

def main(dirname, fcn_dir, save_dir):

    # save_dir == '12_clusters/'

    with open('all_clusters.txt') as f:
        names = f.read().splitlines() 

    for i in range(9, 10):
        print('COMP', i)
        negloglike = None
        codelen = None
        a0, a1, a2, a3 = None, None, None, None

            
        if i in [1, 2, 3, 4, 5, 6]:
            fcn_dir = 'esr/function_library/core_maths/'

        else:
            fcn_dir = 'esr/function_library/best_funcs'

        # fcn_dir = 'esr/function_library/core_maths/'

        with open(fcn_dir + "/compl_%i/unique_equations_%i.txt"%(i,i)) as f:
            fcn = f.read().splitlines()
        # print(len(fcn))
        fcn, all_invs_subs, all_eqs = load_equations_and_matches(fcn_dir, i)
        count_zeroes = np.zeros(len(fcn))

        clusters_with_inf_per_fcn = [[] for _ in range(len(fcn))]
        # for name in ['76', '91', '6', '44', '151', '152', '111', '114', '27', '52', '55', '88']:

        for index, name in enumerate(names):
            try:
                # data = np.genfromtxt(dirname + 'output_WL_' + name +  '_all_fun/combine_DL_comp' + str(i) + '.dat') 
                data = np.genfromtxt(dirname + 'output_WL_' + name +  '/combine_DL_comp' + str(i) + '.dat') 
            except:
                data = np.genfromtxt(dirname + 'output_WL_' + name +  '_all_fun/combine_DL_comp' + str(i) + '.dat') 
            # indices_variant = np.genfromtxt(dirname + 'output_WL_' + name +  '/combine_DL_index_variants_comp' + str(i) + '.dat')
            indices_variant = None

            # Read the equations file
            with open(dirname + 'output_WL_' + name +  '/combine_DL_fcn_comp' + str(i) + '.dat', 'r') as file:
                fcn_variant = file.readlines()

            data_file = 'XXL/' + str(name) + '.txt'
            run_name = 'WL_' + str(name)
            likelihood = WLLikelihood(data_file, run_name)

            # Define p_actual using the parameters from data
            p_actual = np.array([data[:, 1], data[:, 2], data[:, 3], data[:, 4]]).T
    
            p_unique = convert_to_unique_params(all_eqs, all_invs_subs, fcn_variant, indices_variant, p_actual)
            # p_unique = p_actual

            negloglike_value = data[:, -6]
            negloglike_value[negloglike_value == 0] = np.inf
            aifeyn = data[:, -4]

            # print(fcn_variant)

            codelen_value = data[:, -5]
            param0 = p_unique[:, 0].reshape(-1, 1)
            param1 = p_unique[:, 1].reshape(-1, 1)
            param2 = p_unique[:, 2].reshape(-1, 1)
            param3 = p_unique[:, 3].reshape(-1, 1)

            for idx, is_inf in enumerate(negloglike_value == np.inf):
                if is_inf:
                    clusters_with_inf_per_fcn[idx].append(name)

            count_zeroes += (negloglike_value == np.inf)

            if negloglike is None:
                negloglike = negloglike_value
                codelen = codelen_value
                a0, a1, a2, a3 = param0, param1, param2, param3
                # Nconv, Niter, times = Nconv, Niter, times

            else:
                # print(negloglike_value)
                negloglike += negloglike_value
                codelen += codelen_value
                # Nconv += Nconv
                # Niter += Niter
                # times += times

                a0 = np.concatenate((a0, param0), axis=1)
                a1 = np.concatenate((a1, param1), axis=1)
                a2 = np.concatenate((a2, param2), axis=1)
                a3 = np.concatenate((a3, param3), axis=1)

        # Count zeroes                
        # for j in range(len(fcn)):
        #     if count_zeroes[j] != 0 and count_zeroes[j] != len(names):
        #         print(count_zeroes[j], fcn[j], '\n', clusters_with_inf_per_fcn[j])

        # Process your data and calculate L, then save to a CSV file
        mean_a0 = np.mean(a0, axis=1)
        mean_a1 = np.mean(a1, axis=1)
        std_a0 = np.std(a0, axis=1)
        std_a1 = np.std(a1, axis=1)

        #save all params

        L = [neg + code + ai for neg, code, ai in zip(negloglike, codelen, aifeyn)]
        sorted_data = sorted(zip(fcn, L, negloglike, codelen, aifeyn, mean_a0, std_a0, mean_a1, std_a1), key=lambda x: x[1])
        #sort params by L too
        sorted_indices = np.argsort(L)
        functions_sorted = [fcn[i] for i in sorted_indices]
        sorted_a0 = a0[sorted_indices, :]
        sorted_a1 = a1[sorted_indices, :]
        sorted_a2 = a2[sorted_indices, :]
        sorted_a3 = a3[sorted_indices, :]

        ptab = PrettyTable()
        ptab.field_names = ["#", "Function", "L(D)", "-logL", "Codelen", "AIFeyn", "Mean a0", "Std a0", "Mean a1", "Std a1"]


        #SAVE THE DATA TO A FILE
        with open(dirname +  save_dir + 'final_' + str(i) + '.dat', 'w') as f:
            writer = csv.writer(f, delimiter=';')
            for n, (function, l_d, negloglike_value, codelen_value, aifeyn_value, mean_a0, std_a0, mean_a1, std_a1) in enumerate(sorted_data):
                if negloglike_value != 0:
                    data = [n, function, l_d, n, negloglike_value, codelen_value, aifeyn_value]

                    writer.writerow(data)

                    ptab.add_row([n, function, '%.2f'%l_d, '%.2f'%negloglike_value, '%.2f'%codelen_value, '%.2e'%aifeyn_value, '%.2e'%mean_a0, '%.2e'%std_a0, '%.2e'%mean_a1, '%.2e'%std_a1])
                
        print(ptab)

        with open(dirname + save_dir + 'pretty_all_clusters_comp' + str(i) + '.txt', 'w') as f:
            f.write(str(ptab))

        #save params and plot params
        for index, function in enumerate(functions_sorted[:20]):
            print(index, function)
            sanitized_function_name = sanitize_filename(function)
            k = simplifier.count_params([function], 4)[0]
            if k == 0:
                continue
            else:
                plot_params(sorted_a0[index, :], sorted_a1[index, :], k)
                #save  the plot 
                plt.title(function)   
                plt.savefig(dirname + save_dir + 'compl_' + str(i) + '/' + sanitized_function_name + '.png', dpi=300)
                plt.close()
                
        save_params_with_pickle(dirname + save_dir + 'params_comp' + str(i) + '.pkl', sorted_a0, sorted_a1, sorted_a2, sorted_a3)


        # save the best 200 functions to a file
        if save_dir == '12_clusters/':
            with open(dirname + '12_clusters/best_200_comp' + str(i) + '.txt', 'w') as f:
                for n, (function, l_d, negloglike_value, codelen_value, aifeyn_value, mean_a0, std_a0, mean_a1, std_a1) in enumerate(sorted_data[:200]):
                    f.write(function + '\n')

        
