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
    # all_invs_subs = None
    with open(fcn_dir + "/compl_%i/all_equations_%i.txt" % (comp, comp)) as f:
        all_eqs = f.read().splitlines()
    return fcn, all_invs_subs, all_eqs

def main(dirname, fcn_dir, save_dir):

    # save_dir = 'all_clusters/'
    max_params = 4

    with open('10_clusters.txt') as f:
        names = f.read().splitlines() 

    for i in range(6,  7):
        print('COMP', i)
        negloglike = None
        codelen = None
        p0, p1, p2, p3 = None, None, None, None

        fcn_dir = 'esr/function_library/core_maths/'

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
                # data = np.genfromtxt(dirname + 'output_WL_' + name +  '_all_fun/combine_DL_comp' + str(i) + '.dat') 
            except:
                data = np.genfromtxt(dirname + 'output_WL_' + name +  '_all_fun/combine_DL_comp' + str(i) + '.dat') 
            # indices_variant = np.genfromtxt(dirname + 'output_WL_' + name +  '/combine_DL_index_variants_comp' + str(i) + '.dat')
            indices_variant = None

            # Read the equations file
            with open(dirname + 'output_WL_' + name +  '/combine_DL_fcn_comp' + str(i) + '.dat', 'r') as file:
                fcn_variant = file.readlines()

            # print(fcn_variant[224])
            # sys.exit()

            data_file = 'XXL/' + str(name) + '.txt'
            run_name = 'WL_' + str(name)
            likelihood = WLLikelihood(data_file, run_name)

            # Define p_actual using the parameters from data
            # p_actual = np.array([data[:, 1], data[:, 2], data[:, 3], data[:, 4]]).T
            p_actual = data[:, 1:1+ max_params]
            delta_actual = data[:, 1+max_params: -6]

            for k in range(len(p_actual)):  # Assuming the lists have the same length
                for kk in range(len(p_actual[k])):
                    if p_actual[k][kk] != 0 and delta_actual[k][kk] == 0:
                        print('hola')
                        print(name)
                        print(fcn_variant[k])

            L_value = data[:,0]
            # print(name, L_value[289])
            # if np.isinf(L_value[289]):
            #     # print('negloglike', data[289, -6])
            #     # print('codelen', data[289, -5])
            #     print(name)

            # for a in range(len(p_actual)):
            #     for b in range(len(p_actual[i])):
            #         if delta_actual[a][b] == 0 and p_actual[a][b] != 0:
            #             print('p_actual', p_actual[a])
            #             print('delta_actual', delta_actual[a])
            #             print('unique_fcn', fcn[a])
                # print('p_actual', p_actual[i])
                # print('delta_actual', delta_actual[i])

            # sys.exit()
            # delta_actual = p_actual
    
            # p_unique, delta_unique = convert_to_unique_params(all_eqs, all_invs_subs, fcn_variant, indices_variant, p_actual, delta_actual)
            # print('p_unique', p_unique)
            # print('delta_unique', delta_unique)
            p_unique = p_actual
            delta_unique = delta_actual

            # for k in range(len(p_unique)):  # Assuming the lists have the same length
            #     for kk in range(len(p_unique[k])):
            #         if p_unique[k][kk] != 0 and delta_unique[k][kk] == 0:
            #             print('hola')
                    
                    # if p1[i][j] != 0 and d1[i][j] == 0:
                    #     print('hola')
            # sys.exit()

            negloglike_value = data[:, -6]
            negloglike_value[negloglike_value == 0] = np.inf
            aifeyn = data[:, -4]
            L = data[:, 0]

            # print(fcn_variant)

            codelen_value = data[:, -5]
            param0 = p_unique[:, 0].reshape(-1, 1)
            param1 = p_unique[:, 1].reshape(-1, 1)
            param2 = p_unique[:, 2].reshape(-1, 1)
            param3 = p_unique[:, 3].reshape(-1, 1)

            delta0 = delta_unique[:, 0].reshape(-1, 1)
            delta1 = delta_unique[:, 1].reshape(-1, 1)
            delta2 = delta_unique[:, 2].reshape(-1, 1)
            delta3 = delta_unique[:, 3].reshape(-1, 1)

            for idx, is_inf in enumerate(L == np.inf):
                if is_inf:
                    clusters_with_inf_per_fcn[idx].append(name)

            count_zeroes += (L == np.inf)

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
        # for j in range(len(fcn)):
        #     if count_zeroes[j] != 0 and count_zeroes[j] != len(names):
        #         # print(count_zeroes[j], fcn[j], '\n', clusters_with_inf_per_fcn[j])

        # Process your data and calculate L, then save to a CSV file
        mean_p0 = np.mean(p0, axis=1)
        mean_p1 = np.mean(p1, axis=1)
        std_p0 = np.std(p0, axis=1)
        std_p1 = np.std(p1, axis=1)

        #save all params

        L = [neg + code + ai for neg, code, ai in zip(negloglike, codelen, aifeyn)]
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

        ptab = PrettyTable()
        ptab.field_names = ["#", "Function", "L(D)", "-logL", "Codelen", "AIFeyn", "Mean a0", "Std a0", "Mean a1", "Std a1", "Divergence"]

        # for i in range(len(sorted_p0)):  # Assuming the lists have the same length
        #     for j in range(len(sorted_p0[i])):
        #         if sorted_p0[i][j] != 0 and sorted_d0[i][j] == 0:
        #             print('hola')
                
        #         if sorted_p1[i][j] != 0 and sorted_d1[i][j] == 0:
        #             print('hola')

        #SAVE THE DATA TO A FILE
        with open(dirname +  save_dir + 'final_' + str(i) + '.dat', 'w') as f:
            writer = csv.writer(f, delimiter=';')
            for n in range(min(200, len(sorted_data))):
                divergence_flag = "---"
                if negloglike_sorted[n] != 0: # and divergence:
                    data = [n, functions_sorted[n], L_sorted[n], negloglike_sorted[n], negloglike_sorted[n], codelen_sorted[n], aifeyn_sorted[n], mean_p0_sorted[n], std_p0_sorted[n], mean_p1_sorted[n], std_p1_sorted[n], divergence_flag]

                    writer.writerow(data)

                    # ptab.add_row([n, function, '%.2f'%l_d, '%.2f'%negloglike_value, '%.2f'%codelen_value, '%.2e'%aifeyn_value, '%.2e'%mean_a0, '%.2e'%std_a0, '%.2e'%mean_a1, '%.2e'%std_a1])
                    ptab.add_row([n, functions_sorted[n], '%.2f'%L_sorted[n], '%.2f'%negloglike_sorted[n], '%.2f'%codelen_sorted[n], '%.2e'%aifeyn_sorted[n], '%.2e'%mean_p0_sorted[n], '%.2e'%std_p0_sorted[n], '%.2e'%mean_p1_sorted[n], '%.2e'%std_p1_sorted[n], divergence_flag])
                # elif not divergence:
                #     print('Divergences found for function', functions_sorted[n])
                #     continue
        print(ptab)

        with open(dirname + save_dir + 'pretty_all_clusters_comp' + str(i) + '.txt', 'w') as f:
            print('Saving to' + dirname + save_dir + 'pretty_all_clusters_comp' + str(i) + '.txt')
            f.write(str(ptab))

        # save the best 200 functions to a file
        if save_dir == '12_clusters/':
            with open(dirname + '12_clusters/best_200_comp' + str(i) + '.txt', 'w') as f:
                # for n, (function, l_d, negloglike_value, codelen_value, aifeyn_value, mean_a0, std_a0, mean_a1, std_a1) in enumerate(sorted_data[:500]):
                for n in range((min(200, len(sorted_data)))):
                    f.write(functions_sorted[n] + '\n')