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



# @profile
def convert_to_unique_params(all_eqs, all_inv_subs, fcn, index, p, delta):
        # Find codelen matches for that function to unique equations
        max_param = 4
        #take out \n form all_eqs
        all_eqs = [eq.replace('\n', '').strip() for eq in all_eqs]

        #check if delta is empty
        if np.all(delta == 0):
            # delta = np.ones(p.shape)
            delta = p.copy()

        # Find the index of the equation in all_eqs
        p_uniques = np.zeros(p.shape)
        delta_uniques = np.zeros(delta.shape)
        for i, eq in enumerate(fcn):
            # print(i)
            # print('eq', eq)
            # print('unique_fcn', unique_fcn[i]) 
            p_gal = p[i]
            # print('p_gal', p_gal)
            delta_gal = delta[i]
            # print('delta_gal', delta_gal)
            eq = eq.replace('\n', '')
            # eq = '1/(a0*x**2)'
            # print('eq', eq)
            try:
                idx = all_eqs.index(eq)
                # idx = int(index[i])
                # idx = indices[i]
                # print(all_eqs[idx])
                # match = matches[idx]
                # print('match', match)
                # inv_subs = all_inv_subs[match]
                inv_subs = all_inv_subs[idx]
                # print(all_inv_subs[idx])

                # p_gal = [0.03971062, 0., 0., 0.]
                # delta_gal =  [0.01525985, 0., 0., 0.]
                # inv_subs = [{a0: 1/a0}]
                p_unique, delta_unique = inverse_convert_params(p_gal, delta_gal, inv_subs)

                # print('p_unique', p_unique)
                # print(i)

                # if i == 289:
                #     print(eq)
                #     print(inv_subs)
                #     print(p_gal, p_unique)
                #     print(delta_gal, delta_unique)


                if np.any(np.isinf(p_unique)):
                    p_unique[p_unique == np.inf] = p_gal[p_unique == np.inf]
                    delta_unique[p_unique == np.inf] = delta_gal[p_unique == np.inf]

            except ValueError as e:
                # print(i, 'ValueError', e)
                # print(eq)
                p_unique = p_gal
                delta_unique = delta_gal

            # print('p_unqiue',p_unique)
            # print('p_gal',p_gal)
            # sys.exit()
            p_uniques[i] = p_unique
            delta_uniques[i] = delta_unique
        return p_uniques, delta_uniques


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


def check_divergences(param0, param1, param2, param3, fcn):
    # num_params = simplifier.count_params([eq], 4)[0]
    # print(fcn)

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
        try:
            params = [param0[i], param1[i], param2[i], param3[i]]

            a_symbols = [a0, a1, a2, a3]
            substitutions = {a_symbols[i]: params[i] for i in range(len(params))}
            eq_substituted = eq.subs(substitutions)
            # print(params)

            # singularities = sympy.singularities(eq_substituted, x, domain=sympy.Interval(0, sympy.oo))

            domain=sympy.Interval.open(0, sympy.oo)
            continuous = sympy.calculus.util.continuous_domain(eq_substituted, x, domain=domain)


            if continuous == domain:
                # print("The equation is continuous over the entire domain.")
                is_finite = True
            else:
                # Points or regions missing from the continuous domain
                divergence_points = domain - continuous
                print(params)
                print(f"Divergences found at: {divergence_points}", continuous)
                return False
                # is_finite = False
                # return divergence_points

        # if isinstance(singularities, sympy.ConditionSet):
        #     has_solutions = True
        # else:
        #     has_solutions = len(singularities) > 0

        # is_finite = not has_solutions

        # is_finite
        # print(is_finite)
        # is_finite = eq_substituted.is_finite
        except Exception as e:
            print(fcn)
            print(e)
            return True
    return is_finite

def calculate_mass(likelihood, eq_numpy, params, delta):
     M200, r200, error_M200 = likelihood.M_delta(200, eq_numpy, params, delta)

     return M200, r200, error_M200
        

def main(dirname, fcn_dir, save_dir):

    # save_dir = 'all_clusters/'
    max_params = 4

    with open('all_clusters.txt') as f:
        names = f.read().splitlines() 

    for i in range(10,  11):
        print('COMP', i)
        negloglike = None
        codelen = None
        p0, p1, p2, p3 = None, None, None, None

            
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
        negloglike_test = 0

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

            print(fcn_variant[13])
            print('DL', data[13, 0])
            print('negloglike', data[13, -6])
            print('codelen', data[13, -5])
            negloglike_test += data[13, -6]
            # sys.exit()

            data_file = 'XXL/' + str(name) + '.txt'
            run_name = 'WL_' + str(name)
            likelihood = WLLikelihood(data_file, run_name)

            # Define p_actual using the parameters from data
            # p_actual = np.array([data[:, 1], data[:, 2], data[:, 3], data[:, 4]]).T
            p_actual = data[:, 1:1+ max_params]
            delta_actual = data[:, 1+max_params: -6]

            # for k in range(len(p_actual)):  # Assuming the lists have the same length
            #     for kk in range(len(p_actual[k])):
            #         if p_actual[k][kk] != 0 and delta_actual[k][kk] == 0:
            #             print('hola')
            #             print(name)
            #             print(fcn_variant[k])

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
    
            p_unique, delta_unique = convert_to_unique_params(all_eqs, all_invs_subs, fcn_variant, indices_variant, p_actual, delta_actual)
            # print('p_unique', p_unique)
            # print('delta_unique', delta_unique)
            # p_unique = p_actual
            # delta_unique = delta_actual

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

            # Check and count infinities
            is_negloglike_inf = negloglike_value == np.inf
            is_codelen_inf = codelen_value == np.inf

            for idx, (neg_inf, code_inf) in enumerate(zip(is_negloglike_inf, is_codelen_inf)):
                if neg_inf or code_inf:
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
        for j in range(len(fcn)):
            if count_zeroes[j] != 0 and count_zeroes[j] != len(names):
                print(count_zeroes[j], fcn[j], '\n', clusters_with_inf_per_fcn[j])

        # Process your data and calculate L, then save to a CSV file
        mean_p0 = np.mean(p0, axis=1)
        mean_p1 = np.mean(p1, axis=1)
        std_p0 = np.std(p0, axis=1)
        std_p1 = np.std(p1, axis=1)

        #save all params
        print('AQUI', negloglike_test)
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

        sorted_p0 = p0[sorted_indices, :]
        sorted_p1 = p1[sorted_indices, :]
        sorted_p2 = p2[sorted_indices, :]
        sorted_p3 = p3[sorted_indices, :]

        sorted_d0 = d0[sorted_indices, :]
        sorted_d1 = d1[sorted_indices, :]
        sorted_d2 = d2[sorted_indices, :]
        sorted_d3 = d3[sorted_indices, :]

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
            for n in range(min(400, len(sorted_data))):
                no_divergence = check_divergences(sorted_p0[n, :], sorted_p1[n, :], sorted_p2[n, :], sorted_p3[n, :], functions_sorted[n])
                divergence_flag = "---" if no_divergence else "YES"
                if negloglike_sorted[n] != 0: # and divergence:
                    data = [n, functions_sorted[n], L_sorted[n], negloglike_sorted[n], negloglike_sorted[n], codelen_sorted[n], aifeyn_sorted[n], mean_p0_sorted[n], std_p0_sorted[n], mean_p1_sorted[n], std_p1_sorted[n], divergence_flag]

                    # writer.writerow(data)

                    # ptab.add_row([n, function, '%.2f'%l_d, '%.2f'%negloglike_value, '%.2f'%codelen_value, '%.2e'%aifeyn_value, '%.2e'%mean_a0, '%.2e'%std_a0, '%.2e'%mean_a1, '%.2e'%std_a1])
                    ptab.add_row([n, functions_sorted[n], '%.2f'%L_sorted[n], '%.2f'%negloglike_sorted[n], '%.2f'%codelen_sorted[n], '%.2e'%aifeyn_sorted[n], '%.2e'%mean_p0_sorted[n], '%.2e'%std_p0_sorted[n], '%.2e'%mean_p1_sorted[n], '%.2e'%std_p1_sorted[n], divergence_flag])
                # elif not divergence:
                #     print('Divergences found for function', functions_sorted[n])
                #     continue
        print(ptab)

        # with open(dirname + save_dir + 'pretty_all_clusters_comp' + str(i) + '.txt', 'w') as f:
        #     print('Saving to' + dirname + save_dir + 'pretty_all_clusters_comp' + str(i) + '.txt')
        #     f.write(str(ptab))

        #save params and plot params
        # for index, function in enumerate(functions_sorted[:20]):
        #     print(index, function)
        #     sanitized_function_name = sanitize_filename(function)
        #     k = simplifier.count_params([function], 4)[0]
        #     if k == 0:
        #         continue
        #     else:
        #         plot_params(sorted_a0[index, :], sorted_a1[index, :], k)
        #         #save  the plot 
        #         plt.title(function)   
        #         plt.savefig(dirname + save_dir + 'compl_' + str(i) + '/' + sanitized_function_name + '.png', dpi=300)
        #         plt.close()
             
        # save both params and delta to a file  
        #check that for p different from 0, delta is not 0
            
        # for i in range(len(sorted_p0)):  # Assuming the lists have the same length
        #     for j in range(len(sorted_p0[i])):
        #         if sorted_p0[i][j] != 0 and sorted_d0[i][j] == 0:
        #             print(f"Warning: sorted_p0[{i}][{j}] is non-zero but sorted_d0[{i}][{j}] is zero!")
                
        #         if sorted_p1[i][j] != 0 and sorted_d1[i][j] == 0:
        #             print(f"Warning: sorted_p1[{i}][{j}] is non-zero but sorted_d1[{i}][{j}] is zero!")
        

        # save_params_with_pickle(dirname + save_dir + 'params_comp' + str(i) + '.pkl', sorted_p0, sorted_p1, sorted_p2, sorted_p3, sorted_d0, sorted_d1, sorted_d2, sorted_d3)

        # save the best 200 functions to a file
        if save_dir == '12_clusters/':
            with open(dirname + '12_clusters/best_200_comp' + str(i) + '.txt', 'w') as f:
                # for n, (function, l_d, negloglike_value, codelen_value, aifeyn_value, mean_a0, std_a0, mean_a1, std_a1) in enumerate(sorted_data[:500]):
                for n in range((min(200, len(sorted_data)))):
                    f.write(functions_sorted[n] + '\n')


        
        #caculate the mass and concentration for the best fucn
        # best_func = functions_sorted[0]
        # fcn_i, eq= likelihood.run_sympify(best_func)

        # k = simplifier.count_params([fcn_i], 4)[0]
        # if k == 0:
        #     eq_numpy = sympy.lambdify([x], eq, modules=["jax"])
        # elif k > 1:
        #     all_a = ' '.join([f'a{i}' for i in range(k)])
        #     all_a = list(sympy.symbols(all_a, real=True))
        #     eq_numpy = sympy.lambdify([x] + all_a, eq, modules=["jax"])
        # else:
        #     eq_numpy = sympy.lambdify([x, a0], eq, modules=["jax"])
        #     print('eq_numpy', eq_numpy)
        
        # best_params = sorted_p0[0, :]
        # best_delta = sorted_d0[0, :]

        # masses = []
        # radii = []
        # errors = []
        # for index, name in enumerate(names):
        #     data_file = 'XXL/' + str(name) + '.txt'
        #     run_name = 'WL_' + str(name)
        #     likelihood = WLLikelihood(data_file, run_name)
        #     M200, r200, error_M200 = calculate_mass(likelihood, eq_numpy, np.array([best_params[index]]), np.array([best_delta[index]]))

        
