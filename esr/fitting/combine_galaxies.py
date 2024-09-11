import numpy as np
import csv
import sys
import esr.plotting.plot
from mpi4py import MPI
from prettytable import PrettyTable
from esr.fitting.WL_likelihood import WLLikelihood


def main(comp, likelihood):

    with open('10_clusters.txt') as f:
        names = f.read().splitlines() 

    for i in range(1, comp):
        with open(likelihood.fn_dir + "/compl_%i/unique_equations_%i.txt"%(comp,comp)) as f:
            fcn = f.read().splitlines()

        negloglike = None
        codelen = None
        a0, a1, a2, a3 = None, None, None, None
        not_conv_count = 0
        for name in names:
            data_file = 'XXL/' + name + '.pickle'
            run_name = 'WL_' + name
            likelihood = WLLikelihood(data_file, run_name, data_dir=None, fn_set = 'core_maths')   
            data = np.genfromtxt(likelihood.out_dir + '/combine_DL_comp' + str(i) + '.dat') 
            negloglike_value = data[:,-3]
            codelen_value = data[:,-2]
            aifeyn = data[:,-1]
            param0 = data[:,1].reshape(-1,1)
            param1 = data[:,2].reshape(-1,1)
            param2 = data[:,3].reshape(-1,1)
            param3 = data[:,4].reshape(-1,1)
            Nconv = data[:,-3].reshape(-1,1)
            Niter = data[:,-2].reshape(-1,1)
            times = data[:,-1].reshape(-1,1)
            comp = np.array([i]*len(negloglike_value)).reshape(-1,1)

            if Nconv == 0:
                not_conv_count += 1

            if negloglike is None:
                negloglike = negloglike_value
                codelen = codelen_value
                a0, a1, a2, a3 = param0, param1, param2, param3
                Nconv, Niter, times = Nconv, Niter, times

            else:
                negloglike += negloglike_value
                codelen += codelen_value
                Nconv += Nconv
                Niter += Niter
                times += times

                a0 = np.concatenate((a0, param0), axis=1)
                a1 = np.concatenate((a1, param1), axis=1)
                a2 = np.concatenate((a2, param2), axis=1)
                a3 = np.concatenate((a3, param3), axis=1)

        # Process your data and calculate L, then save to a CSV file
        L = [neg + code + ai for neg, code, ai in zip(negloglike, codelen, aifeyn)]
        sorted_data = sorted(zip(fcn, L, negloglike, codelen, aifeyn, comp, Nconv, Niter, times, not_conv_count), key=lambda x: x[1])
        ptab = PrettyTable()
        ptab.field_names = ["Function", "L(D)", "-logL", "Codelen", "AIFeyn", "comp", "Nconv", "Niter", "Time", "not_conv_count"]

        # for n in range(len(negloglike)):
        # for n, (fcn, L, negloglike, codelen, aifeyn_value) in enumerate(sorted_data):
        #     print(function[n])
        #     if negloglike[n] != 0:
        #         data = [n, fcn[n], L[n], n, negloglike[n], codelen[n], aifeyn[n]]
        #         data[1] = np.char.rstrip(data[1], '\n')
        #         with open('fitting/output/local_params/final_' + str(i) + '.dat', 'a') as f:
        #             writer = csv.writer(f, delimiter=';')
        #             writer.writerow(data)

        #         ptab.add_row([fcn[n], '%.2f'%L[n], '%.2f'%negloglike[n], '%.2f'%codelen[n], '%.2e'%aifeyn[n]])

        # print(ptab)

        for n, (function, l_d, negloglike_value, codelen_value, aifeyn_value, comp_value, Nconv_value, Niter_value, times_value, not_conv_value) in enumerate(sorted_data):
            if negloglike_value != 0:
                data = [n, function, l_d, n, negloglike_value, codelen_value, aifeyn_value, comp_value, Nconv_value, Niter_value, times_value, not_conv_value]
                data[1] = np.char.rstrip(data[1], '\n')

                with open('fitting/output/combine_final_' + str(i) + '.dat', 'a') as f:
                    writer = csv.writer(f, delimiter=';')
                    writer.writerow(data)

                ptab.add_row([function, '%.2f'%l_d, '%.2f'%negloglike_value, '%.2f'%codelen_value, '%.2e'%aifeyn_value, comp_value])

    print(ptab)