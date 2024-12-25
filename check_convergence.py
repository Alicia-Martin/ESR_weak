import numpy as np
import sys

comp = 8

with open('10_clusters.txt') as f:
        names = f.read().splitlines() 

for i, name in enumerate(names):
    print(i, name)

    #Read the function
    with open('esr/fitting/output_decreasing/output/output_WL_' + name + '/combine_DL_fcn_comp' + str(comp) + '.dat', "r") as f:         # All
        fcn_list = f.read().splitlines()

    # Read data from last file
    data = np.genfromtxt('esr/fitting//output_decreasing/output/output_WL_' + name + '/combine_DL_comp' + str(comp) + '.dat')

    DL = data[:,0]
    negloglike = data[:,-6]
    codelen = data[:,-5]
    aifeyn = data[:,-4]
    Nconv = data[:,-3]
    Niter = data[:,-2]
    time_taken = data[:,-1]

    fcn_list = np.array(fcn_list)

    #with alpha
    vmin = 1e-5
    N_convergence = 5
    DL_min = np.amin(DL[np.isfinite(DL)])
    alpha = DL_min - DL
    alpha = np.exp(alpha)
    m = np.array(alpha > vmin)
    converged = Nconv > N_convergence
    not_converged = (Nconv <= N_convergence) & (Niter !=0)

    # print_fucntions with high probability and low convergence
    # print(name)
    # # print(alpha[m])
    # # print(Nconv[m])
    # for i in range(len(fcn_list[not_converged & m])):
    #      print(fcn_list[not_converged & m][i])
    #      print('alpha:', alpha[not_converged & m][i])
    #      print('Nconv:', Nconv[not_converged & m][i])
    # print(fcn_list[not_converged & m])
    # sys.exit()

    #with negloglike
    #do you want to check the DL or the negloglike?
    # dif_min = 20
    # negloglike_min = np.amin(negloglike[np.isfinite(DL)])

    # alpha = negloglike - negloglike_min
    # m = (alpha < dif_min) & (alpha > 0)

    # for i in range(len(fcn_list[not_converged & m])):
    #      print(fcn_list[not_converged & m][i])
    #      print('alpha:', alpha[not_converged & m][i])
    #      print('Nconv:', Nconv[not_converged & m][i])

    print(fcn_list[not_converged & m])

    # fcn_list = [d for i, d in enumerate(fcn_list) if m[i]]
    # alpha = alpha[m]
    # Nconv = Nconv[m]

    # print(len(fcn_list))
    # print('fcn_list:', fcn_list)
    # # print('params:', params)
    # print('alpha:', alpha)
    # print('Nconv:', Nconv)