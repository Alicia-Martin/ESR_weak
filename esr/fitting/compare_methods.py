import numpy as np
import pandas as pd
import os
from mpi4py import MPI
import sys
import matplotlib.pyplot as plt


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def get_functions(comp, likelihood, unique=True):
    """Load all functions for a given complexity to use and distribute among ranks
    
    Args:
        :comp (int): complexity of functions to consider
        :likelihood (fitting.likelihood object): object containing data, functions to convert SR expressions to variable of data and file path
        :unique (bool, default=True): whether to load just the unique functions (True) or all functions (False)
        
    Returns:
        :fcn_list (list): list of strings representing functions to be used by given rank
        :data_start (int): first index of function used by rank
        :data_end (int): last index of function used by rank
        
    """

    if unique:
        unifn_file = likelihood.fn_dir + "/compl_%i/unique_equations_%i.txt"%(comp,comp)
    else:
        unifn_file = likelihood.fn_dir + "/compl_%i/all_equations_%i.txt"%(comp,comp)
    
    if comp>=8:
        sys.setrecursionlimit(2000 + 500 * (comp - 8))

    if rank == 0:
        for dirname in [likelihood.base_out_dir, likelihood.out_dir, likelihood.temp_dir]:
            if not os.path.isdir(dirname):
                print('Making dir:', dirname)
                os.mkdir(dirname)
    comm.Barrier()

    if rank==0:
        print("Number of cores:", size, flush=True)

    with open(unifn_file, "r") as f:
        fcn_list = f.readlines()

    nLs = int(np.ceil(len(fcn_list) / float(size)))       # Number of lines per file for given thread

    while nLs*(size-1) > len(fcn_list):
        if rank==0:
            print("Correcting for many cores.", flush=True)
        nLs -= 1

    if rank==0:
        print("Total number of functions: ", len(fcn_list), flush=True)
        print("Number of test points per proc: ", nLs, flush=True)

    data_start = rank*nLs
    data_end = (rank+1)*nLs

    if rank==size-1:
        data_end = len(fcn_list)
    
    return fcn_list[data_start:data_end], data_start, data_end

def create_comparison_table(comp, likelihood):

    fcn_list, _, _ = get_functions(comp, likelihood)
    # Define file paths
    bfgs_file = os.path.join(likelihood.out_dir, f'negloglike_comp{comp}_BFGS.dat')
    nelder_mead_file = os.path.join(likelihood.out_dir, f'negloglike_comp{comp}_Nelder-Mead.dat')
    
    # Read the data from the files
    bfgs_data = np.loadtxt(bfgs_file)
    nelder_mead_data = np.loadtxt(nelder_mead_file)
    
    # Extract negloglike and N_conv columns (assuming negloglike is the first column and N_conv is the last)
    bfgs_negloglike = bfgs_data[:, 0]
    bfgs_nconv = bfgs_data[:, -3]  # N_conv is the third last column
    
    nelder_mead_negloglike = nelder_mead_data[:, 0]
    nelder_mead_nconv = nelder_mead_data[:, -3]  # N_conv is the third last column
    
    good_fun = 0
    difference = np.zeros(len(fcn_list))
    for i in range(len(fcn_list)):
        if np.isinf(bfgs_negloglike[i]) or np.isinf(nelder_mead_negloglike[i]):
            difference[i] = np.inf
        elif np.isnan(bfgs_negloglike[i]) or np.isnan(nelder_mead_negloglike[i]):
            difference[i] = np.inf
        else:
            good_fun += 1
            difference[i] = bfgs_negloglike[i] - nelder_mead_negloglike[i]
        # difference = bfgs_negloglike - nelder_mead_negloglike
    print(f"Number of good functions: {good_fun}")
    # difference = np.abs(difference)


    # Create a comparison table
    comparison_data = {
        'Function': fcn_list,
        'negloglike (BFGS)': bfgs_negloglike,
        'negloglike (Nelder-Mead)': nelder_mead_negloglike,
        'Convergence (BFGS)': bfgs_nconv,
        'Convergence (Nelder-Mead)': nelder_mead_nconv,
        'Difference in negloglike': difference,
        'time (BFGS)': bfgs_data[:, -1],
        'time (Nelder-Mead)': nelder_mead_data[:, -1]
    }
    
    comparison_df = pd.DataFrame(comparison_data)
    sorted_comparison_df = comparison_df.sort_values(by='Difference in negloglike', ascending=False)

    print(comparison_df)
    # print(sorted_comparison_df)
    
    # Save the comparison table to a new file
    comparison_file = os.path.join(likelihood.out_dir, f'negloglike_comparison_comp{comp}.csv')
    comparison_df.to_csv(comparison_file, index=False)
    
    print(f"Comparison table saved to {comparison_file}")
    plt.scatter(nelder_mead_negloglike, bfgs_negloglike)
    x = np.linspace(0, 10**10, 1000)

    plt.plot(x, x, color='red')
    plt.xlim(0, 10**10)
    plt.ylim(0, 10**10)
    plt.xlabel('Nelder-Mead negloglike')
    plt.ylabel('BFGS negloglike')
    plt.show()



# create_comparison_table(comp, likelihood)
