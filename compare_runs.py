import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

physicalize = False
dirname = 'esr/fitting/output_decreasing/output/'

max_params = 4
if physicalize:
    max_params += 2

with open('all_clusters.txt') as f:
    names = f.read().splitlines() 

#choose a complexity
for comp in range(7,  8):
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

    negloglike_diffs = np.zeros(len(names))
    codelen_diffs = np.zeros(len(names))
    for index, name in enumerate(names):
        # print(name)
        try:
            out_dir = dirname + 'output_WL_' + name 
        except: 
            out_dir = dirname + 'output_WL_' + name +  '_all_fun'

        run_1 = np.genfromtxt(out_dir + "/codelen_matches_comp"+str(comp)+".dat") # All
        run_2 = np.genfromtxt(out_dir + "_2/codelen_matches_comp"+str(comp)+".dat") # All

        DL_value_1 = run_1[:,1]
        DL_value_2 = run_2[:,1]

        negloglike_value_1 = run_1[:,0]
        negloglike_value_2 = run_2[:,0]

        negloglike_diff = negloglike_value_2 - negloglike_value_1
        negloglike_diffs[index] = negloglike_diff


    #if codelen diff 

    #plot for ecah fucntion all the differences
    plt.figure(figsize=(10, 6))
    for i, fcn in enumerate(all_fcn):
        # print(fcn)
        fcn_diffs = negloglike_diffs[np.where(np.array(all_fcn) == fcn)]
        plt.plot(fcn_diffs, label=fcn)


    # Save plots in a single PDF
    with PdfPages('output_plots.pdf') as pdf:
        for i, fcn in enumerate(all_fcn):
            fcn_diffs = negloglike_diffs[np.where(np.array(all_fcn) == fcn)]
            plt.figure(figsize=(10, 6))
            plt.plot(fcn_diffs, label=fcn)
            plt.title(fcn)  # Set the equation name as the title of the plot
            plt.xlabel('Index')
            plt.ylabel('Negloglike Difference')
            plt.legend()
            pdf.savefig()  # Save the current figure to the PDF
            plt.close()  # Close the figure to avoid memory issues

    print("All plots saved to output_plots.pdf")



