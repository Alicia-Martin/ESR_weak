import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import csv

physicalize = False
dirname = 'esr/fitting/output/'

max_params = 4
if physicalize:
    max_params += 2

with open('all_clusters.txt') as f:
    names = f.read().splitlines() 

#choose a complexity

#open the appropriate files 
fcn_dir = 'esr/function_library/rerun_katz/'
# fcn_dir = 'esr/function_library/core_maths/'

unifn_file_7 = fcn_dir + "/compl_7/unique_equations_7.txt"
allfn_file_7 = fcn_dir + "/compl_7/all_equations_7.txt"
aifeyn_file_7 = fcn_dir + "/compl_7/aifeyn_7.txt"

with open(unifn_file_7, "r") as f:         # All
    unique_fcn_7 = f.read().splitlines()

with open(allfn_file_7, "r") as f:         # All
    all_fcn_7 = f.read().splitlines()

unifn_file_10 = fcn_dir + "/compl_10/unique_equations_10.txt"
allfn_file_10 = fcn_dir + "/compl_10/all_equations_10.txt"
aifeyn_file_10 = fcn_dir + "/compl_10/aifeyn_10.txt"

with open(unifn_file_10, "r") as f:         # All
    unique_fcn_10 = f.read().splitlines()

with open(allfn_file_10, "r") as f:         # All
    all_fcn_10 = f.read().splitlines()


negloglike_diffs = np.zeros(len(names))
codelen_diffs = np.zeros(len(names))

for index, name in enumerate(names):
    # print(name)
    out_dir = dirname + 'output_WL_' + name + '_rerun'

    run_1 = np.genfromtxt(out_dir + "/codelen_matches_comp7.dat") # All
    run_2 = np.genfromtxt(out_dir + "/codelen_matches_comp10.dat") # All

    negloglike_value_1 = run_1[:,0]
    negloglike_value_2 = run_2[:,0]

    codelen_value_1 = run_1[:,1]
    codelen_value_2 = run_2[:,1]

    #find the fun that we want to compare
    fcn_7 = 'a1*pow((Abs(a0)/x),x)'
    index_7 = np.where(np.array(all_fcn_7) == fcn_7)[0][0]
    negloglike_value_1 = negloglike_value_1[index_7]
    codelen_value_1 = codelen_value_1[index_7]

    fcn_10 = 'a0*pow((Abs(a1)/x),x)'
    index_10 = np.where(np.array(all_fcn_10) == fcn_10)[0][0]
    negloglike_value_2 = negloglike_value_2[index_10]
    codelen_value_2 = codelen_value_2[index_10]


    negloglike_diff = negloglike_value_2 - negloglike_value_1
    negloglike_diffs[index] = negloglike_diff

    codelen_diff = codelen_value_2 - codelen_value_1
    codelen_diffs[index] = codelen_diff


#plot the diffs
    
plt.figure(figsize=(10, 6))
plt.plot(negloglike_diffs, marker='o')
plt.title("Negloglike Difference")
plt.xlabel('Index')
plt.ylabel('Negloglike Difference')
plt.savefig('negloglike_diffs.png')
plt.close()

plt.figure(figsize=(10, 6))
plt.plot(codelen_diffs, marker='o')
plt.title("Codelen Difference")
plt.xlabel('Index')
plt.ylabel('Codelen Difference')
plt.savefig('codelen_diffs.png')
plt.close()


# #plot for ecah fucntion all the differences
# plt.figure(figsize=(10, 6))
# for i, fcn in enumerate(all_fcn):
#     # print(fcn)
#     fcn_diffs = negloglike_diffs[np.where(np.array(all_fcn) == fcn)]
#     plt.plot(fcn_diffs, label=fcn)


# # Save plots in a single PDF
# with PdfPages('output_plots.pdf') as pdf:
#     for i, fcn in enumerate(all_fcn):
#         #if fcn in unqiue functions
#         if fcn in unique_fcn:
#             fcn_diffs = negloglike_diffs[np.where(np.array(all_fcn) == fcn)]
#             plt.figure(figsize=(10, 6))
#             plt.plot(fcn_diffs, label=fcn)
#             plt.title(fcn)  # Set the equation name as the title of the plot
#             plt.xlabel('Index')
#             plt.ylabel('Negloglike Difference')
#             plt.legend()
#             pdf.savefig()  # Save the current figure to the PDF
#             plt.close()  # Close the figure to avoid memory issues

# print("All plots saved to output_plots.pdf")

# sum_abs_diffs = np.sum(np.abs(negloglike_diffs), axis=0)

# # Plot the sum of absolute differences
# plt.figure(figsize=(10, 6))
# plt.plot(sum_abs_diffs, marker='o')
# plt.title("Sum of Negloglike Diff (All Galaxies)")
# plt.xlabel('Index')
# plt.ylabel('Sum of |Negloglike Difference|')
# plt.grid(True)
# plt.show()
