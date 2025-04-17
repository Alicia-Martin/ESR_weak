import numpy as np
#Create match file, sub_invs, all_equations, aifeyn
comp = 8
#files
invsubs_file = 'esr/function_library/core_maths/compl_' + str(comp) + '/inv_subs_' +str(comp) + '.txt'
match_file = 'esr/function_library/core_maths/compl_' + str(comp) + '/matches_' + str(comp) + '.txt'
fcn_list = 'esr/function_library/core_maths/compl_' + str(comp) + '/all_equations_' + str(comp) + '.txt'
unique_fcn_list = 'esr/function_library/core_maths/compl_' + str(comp) + '/unique_equations_' + str(comp) + '.txt'
aifeyn_file = 'esr/function_library/core_maths/compl_' + str(comp) + '/aifeyn_' + str(comp) + '.txt'


#best 200 funcs
best_funcs_file = 'esr/fitting/output_decreasing/output/12_clusters/best_200_comp' + str(comp) + '.txt'


#Read all files
with open(best_funcs_file, "r") as f:
    best_fcn_list = f.readlines()

with open(invsubs_file, "r") as f:
    inv_subs = f.readlines()

with open(match_file, "r") as f:
    matches = f.readlines()

with open(fcn_list, "r") as f:
    all_fcn_list = f.readlines()

with open(unique_fcn_list, "r") as f:
    unique_fcn_list = f.readlines()

with open(aifeyn_file, "r") as f:
    aifeyn = f.readlines()

# print(all_fcn_list)

#Get the index of the best functions in unique_fcn_list
matches = [float(m.strip()) for m in matches]

best_fcn_indices = []
unique_funcs = []
all_funcs_best_funcs_list = []
inv_subs_best_funcs_list = []
matches_best_funcs_list = []
aifeyn_best_funcs_list = []
for i, fcn in enumerate(best_fcn_list):
    # print(fcn)
    # Find index of the function in the unique list
    all_eqs_index = all_fcn_list.index(fcn)
    best_fcn_index = inv_subs[all_eqs_index]
    # best_fcn_index = unique_fcn_list.index(fcn)
    best_fcn_indices.append(best_fcn_index)
    unique_funcs.append(unique_fcn_list[int(best_fcn_index)])

    # Retrieve the corresponding function from all_fcn_list using the matches array
    corresponding_indices = [i for i, m in enumerate(matches) if m == float(best_fcn_index)]
    # np.where(np.array(matches) == best_fcn_index)
    # print(corresponding_indices)
    for idx in corresponding_indices:
        all_funcs_best_funcs_list.append(all_fcn_list[idx])
        inv_subs_best_funcs_list.append(inv_subs[idx])
        matches_best_funcs_list.append(i)
        aifeyn_best_funcs_list.append(aifeyn[idx])


# print(len(all_funcs_best_funcs_list))
# print(matches_best_funcs_list)

#Save all these files to esr/function_library/best_funcs
best_funcs_file = 'esr/function_library/best_funcs/compl_' + str(comp) + '/unique_equations_' + str(comp) +  '.txt'
inv_subs_file = 'esr/function_library/best_funcs/compl_' + str(comp) + '/inv_subs_' + str(comp) +  '.txt'
matches_file = 'esr/function_library/best_funcs/compl_' + str(comp) + '/matches_' + str(comp) +  '.txt'
aifeyn_file = 'esr/function_library/best_funcs/compl_' + str(comp) + '/aifeyn_' + str(comp) +  '.txt'
all_equations_file = 'esr/function_library/best_funcs/compl_' + str(comp) + '/all_equations_' + str(comp) +  '.txt'

# print('Saving files...')

with open(best_funcs_file, "w") as f:
    for func in unique_funcs:
        f.write(func)

with open(inv_subs_file, "w") as f:
    for inv_sub in inv_subs_best_funcs_list:
        f.write(inv_sub)

with open(matches_file, "w") as f:
    for match in matches_best_funcs_list:
        f.write(str(match) + '\n')

with open(aifeyn_file, "w") as f:
    for aifeyn_eq in aifeyn_best_funcs_list:
        f.write(aifeyn_eq)


with open(all_equations_file, "w") as f:
    for eq in all_funcs_best_funcs_list:
        f.write(eq)



