#compare runs

cluster_names_file = '../data/CLASH/cluster_names.txt'
comp = 10

with open(cluster_names_file, 'r') as f:
    names = [line.strip() for line in f if line.strip() and not line.strip().startswith('#')]


for name in names:
    print(f"\n>>> CLUSTER: {name}")

    best_funcs_file = '../esr/fitting/output/output_WL_' + name + '_best_funcs/combine_DL_comp' + str(comp) + '.txt'
    best_funcs_2_file = '../esr/fitting/output/output_WL_' + name +  '_best_funcs_2/combine_DL_comp' + str(comp) + '.txt'

    best_funcs_fn_file = '../esr/fitting/output/output_WL_' + name + '_best_funcs/combine_DL_fcn_comp' + str(comp) + '.txt'


    #read the fucntions
    best_funcs = []
    with open(best_funcs_fn_file, 'r') as ffn:
        for line in ffn:
            if line.strip() and not line.strip().startswith('#'):
                best_funcs.append(line.strip())


    #read the data from run 1
    run1_data = {}
    with open(best_funcs_file, 'r') as f1:
        for line in f1:
            if line.strip() and not line.strip().startswith('#'):
                parts = line.strip().split()
                #fn is the corresponding line in best_funcs
                fn = best_funcs[len(run1_data)]
                run1_data[fn] = {
                    'negloglike': float(parts[-6]),
                    'codelen': float(parts[-5]),
                    'aifeyn': parts[-4],
                    'params': ' '.join(parts[1:4])
                }


    #read the data from run 2
    run2_data = {}
    with open(best_funcs_2_file, 'r') as f2: 
        for line in f2:
            if line.strip() and not line.strip().startswith('#'):
                parts = line.strip().split()
                fn = ' '.join(parts[1:-3])
                run2_data[fn] = {
                    'negloglike': float(parts[-6]),
                    'codelen': float(parts[-5]),
                    '§aifeyn': parts[-4],
                    'params': ' '.join(parts[1:4])
                }


    #compare the likelihoods of each function
    for fn in run1_data:
        if fn in run2_data:
            negloglike1 = run1_data[fn]['negloglike']
            negloglike2 = run2_data[fn]['negloglike']
            codelen1 = run1_data[fn]['codelen']
            codelen2 = run2_data[fn]['codelen']
            params1 = run1_data[fn]['params']
            params2 = run2_data[fn]['params']

            #if the dff is bigger than 0.1 print the results
            dff = abs(negloglike1 - negloglike2)
            if dff > 0.1:
                print(f"Function: {fn}, cluster: {name}")
                print(f" Run 1 - NegLogLike: {negloglike1}, Codelen: {codelen1}, Params: {params1}")
                print(f" Run 2 - NegLogLike: {negloglike2}, Codelen: {codelen2}, Params: {params2}")
                print(f" Difference in NegLogLike: {dff}\n")









