import os
import numpy as np
import csv
from prettytable import PrettyTable


############################################
# Read one ESR codelen_matches file
############################################

def read_codelen_matches(path, functions, matches_path=None):
    """
    Read a codelen_matches file and return arrays of data.
    """
    if not os.path.exists(path):
        print("Missing:", path)
        return None

    data = np.genfromtxt(path)
    
    # Check if file is empty
    if data.size == 0:
        print(f"Empty file: {path}")
        return None
    
    if data.ndim == 1:
        data = data.reshape(1, -1)
    
    max_params = 4
    negloglike = data[:, 0]
    codelen = data[:, 1]
    index = data[:, 2].astype(int)
    if matches_path is not None:
        with open(matches_path, "r") as f:
            match_lines = [ln.strip() for ln in f if ln.strip()]
        matches = np.array([int(x) for x in match_lines], dtype=int)
        print('len matches:', len(matches))
        if len(matches) != len(index):
            print(f"WARNING: matches length {len(matches)} != data length {len(index)}; using min length")
            min_len = min(len(matches), len(index))
            matches = matches[:min_len]
            index = index[:min_len]
            data = data[:min_len]
            negloglike = data[:, 0]
            codelen = data[:, 1]
            params = data[:, 3:3 + max_params]
            delta = data[:, 3 + max_params:3 + 2*max_params]
            Nconv = data[:, -3]
            Niter = data[:, -2]
            time = data[:, -1]
        index = matches
    params = data[:, 3:3 + max_params]
    delta = data[:, 3 + max_params:3 + 2*max_params]
    Nconv = data[:, -3]
    Niter = data[:, -2]
    time = data[:, -1]

    return {
        "negloglike": negloglike,
        "codelen": codelen,
        "index": index,
        "params": params,
        "delta": delta,
        "Nconv": Nconv,
        "Niter": Niter,
        "time": time,
        "functions": functions
    }


############################################
# Load equations from function library
############################################

def load_equations(fcn_dir, comp):
    """
    Load unique and all equations for a given complexity.
    """
    unifn_file = f"{fcn_dir}/compl_{comp}/unique_equations_{comp}.txt"
    allfn_file = f"{fcn_dir}/compl_{comp}/all_equations_{comp}.txt"
    aifeyn_file = f"{fcn_dir}/compl_{comp}/aifeyn_{comp}.txt"

    with open(unifn_file, "r") as f:
        unique_fcn = f.read().splitlines()

    with open(allfn_file, "r") as f:
        all_fcn = f.read().splitlines()

    aifeyn_value = np.genfromtxt(aifeyn_file)
    aifeyn = np.atleast_1d(aifeyn_value)

    print('len unique_fcn:', len(unique_fcn))
    print('len all_fcn:', len(all_fcn))

    return unique_fcn, all_fcn, aifeyn


def load_functions_list(fcn_dir, comp):
    """
    Load all_equations list for a given complexity.
    """
    allfn_file = f"{fcn_dir}/compl_{comp}/all_equations_{comp}.txt"

    with open(allfn_file, "r") as f:
        all_fcn = f.read().splitlines()

    return all_fcn


############################################
# Combine HSC and CLASH results
############################################

def combine_HSC_CLASH(hsc_data, clash_data, all_fcn, aifeyn):
    """
    Combine DL from HSC and CLASH for functions present in both.
    Returns combined arrays.
    """
    n_funcs = len(hsc_data["functions"])
    
    # Initialize arrays for combined data
    combined_negloglike = np.full(n_funcs, np.nan)
    combined_codelen = np.full(n_funcs, np.nan)
    combined_DL = np.full(n_funcs, np.nan)
    hsc_sum = np.full(n_funcs, np.nan)
    clash_sum = np.full(n_funcs, np.nan)
    hsc_negloglike = np.full(n_funcs, np.nan)
    hsc_codelen = np.full(n_funcs, np.nan)
    clash_negloglike = np.full(n_funcs, np.nan)
    clash_codelen = np.full(n_funcs, np.nan)
    combined_params = np.zeros((n_funcs, 4))
    combined_delta = np.zeros((n_funcs, 4))
    combined_Nconv = np.zeros(n_funcs)
    combined_Niter = np.zeros(n_funcs)
    combined_time = np.zeros(n_funcs)
    combined_index = hsc_data["index"].copy()
    
    # Track which functions are in both datasets
    valid_mask = np.zeros(n_funcs, dtype=bool)

    aifeyn_map = {f: aifeyn[i] for i, f in enumerate(all_fcn)}
    
    # Build list of all CLASH indices for each function string
    clash_map = {}
    for i, f in enumerate(clash_data["functions"]):
        if f not in clash_map:
            clash_map[f] = []
        clash_map[f].append(i)
    
    hsc_map = {f: i for i, f in enumerate(hsc_data["functions"])}

    shared_functions = sorted(set(hsc_map.keys()) & set(clash_map.keys()))
    print(f"Found {len(shared_functions)} shared functions between HSC and CLASH")

    debug_funcs = ['1/(a0 + x)', '1/(a0 + 2*x)', 'a1/(a0 - x)', '(a0 + a1/pow(Abs(a2),x))/x']
    clash_inf_funcs = set()
    
    # Build mapping from unique index to functions for tracking
    hsc_functions = hsc_data["functions"]
    index_to_functions = {}  # unique_index -> list of function strings
    for i, fcn in enumerate(hsc_functions):
        idx_val = hsc_data["index"][i]
        if idx_val not in index_to_functions:
            index_to_functions[idx_val] = []
        index_to_functions[idx_val].append(fcn)
    
    # Track reasons for missing unique indices
    unique_not_in_clash = {}    # index -> list of fcns
    unique_hsc_invalid = {}     # index -> list of fcns
    unique_all_clash_invalid = {} # index -> list of fcns
    
    for fcn in shared_functions:
        hsc_i = hsc_map[fcn]
        clash_indices = clash_map[fcn]  # List of all CLASH indices with this function
        idx = hsc_i
        idx_val = hsc_data["index"][idx]

        hsc_neglog = hsc_data["negloglike"][hsc_i]
        hsc_code = hsc_data["codelen"][hsc_i]

        # Check for HSC validity
        if (np.isinf(hsc_neglog) or np.isinf(hsc_code) or
            np.isnan(hsc_neglog) or np.isnan(hsc_code)):
            if idx_val not in unique_hsc_invalid:
                unique_hsc_invalid[idx_val] = []
            unique_hsc_invalid[idx_val].append(fcn)
            continue
        
        if fcn in debug_funcs:
            print(f"\nDEBUG: Processing {fcn}")
            print(f"  HSC: negloglike={hsc_neglog:.4f}, codelen={hsc_code:.4f}, sum={hsc_neglog + hsc_code:.4f}")

        # Find best CLASH variant for this function
        best_clash_DL = np.inf
        best_clash_i = None
        
        if fcn in debug_funcs:
            print(f"  Found {len(clash_indices)} CLASH variants at indices: {clash_indices[:5]}...")
        
        for clash_i in clash_indices:
            clash_neglog = clash_data["negloglike"][clash_i]
            clash_code = clash_data["codelen"][clash_i]

            if np.isinf(clash_neglog):
                clash_inf_funcs.add(fcn)
            
            if fcn in debug_funcs:
                print(f"    CLASH variant {clash_i}: negloglike={clash_neglog:.4f}, codelen={clash_code:.4f}, sum={clash_neglog + clash_code:.4f}")
            
            # Check for CLASH validity
            if (np.isinf(clash_neglog) or np.isinf(clash_code) or
                np.isnan(clash_neglog) or np.isnan(clash_code)):
                if fcn in debug_funcs:
                    print(f"      -> SKIPPED (inf/nan)")
                continue
            
            # Calculate CLASH DL for this variant
            clash_DL = clash_neglog + clash_code
            
            if clash_DL < best_clash_DL:
                best_clash_DL = clash_DL
                best_clash_i = clash_i
        
        if fcn in debug_funcs and best_clash_i is not None:
            print(f"  Best CLASH variant: {best_clash_i} (negloglike={clash_data['negloglike'][best_clash_i]:.4f}, codelen={clash_data['codelen'][best_clash_i]:.4f}, DL={best_clash_DL:.4f})")
        elif fcn in debug_funcs:
            print(f"  Best CLASH variant: None (no valid variants found)")
        
        # If no valid CLASH variant found, skip
        if best_clash_i is None:
            if idx_val not in unique_all_clash_invalid:
                unique_all_clash_invalid[idx_val] = []
            unique_all_clash_invalid[idx_val].append(fcn)
            continue
        
        # Use best CLASH variant
        clash_neglog = clash_data["negloglike"][best_clash_i]
        clash_code = clash_data["codelen"][best_clash_i]

        # Combine negloglike and codelen
        combined_negloglike[idx] = hsc_neglog + clash_neglog
        combined_codelen[idx] = hsc_code + clash_code
        hsc_sum[idx] = hsc_neglog + hsc_code
        clash_sum[idx] = clash_neglog + clash_code
        hsc_negloglike[idx] = hsc_neglog
        hsc_codelen[idx] = hsc_code
        clash_negloglike[idx] = clash_neglog
        clash_codelen[idx] = clash_code
        
        if fcn in debug_funcs:
            print(f"  Combined: negloglike={combined_negloglike[idx]:.4f}, codelen={combined_codelen[idx]:.4f}, sum={combined_negloglike[idx] + combined_codelen[idx]:.4f}")
        
        # Calculate total DL
        aifeyn_value = aifeyn_map.get(fcn, np.nan)
        combined_DL[idx] = combined_negloglike[idx] + combined_codelen[idx] + aifeyn_value
        
        # Use HSC params and delta as representative
        combined_params[idx] = hsc_data["params"][hsc_i]
        combined_delta[idx] = hsc_data["delta"][hsc_i]
        combined_Nconv[idx] = hsc_data["Nconv"][hsc_i]
        combined_Niter[idx] = hsc_data["Niter"][hsc_i]
        combined_time[idx] = hsc_data["time"][hsc_i]
        valid_mask[idx] = True
        # if fcn =='1/(a0 + x)':
        #     print(valid_mask[idx], hsc_neglog, hsc_code, clash_neglog, clash_code, combined_negloglike[idx], combined_codelen[idx], combined_DL[idx])

    # Print summary of missing functions
    total_hsc = len(hsc_data["functions"])
    total_valid = np.sum(valid_mask)
    total_missing = total_hsc - total_valid
    
    # Check which unique indices have no function in shared_functions
    for idx_val in index_to_functions:
        if idx_val not in unique_not_in_clash and idx_val not in unique_hsc_invalid and idx_val not in unique_all_clash_invalid:
            fcns_for_idx = index_to_functions[idx_val]
            if not any(f in shared_functions for f in fcns_for_idx):
                unique_not_in_clash[idx_val] = fcns_for_idx
    
    print("\n" + "="*70)
    print(f"MISSING FUNCTIONS SUMMARY:")
    print("="*70)
    print(f"Total HSC functions: {total_hsc}")
    print(f"Successfully combined: {total_valid}")
    print(f"Missing from output: {total_missing}")
    print(f"\nBreakdown of missing unique indices:")
    print(f"  - Not in CLASH dataset: {len(unique_not_in_clash)}")
    print(f"  - All CLASH variants inf/nan: {len(unique_all_clash_invalid)}")
    
    # if unique_not_in_clash:
    #     print(f"\nUnique indices NOT in CLASH ({len(unique_not_in_clash)}):")
    #     for idx_val in sorted(unique_not_in_clash.keys()):
    #         fcns = unique_not_in_clash[idx_val]
    #         print(f"  Index {idx_val}: {fcns}")
    
    # if clash_inf_funcs:
    #     print(f"\nCLASH functions with inf negloglike ({len(clash_inf_funcs)}):")
    #     for fcn in sorted(clash_inf_funcs):
    #         print(f"  {fcn}")
    
    # if unique_all_clash_invalid:
    #     print(f"\nUnique indices where ALL CLASH variants are inf/nan ({len(unique_all_clash_invalid)}):")
    #     for idx_val in sorted(unique_all_clash_invalid.keys()):
    #         fcns = unique_all_clash_invalid[idx_val]
    #         print(f"  Index {idx_val}: {fcns}")
    # print("="*70 + "\n")

    return {
        "negloglike": combined_negloglike,
        "codelen": combined_codelen,
        "DL": combined_DL,
        "hsc_sum": hsc_sum,
        "clash_sum": clash_sum,
        "hsc_negloglike": hsc_negloglike,
        "hsc_codelen": hsc_codelen,
        "clash_negloglike": clash_negloglike,
        "clash_codelen": clash_codelen,
        "index": combined_index,
        "params": combined_params,
        "delta": combined_delta,
        "Nconv": combined_Nconv,
        "Niter": combined_Niter,
        "time": combined_time,
        "valid_mask": valid_mask
    }


############################################
# Find best variant for each unique function
############################################

def find_best_variants(unique_fcn, functions_list, combined_data):
    """
    For each unique function, find the variant with the lowest DL.
    Similar to find_lowest_DL in combine_DL_galaxies.py
    """
    n_unique = len(unique_fcn)
    print('N unique functions:', n_unique)
    
    # Get the unique indices that actually have valid data
    valid_indices = np.unique(combined_data["index"][combined_data["valid_mask"]])
    
    # Check which unique functions are missing
    all_unique_indices = set(range(n_unique))
    present_indices = set(valid_indices)
    missing_indices = all_unique_indices - present_indices
    
    print(f'Unique function indices with valid data: {len(valid_indices)}')
    print(f'Missing unique function indices: {len(missing_indices)}')
    
    if missing_indices:
        print(f"\nMissing unique functions:")
        for idx in sorted(missing_indices):
            print(f"  Index {idx}: {unique_fcn[idx]}")
    
    # Initialize arrays for best results
    best_fcn = [None] * n_unique
    best_DL = np.full(n_unique, np.nan)
    best_negloglike = np.full(n_unique, np.nan)
    best_codelen = np.full(n_unique, np.nan)
    best_aifeyn = np.full(n_unique, np.nan)
    best_hsc_sum = np.full(n_unique, np.nan)
    best_clash_sum = np.full(n_unique, np.nan)
    best_hsc_negloglike = np.full(n_unique, np.nan)
    best_hsc_codelen = np.full(n_unique, np.nan)
    best_clash_negloglike = np.full(n_unique, np.nan)
    best_clash_codelen = np.full(n_unique, np.nan)
    best_params = np.zeros((n_unique, 4))
    best_delta = np.zeros((n_unique, 4))
    best_Nconv = np.zeros(n_unique)
    best_Niter = np.zeros(n_unique)
    best_time = np.zeros(n_unique)

    # For each unique index that has data, find the best variant
    for idx_val in valid_indices:
        # Check if this index is within our unique function list
        if idx_val >= n_unique:
            print(f"WARNING: Index {idx_val} exceeds unique function list size ({n_unique})")
            continue
            
        # Find all variants of this unique function (same index)
        mask = (combined_data["index"] == idx_val) & combined_data["valid_mask"]
        
        if not np.any(mask):
            print(f"WARNING: No valid variants found for unique function {unique_fcn[idx_val]} (index {idx_val})")
            continue
        
        # Get DL values for all variants
        DL_variants = combined_data["DL"][mask]
        fcn_variants = [functions_list[j] for j in range(len(mask)) if mask[j]]
        
        if len(DL_variants) == 0 or np.all(np.isnan(DL_variants)):
            continue
        
        # Find minimum DL
        min_DL = np.nanmin(DL_variants)
        min_idx_local = np.nanargmin(DL_variants)
        
        # Prioritize the unique function if DL matches
        for idx, fcn in enumerate(fcn_variants):
            if DL_variants[idx] == min_DL and fcn == unique_fcn[idx_val]:
                min_idx_local = idx
                break
        
        # Map local index to global index
        global_indices = np.where(mask)[0]
        global_idx = global_indices[min_idx_local]
        
        # Store best results at the position corresponding to the index value
        best_fcn[idx_val] = fcn_variants[min_idx_local]
        best_DL[idx_val] = min_DL
        best_negloglike[idx_val] = combined_data["negloglike"][global_idx]
        best_codelen[idx_val] = combined_data["codelen"][global_idx]
        best_aifeyn[idx_val] = min_DL - best_negloglike[idx_val] - best_codelen[idx_val]  # Recover aifeyn
        best_hsc_sum[idx_val] = combined_data["hsc_sum"][global_idx]
        best_clash_sum[idx_val] = combined_data["clash_sum"][global_idx]
        best_hsc_negloglike[idx_val] = combined_data["hsc_negloglike"][global_idx]
        best_hsc_codelen[idx_val] = combined_data["hsc_codelen"][global_idx]
        best_clash_negloglike[idx_val] = combined_data["clash_negloglike"][global_idx]
        best_clash_codelen[idx_val] = combined_data["clash_codelen"][global_idx]
        best_params[idx_val] = combined_data["params"][global_idx]
        best_delta[idx_val] = combined_data["delta"][global_idx]
        best_Nconv[idx_val] = combined_data["Nconv"][global_idx]
        best_Niter[idx_val] = combined_data["Niter"][global_idx]
        best_time[idx_val] = combined_data["time"][global_idx]
    
    return {
        "function": best_fcn,
        "DL": best_DL,
        "negloglike": best_negloglike,
        "codelen": best_codelen,
        "aifeyn": best_aifeyn,
        "hsc_sum": best_hsc_sum,
        "clash_sum": best_clash_sum,
        "hsc_negloglike": best_hsc_negloglike,
        "hsc_codelen": best_hsc_codelen,
        "clash_negloglike": best_clash_negloglike,
        "clash_codelen": best_clash_codelen,
        "params": best_params,
        "delta": best_delta,
        "Nconv": best_Nconv,
        "Niter": best_Niter,
        "time": best_time
    }


############################################
# Write output
############################################

def write_output(output_path, best_data):
    """
    Write results to a codelen_matches format file.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as f:
        n_funcs = len(best_data["function"])
        
        for i in range(n_funcs):
            if best_data["function"][i] is None or np.isnan(best_data["DL"][i]):
                continue
            
            params = best_data["params"][i]
            delta = best_data["delta"][i]
            
            # Format values in scientific notation with 7 decimal places
            neglog_str = f"{best_data['negloglike'][i]:.7e}"
            codelen_str = f"{best_data['codelen'][i]:.7e}"
            func_idx_str = f"{float(i):.7e}"
            
            # Format params (4 values)
            params_str = " ".join([f"{float(p):.7e}" for p in params])
            
            # Format delta (4 values)
            delta_str = " ".join([f"{float(d):.7e}" for d in delta])
            
            # Format extra info
            nconv_str = f"{float(best_data['Nconv'][i]):.7e}"
            niter_str = f"{float(best_data['Niter'][i]):.7e}"
            time_str = f"{float(best_data['time'][i]):.7e}"

            line = (
                f"{neglog_str} {codelen_str} {func_idx_str} "
                f"{params_str} {delta_str} "
                f"{nconv_str} {niter_str} {time_str}"
            )

            f.write(line + "\n")

    print("Saved:", output_path)


############################################
# Write functions list
############################################

def write_functions_list(output_path, best_data):
    """
    Write the functions list (one per line) to a text file.
    Similar to unique_equations files.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    valid_functions = [f for f in best_data["function"] if f is not None]
    
    with open(output_path, 'w') as f:
        for fcn in valid_functions:
            f.write(fcn + "\n")
    
    print(f"Saved {len(valid_functions)} functions to: {output_path}")


############################################
# Write detailed results
############################################

def write_detailed_results(output_path, best_data):
    """
    Write detailed results including function names and DL breakdown.
    """
    ptab = PrettyTable()
    ptab.field_names = ["#", "Function", "DL", "-logL", "Codelen", "AIFeyn", "HSC-logL", "HSC-C", "CLASH-logL", "CLASH-C"]

    # Sort by DL
    valid_indices = [i for i in range(len(best_data["function"])) 
                     if best_data["function"][i] is not None and not np.isnan(best_data["DL"][i])]
    sorted_indices = sorted(valid_indices, key=lambda i: best_data["DL"][i])

    with open(output_path, 'w') as f:
        writer = csv.writer(f, delimiter=';')
        writer.writerow(["#", "Function", "DL", "NegLogLike", "Codelen", "AIFeyn", "HSC_NegLogLike", "HSC_Codelen", "CLASH_NegLogLike", "CLASH_Codelen", "Params"])
        
        for rank, idx in enumerate(sorted_indices):
            data = [
                rank,
                best_data["function"][idx],
                best_data["DL"][idx],
                best_data["negloglike"][idx],
                best_data["codelen"][idx],
                best_data["aifeyn"][idx],
                best_data["hsc_negloglike"][idx],
                best_data["hsc_codelen"][idx],
                best_data["clash_negloglike"][idx],
                best_data["clash_codelen"][idx],
                best_data["params"][idx].tolist()
            ]
            writer.writerow(data)
            
            if rank < 200:
                ptab.add_row([
                    rank,
                    best_data["function"][idx],
                    f'{best_data["DL"][idx]:.2f}',
                    f'{best_data["negloglike"][idx]:.2f}',
                    f'{best_data["codelen"][idx]:.2f}',
                    f'{best_data["aifeyn"][idx]:.2e}',
                    f'{best_data["hsc_negloglike"][idx]:.2f}',
                    f'{best_data["hsc_codelen"][idx]:.2f}',
                    f'{best_data["clash_negloglike"][idx]:.2f}',
                    f'{best_data["clash_codelen"][idx]:.2f}'
                ])

    print(ptab)
    
    with open(output_path.replace('.dat', '_pretty.txt'), 'w') as f:
        f.write(str(ptab))


############################################
# MAIN
############################################

def main():
    comp = 10

    # Paths to the codelen_matches files - UPDATE THESE PATHS
    hsc_path = "../results/HSC_CLASH/HSC/codelen_matches_comp" + str(comp) + ".dat"
    hsc_matches_path = "../results/HSC_CLASH/HSC_funcs/compl_" + str(comp) + "/matches_" + str(comp) + ".txt"
    clash_path = "../results/HSC_CLASH/CLASH/codelen_matches_comp" + str(comp) + ".dat"
    
    # Function library directories
    fcn_dir = "../results/HSC_CLASH/funcs"
    hsc_fcn_dir = "../results/HSC_CLASH/HSC_funcs"
    
    # Output directory
    output_dir = "../results/HSC_CLASH/results"
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading equations for complexity {comp}...")
    unique_fcn, all_fcn, aifeyn = load_equations(fcn_dir, comp)
    hsc_unique_fcn = load_functions_list(hsc_fcn_dir, comp).copy()
    hsc_unique_file = f"{hsc_fcn_dir}/compl_{comp}/unique_equations_{comp}.txt"
    with open(hsc_unique_file, "r") as f:
        hsc_unique_fcn = f.read().splitlines()
    print(f"Loaded {len(unique_fcn)} unique functions, {len(all_fcn)} total functions")

    print("\nReading HSC data...")
    hsc_functions = load_functions_list(hsc_fcn_dir, comp)
    hsc_data = read_codelen_matches(hsc_path, hsc_functions, matches_path=hsc_matches_path)
    if hsc_data is None:
        print("ERROR: Could not read HSC data")
        return
    print(f"Found {len(hsc_data['negloglike'])} functions in HSC")

    print("\nReading CLASH data...")
    clash_data = read_codelen_matches(clash_path, all_fcn)
    if clash_data is None:
        print("ERROR: Could not read CLASH data")
        return
    print(f"Found {len(clash_data['negloglike'])} functions in CLASH")

    print("\nCombining HSC and CLASH results...")
    combined_data = combine_HSC_CLASH(hsc_data, clash_data, all_fcn, aifeyn)
    n_valid = np.sum(combined_data["valid_mask"])
    print(f"Successfully combined {n_valid} functions present in both datasets")

    print("\nFinding best variants...")
    best_data = find_best_variants(hsc_unique_fcn, hsc_functions, combined_data)
    n_best = sum(1 for f in best_data["function"] if f is not None)
    print(f"Found best variants for {n_best} unique functions")

    print("\nWriting output...")
    output_path = f"{output_dir}/codelen_matches_comp{comp}_combined.dat"
    write_output(output_path, best_data)

    detailed_path = f"{output_dir}/detailed_results_comp{comp}.dat"
    write_detailed_results(detailed_path, best_data)

    functions_path = f"{output_dir}/{comp}_CLASH_funcs_in_HSC.txt"
    write_functions_list(functions_path, best_data)

    print("\nDone!")


if __name__ == "__main__":
    main()
