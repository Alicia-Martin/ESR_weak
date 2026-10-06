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
        matches = np.array([int(float(x)) for x in match_lines], dtype=int)
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
# Combine HSC best and CLASH results
############################################

def combine_HSC_best_CLASH(hsc_data, clash_data, all_fcn, aifeyn):
    """
    Combine DL from HSC best and CLASH for functions present in both.
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
    
    print(f"\nFUNCTION COUNT SUMMARY:")
    print(f"Total HSC function rows: {len(hsc_data['functions'])}")
    print(f"Unique HSC functions: {len(hsc_map)}")
    print(f"Total CLASH function rows: {len(clash_data['functions'])}")
    print(f"Unique CLASH functions: {len(clash_map)}")

    shared_functions = sorted(set(hsc_map.keys()) & set(clash_map.keys()))
    print(f"\nShared functions between HSC best and CLASH: {len(shared_functions)}")
    
    # Diagnose missing matches
    hsc_not_in_clash = [f for f in hsc_map.keys() if f not in clash_map]
    print(f"\nDIAGNOSTICS:")
    print(f"Unique HSC functions not found in CLASH: {len(hsc_not_in_clash)}")
    if len(hsc_not_in_clash) > 0:
        print(f"\nFirst 10 HSC functions NOT in CLASH:")
        for f in hsc_not_in_clash[:10]:
            print(f"  '{f}'")
        print(f"\nFirst 10 CLASH functions (for comparison):")
        for f in list(clash_map.keys())[:10]:
            print(f"  '{f}'")

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
    
    # Track individual function rows (not just unique indices)
    rows_not_in_clash = 0
    rows_hsc_invalid = 0
    rows_clash_invalid = 0
    rows_combined = 0
    
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
            rows_hsc_invalid += 1
            continue

        # Find best CLASH variant for this function
        best_clash_DL = np.inf
        best_clash_i = None
        
        for clash_i in clash_indices:
            clash_neglog = clash_data["negloglike"][clash_i]
            clash_code = clash_data["codelen"][clash_i]

            if np.isinf(clash_neglog):
                clash_inf_funcs.add(fcn)
            
            # Check for CLASH validity
            if (np.isinf(clash_neglog) or np.isinf(clash_code) or
                np.isnan(clash_neglog) or np.isnan(clash_code)):
                continue
            
            # Calculate CLASH DL for this variant
            clash_DL = clash_neglog + clash_code
            
            if clash_DL < best_clash_DL:
                best_clash_DL = clash_DL
                best_clash_i = clash_i
        
        # If no valid CLASH variant found, skip
        if best_clash_i is None:
            if idx_val not in unique_all_clash_invalid:
                unique_all_clash_invalid[idx_val] = []
            unique_all_clash_invalid[idx_val].append(fcn)
            rows_clash_invalid += 1
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
        rows_combined += 1

    # if clash_inf_funcs:
    #     print(f"\nCLASH functions with inf negloglike ({len(clash_inf_funcs)}):")
    #     for fcn in sorted(clash_inf_funcs):
    #         print(f"  {fcn}")
    
    # Count HSC functions NOT in CLASH at all
    for i, fcn in enumerate(hsc_data["functions"]):
        if fcn not in shared_functions:
            rows_not_in_clash += 1
            idx_val = hsc_data["index"][i]
            if idx_val not in unique_not_in_clash:
                unique_not_in_clash[idx_val] = []
            unique_not_in_clash[idx_val].append(fcn)
    
    # Print summary of missing functions
    total_hsc = len(hsc_data["functions"])
    total_valid = np.sum(valid_mask)
    total_missing = total_hsc - total_valid
    
    print("\n" + "="*70)
    print(f"MISSING FUNCTIONS SUMMARY:")
    print("="*70)
    print(f"Total HSC best functions: {total_hsc}")
    print(f"Successfully combined: {total_valid}")
    print(f"Missing from output: {total_missing}")
    print(f"\nBreakdown by individual function ROWS:")
    print(f"  - Rows combined successfully: {rows_combined}")
    print(f"  - Rows NOT in CLASH dataset: {rows_not_in_clash}")
    print(f"  - Rows where HSC has inf/nan: {rows_hsc_invalid}")
    print(f"  - Rows where all CLASH variants inf/nan: {rows_clash_invalid}")
    print(f"  - Total accounted: {rows_combined + rows_not_in_clash + rows_hsc_invalid + rows_clash_invalid}")
    print(f"\nBreakdown by UNIQUE indices:")
    print(f"  - Unique indices not in CLASH: {len(unique_not_in_clash)}")
    print(f"  - Unique indices with HSC inf/nan: {len(unique_hsc_invalid)}")
    print(f"  - Unique indices with all CLASH inf/nan: {len(unique_all_clash_invalid)}")
    
    if unique_not_in_clash:
        print(f"\nSample unique indices NOT in CLASH ({len(unique_not_in_clash)} total, showing first 10):")
        for idx_val in sorted(unique_not_in_clash.keys())[:10]:
            fcns = unique_not_in_clash[idx_val]
            fcn_sample = fcns[0] if len(fcns) == 1 else f"{fcns[0]} (+ {len(fcns)-1} more variants)"
            print(f"  Index {idx_val}: {fcn_sample}")
    
    # if unique_hsc_invalid:
    #     print(f"\nUnique indices with HSC inf/nan ({len(unique_hsc_invalid)}):")
    #     for idx_val in sorted(unique_hsc_invalid.keys()):
    #         fcns = unique_hsc_invalid[idx_val]
    #         print(f"  Index {idx_val}: {fcns}")
    
    # if unique_all_clash_invalid:
    #     print(f"\nUnique indices where ALL CLASH variants are inf/nan ({len(unique_all_clash_invalid)}):")
    #     for idx_val in sorted(unique_all_clash_invalid.keys()):
    #         fcns = unique_all_clash_invalid[idx_val]
    #         print(f"  Index {idx_val}: {fcns}")
    print("="*70 + "\n")

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
# Print combined results table
############################################

def print_combined_results(functions_list, combined_data, limit=800):
    """
    Print combined results in a table format.
    """
    ptab = PrettyTable()
    ptab.field_names = ["#", "Function", "DL", "-logL", "Codelen", "AIFeyn", "HSC -logL", "HSC Codelen", "CLASH -logL", "CLASH Codelen"]
    
    # Get valid indices sorted by DL (negloglike + codelen)
    valid_indices = np.where(combined_data["valid_mask"])[0]
    
    # Calculate DL for sorting (includes aifeyn)
    DL_values = combined_data["DL"]
    # De-duplicate exact DLs by keeping the alphabetically-first function
    dl_to_best = {}
    for idx in valid_indices:
        dl = DL_values[idx]
        if np.isnan(dl) or np.isinf(dl):
            continue
        fcn = functions_list[idx] if idx < len(functions_list) else f"[Index {idx}]"
        if dl not in dl_to_best:
            dl_to_best[dl] = (idx, fcn)
        else:
            _, best_fcn = dl_to_best[dl]
            if fcn < best_fcn:
                dl_to_best[dl] = (idx, fcn)

    dedup_indices = [v[0] for v in dl_to_best.values()]
    # Sort by DL
    sorted_valid = sorted(dedup_indices, key=lambda i: DL_values[i])
    for rank, idx in enumerate(sorted_valid[:limit]):
        fcn = functions_list[idx] if idx < len(functions_list) else f"[Index {idx}]"
        dl = DL_values[idx]
        neglog = combined_data["negloglike"][idx]
        codelen = combined_data["codelen"][idx]
        aifeyn = dl - neglog - codelen
        hsc_neglog = combined_data["hsc_negloglike"][idx]
        hsc_code = combined_data["hsc_codelen"][idx]
        clash_neglog = combined_data["clash_negloglike"][idx]
        clash_code = combined_data["clash_codelen"][idx]
        
        ptab.add_row([
            rank,
            fcn,
            f'{dl:.2f}',
            f'{neglog:.2f}',
            f'{codelen:.2f}',
            f'{aifeyn:.2e}',
            f'{hsc_neglog:.2f}',
            f'{hsc_code:.2f}',
            f'{clash_neglog:.2f}',
            f'{clash_code:.2f}'
        ])
    

    print(ptab)



############################################
# Write combined results table to file
############################################

def write_combined_results_table(output_path, functions_list, combined_data):
    """
    Write combined results to a text file with table format, sorted by DL.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    ptab = PrettyTable()
    ptab.field_names = ["#", "Function", "DL", "-logL", "Codelen", "AIFeyn", "HSC -logL", "HSC Codelen", "CLASH -logL", "CLASH Codelen"]

    # Get valid indices sorted by DL (negloglike + codelen)
    valid_indices = np.where(combined_data["valid_mask"])[0]

    # Calculate DL for sorting (includes aifeyn)
    DL_values = combined_data["DL"]

    # De-duplicate exact DLs by keeping the alphabetically-first function
    dl_to_best = {}
    for idx in valid_indices:
        dl = DL_values[idx]
        if np.isnan(dl) or np.isinf(dl):
            continue
        fcn = functions_list[idx] if idx < len(functions_list) else f"[Index {idx}]"
        if dl not in dl_to_best:
            dl_to_best[dl] = (idx, fcn)
        else:
            _, best_fcn = dl_to_best[dl]
            if fcn < best_fcn:
                dl_to_best[dl] = (idx, fcn)

    dedup_indices = [v[0] for v in dl_to_best.values()]

    # Sort by DL
    sorted_valid = sorted(dedup_indices, key=lambda i: DL_values[i])

    # Build table for all functions
    for rank, idx in enumerate(sorted_valid):
        fcn = functions_list[idx] if idx < len(functions_list) else f"[Index {idx}]"
        dl = DL_values[idx]
        neglog = combined_data["negloglike"][idx]
        codelen = combined_data["codelen"][idx]
        aifeyn = dl - neglog - codelen
        hsc_neglog = combined_data["hsc_negloglike"][idx]
        hsc_code = combined_data["hsc_codelen"][idx]
        clash_neglog = combined_data["clash_negloglike"][idx]
        clash_code = combined_data["clash_codelen"][idx]

        ptab.add_row([
            rank,
            fcn,
            f"{dl:.2f}",
            f"{neglog:.2f}",
            f"{codelen:.2f}",
            f"{aifeyn:.2e}",
            f"{hsc_neglog:.2f}",
            f"{hsc_code:.2f}",
            f"{clash_neglog:.2f}",
            f"{clash_code:.2f}"
        ])

    with open(output_path, "w") as f:
        f.write(str(ptab))

    print(f"Saved results table: {output_path}")


############################################

def write_combined_output(output_path, functions_list, combined_data):
    """
    Write combined results to a codelen_matches format file.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as f:
        n_rows = len(combined_data["negloglike"])
        rows_written = 0
        rows_with_sentinel = 0
        
        for i in range(n_rows):
            if combined_data["valid_mask"][i]:
                params = combined_data["params"][i]
                delta = combined_data["delta"][i]
                neglog_val = combined_data['negloglike'][i]
                codelen_val = combined_data['codelen'][i]
                nconv_val = combined_data['Nconv'][i]
                niter_val = combined_data['Niter'][i]
                time_val = combined_data['time'][i]
            else:
                # Sentinel values for missing data
                params = np.zeros(4)
                delta = np.zeros(4)
                neglog_val = np.inf
                codelen_val = np.inf
                nconv_val = 0.0
                niter_val = 0.0
                time_val = 0.0
                rows_with_sentinel += 1
            
            rows_written += 1
            
            # Format values in scientific notation with 7 decimal places
            neglog_str = f"{neglog_val:.7e}"
            codelen_str = f"{codelen_val:.7e}"
            func_idx_str = f"{float(combined_data['index'][i]):.7e}"
            
            # Format params (4 values)
            params_str = " ".join([f"{float(p):.7e}" for p in params])
            
            # Format delta (4 values)
            delta_str = " ".join([f"{float(d):.7e}" for d in delta])
            
            # Format extra info
            nconv_str = f"{float(nconv_val):.7e}"
            niter_str = f"{float(niter_val):.7e}"
            time_str = f"{float(time_val):.7e}"

            line = (
                f"{neglog_str} {codelen_str} {func_idx_str} "
                f"{params_str} {delta_str} "
                f"{nconv_str} {niter_str} {time_str}"
            )

            f.write(line + "\n")

    print("Saved:", output_path)
    print(f"Rows written: {rows_written}")
    print(f"Rows with sentinel values (inf): {rows_with_sentinel}")


############################################

def write_output(output_path, best_data):
    """
    Write results to a codelen_matches format file.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as f:
        n_funcs = len(best_data["function"])
        rows_written = 0
        rows_with_sentinel = 0
        
        for i in range(n_funcs):
            if best_data["function"][i] is not None and not np.isnan(best_data["DL"][i]):
                params = best_data["params"][i]
                delta = best_data["delta"][i]
                neglog_val = best_data['negloglike'][i]
                codelen_val = best_data['codelen'][i]
                nconv_val = best_data['Nconv'][i]
                niter_val = best_data['Niter'][i]
                time_val = best_data['time'][i]
            else:
                # Sentinel values for missing data
                params = np.zeros(4)
                delta = np.zeros(4)
                neglog_val = np.inf
                codelen_val = np.inf
                nconv_val = 0.0
                niter_val = 0.0
                time_val = 0.0
                rows_with_sentinel += 1
            
            rows_written += 1
            
            # Format values in scientific notation with 7 decimal places
            neglog_str = f"{neglog_val:.7e}"
            codelen_str = f"{codelen_val:.7e}"
            func_idx_str = f"{float(i):.7e}"
            
            # Format params (4 values)
            params_str = " ".join([f"{float(p):.7e}" for p in params])
            
            # Format delta (4 values)
            delta_str = " ".join([f"{float(d):.7e}" for d in delta])
            
            # Format extra info
            nconv_str = f"{float(nconv_val):.7e}"
            niter_str = f"{float(niter_val):.7e}"
            time_str = f"{float(time_val):.7e}"

            line = (
                f"{neglog_str} {codelen_str} {func_idx_str} "
                f"{params_str} {delta_str} "
                f"{nconv_str} {niter_str} {time_str}"
            )

            f.write(line + "\n")

    print("Saved:", output_path)
    print(f"Rows written: {rows_written}")
    print(f"Rows with sentinel values (inf): {rows_with_sentinel}")


############################################
# Write functions list
############################################

def write_functions_list(output_path, best_data):
    """
    Write the functions list (one per line) to a text file.
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
    ptab.field_names = ["#", "Function", "DL", "-logL", "Codelen", "AIFeyn", "HSC -logL", "HSC Codelen", "CLASH -logL", "CLASH Codelen"]

    # Sort by DL, de-duplicated by exact DL
    valid_indices = [i for i in range(len(best_data["function"])) 
                     if best_data["function"][i] is not None and not np.isnan(best_data["DL"][i])]
    dl_to_best = {}
    for idx in valid_indices:
        dl = best_data["DL"][idx]
        if np.isnan(dl) or np.isinf(dl):
            continue
        fcn = best_data["function"][idx]
        if dl not in dl_to_best:
            dl_to_best[dl] = (idx, fcn)
        else:
            _, best_fcn = dl_to_best[dl]
            if fcn < best_fcn:
                dl_to_best[dl] = (idx, fcn)

    dedup_indices = [v[0] for v in dl_to_best.values()]
    sorted_indices = sorted(dedup_indices, key=lambda i: best_data["DL"][i])

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

            if rank < 50:
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
    print('HERE')
    print(ptab)
    
    with open(output_path.replace('.dat', '_pretty.txt'), 'w') as f:
        f.write(str(ptab))


############################################
# MAIN
############################################

def main():
    comp = 6

    # Paths to the codelen_matches files
    hsc_path = "../results/HSC_CLASH/HSC_best_funcs_all_clusters/codelen_matches_comp" + str(comp) + "_HSC_funcs.dat"
    hsc_matches_path = "../results/HSC_CLASH/HSC_best_funcs_all_clusters_funcs/compl_" + str(comp) + "/matches_" + str(comp) + ".txt"
    clash_path = "../results/HSC_CLASH/CLASH/codelen_matches_comp" + str(comp) + ".dat"
    
    # Function library directories
    fcn_dir = "../results/HSC_CLASH/funcs"
    hsc_fcn_dir = "../results/HSC_CLASH/HSC_best_funcs_all_clusters_funcs"
    
    # Output directory
    output_dir = "../results/HSC_CLASH/results/HSC_funcs/"
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading equations for complexity {comp}...")
    unique_fcn, all_fcn, aifeyn = load_equations(fcn_dir, comp)
    hsc_unique_fcn = load_functions_list(hsc_fcn_dir, comp).copy()
    hsc_unique_file = f"{hsc_fcn_dir}/compl_{comp}/unique_equations_{comp}.txt"
    with open(hsc_unique_file, "r") as f:
        hsc_unique_fcn = f.read().splitlines()
    print(f"Loaded {len(unique_fcn)} unique functions, {len(all_fcn)} total functions")

    print("\nReading HSC best data...")
    hsc_functions = load_functions_list(hsc_fcn_dir, comp)
    hsc_data_raw = read_codelen_matches(hsc_path, hsc_functions)
    if hsc_data_raw is None:
        print("ERROR: Could not read HSC best data")
        return
    hsc_data_matches = read_codelen_matches(hsc_path, hsc_functions, matches_path=hsc_matches_path)
    if hsc_data_matches is None:
        print("ERROR: Could not read HSC matches data")
        return
    print(f"Found {len(hsc_data_raw['negloglike'])} functions in HSC best")

    print("\nReading CLASH data...")
    clash_data = read_codelen_matches(clash_path, all_fcn)
    if clash_data is None:
        print("ERROR: Could not read CLASH data")
        return
    print(f"Found {len(clash_data['negloglike'])} functions in CLASH")

    print("\nCombining HSC best and CLASH results...")
    combined_data = combine_HSC_best_CLASH(hsc_data_raw, clash_data, all_fcn, aifeyn)
    combined_data["index"] = hsc_data_matches["index"]
    n_valid = np.sum(combined_data["valid_mask"])
    print(f"Successfully combined {n_valid} functions present in both datasets")

    print_combined_results(hsc_functions, combined_data, limit=200)

    table_path = f"{output_dir}/combined_results_table_comp{comp}.txt"
    write_combined_results_table(table_path, hsc_functions, combined_data)

    print("\nSelecting best variants by unique function...")
    best_data = find_best_variants(hsc_unique_fcn, hsc_functions, combined_data)

    print("Writing output...")
    output_path = f"{output_dir}/codelen_matches_comp{comp}_combined.dat"
    write_output(output_path, best_data)
    detailed_path = f"{output_dir}/detailed_results_comp{comp}.dat"
    write_detailed_results(detailed_path, best_data)

    functions_path = f"{output_dir}/{comp}_CLASH_funcs_in_HSC.txt"
    write_functions_list(functions_path, best_data)

    print("\nDone!")


if __name__ == "__main__":
    main()
