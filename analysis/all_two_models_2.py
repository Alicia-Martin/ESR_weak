#!/usr/bin/env python3
import os
import math
import itertools
import csv # Keeping import for initial file loading convention, though final save is .txt
import numpy as np
from pathlib import Path

# ==========================================
# CONFIGURATION
# ==========================================

# 1. Main Paths
COMBINE_FILE = 'esr/fitting/output/combining_clusters/combine_all_comp_all_clusters_duplicate.dat'
CLUSTERS_LIST = "all_clusters.txt"

# 2. Folder Settings
# *** MODIFIED OUTPUT PATH ***
OUTPUT_DIR = Path("esr/fitting/output/combining_clusters")
OUTPUT_FILE = OUTPUT_DIR / "two_func_pairs_results.txt" 

# Where the equation text files live
LIB_ROOT_NORMAL = "../esr/function_library/rerun_100"
LIB_ROOT_BEST   = "../esr/function_library/best_funcs"

# Where the cluster fitting results live (prefix for the folder name)
FIT_ROOT_DIR = "../esr/fitting/output"
SUFFIX_NORMAL = "_rerun_100_2"
SUFFIX_BEST   = "_600"

TOP_N = 20
ADD_MIXTURE_PENALTY = True

# ==========================================
# HELPER FUNCTIONS (No major changes here, only slight path adjustments)
# ==========================================

def get_paths_for_comp(comp):
    """Returns (LibraryPath, OutputSuffix) based on complexity."""
    if comp in [7, 8]:
        return LIB_ROOT_BEST, SUFFIX_BEST
    return LIB_ROOT_NORMAL, SUFFIX_NORMAL

def load_top_functions(path, n=20):
    """Reads the top N functions from the combine table."""
    print(f"Loading top {n} functions...")
    raw = np.genfromtxt(path, dtype=str, delimiter='\t', skip_header=1, comments=None)
    
    results = []
    for i in range(min(n, len(raw))):
        func_str = raw[i, 1]
        aifeyn = float(raw[i, 6]) if raw[i, 6] != 'nan' else 0.0
        comp = int(raw[i, -1])
        results.append({'func': func_str, 'ay': aifeyn, 'comp': comp})
    return results

def resolve_function_index(func_data):
    """Finds the line number (index) of the function in its specific library file."""
    lib_root, _ = get_paths_for_comp(func_data['comp'])
    comp_dir = Path(lib_root) / f"compl_{func_data['comp']}"
    
    candidates = sorted(comp_dir.glob("all_equations*.txt"))
    if not candidates:
        print(f"  [Error] No equations file found in {comp_dir}")
        return None
    
    eq_file = candidates[0]
    target = func_data['func'].replace(" ", "").strip()

    with open(eq_file, 'r') as f:
        for idx, line in enumerate(f):
            parts = line.strip().split(';')
            clean_eq = parts[-1].replace(" ", "").strip()
            if clean_eq == target:
                return idx
                
    print(f"  [Error] Function not found in {eq_file}: {func_data['func'][:30]}...")
    return None

def get_cluster_metrics(cluster_name, func_idx, comp):
    """Reads the codelen file for a specific cluster and complexity."""
    _, suffix = get_paths_for_comp(comp)
    
    folder_name = f"output_WL_{cluster_name}{suffix}"
    file_path = Path(FIT_ROOT_DIR) / folder_name / f"codelen_matches_comp{comp}.dat"

    if not file_path.exists():
        print(file_path, "doesn't exist")
        return None

    try:
        data = np.genfromtxt(file_path)
        if data.ndim == 1: data = data.reshape(1, -1)
        
        if func_idx < len(data):
            matches = data[func_idx]
            if matches.ndim == 1:
                 matches = matches.reshape(1, -1)

        else:
            matches = np.array([])

        if len(matches) == 0:
            #print('here')
            return None
            
        best = matches[np.argmin(matches[:, 0] + matches[:, 1])]
        return (best[0], best[1])
        
    except Exception as e:
        print(e)
        return None

# ==========================================
# MAIN EXECUTION
# ==========================================

# 1. Setup
# Store all top funcs initially to keep track of their original index
all_top_funcs = load_top_functions(COMBINE_FILE, TOP_N) 
with open(CLUSTERS_LIST) as f:
    clusters = [line.strip() for line in f if line.strip()]

# 2. Resolve Indices
print("Resolving function indices in library files...")
valid_funcs = []
for i, f in enumerate(all_top_funcs):
    idx = resolve_function_index(f)
    if idx is not None:
        f['idx'] = idx
        f['original_id'] = i  # *** Store the original 0-N index ***
        valid_funcs.append(f)
    else:
        print(f"Skipping function {i} (Unresolved)")

# 3. Scan Pairs
results = []
combinations = list(itertools.combinations(valid_funcs, 2))
print(f"\nScanning {len(combinations)} pairs over {len(clusters)} clusters...\n")

for A, B in combinations:
    print(f"Processing Pair: {A['original_id']} vs {B['original_id']}")
    
    wins_a = 0; wins_b = 0; sum_neg = 0.0; sum_code = 0.0
    
    for clust in clusters:
        #print(A)
        #print(B)
        m_a = get_cluster_metrics(clust, A['idx'], A['comp'])
        m_b = get_cluster_metrics(clust, B['idx'], B['comp'])
        #print(m_b) 
        if m_a is None and m_b is None:
            continue
        #he cambiado esto
        #L_a = (m_a[0] + m_a[1] + A['ay']) if m_a else float('inf')
        #L_b = (m_b[0] + m_b[1] + B['ay']) if m_b else float('inf')

        L_a = (m_a[0] + m_a[1] ) if m_a else float('inf')
        L_b = (m_b[0] + m_b[1] ) if m_b else float('inf')
        
        if L_a < L_b:
            wins_a += 1
            sum_neg += m_a[0]
            sum_code += m_a[1]
        else:
            wins_b += 1
            sum_neg += m_b[0]
            sum_code += m_b[1]

    # 4. Calculate Final Description Length
    N = wins_a + wins_b
    if N == 0: continue
        
    total_per_halo = sum_neg + sum_code
    
    # Library Cost (Aifeyn)
    lib_cost = (A['ay'] if wins_a > 0 else 0) + (B['ay'] if wins_b > 0 else 0)
    
    # Selection/Assignment Cost
    #log_n_fact = math.lgamma(N + 1)
    #log_k_fact_sum = math.lgamma(wins_a + 1) + math.lgamma(wins_b + 1)
    #assign_cost = log_n_fact - log_k_fact_sum
   
    func_counts = {
        'A': wins_a, 
        'B': wins_b
    }
    
    assign_K = 0.0
    #print(N)
    for count in func_counts.values():
        if count > 0:
            # We use np.log (natural logarithm) as standard for entropy/information cost
            assign_K -= count * np.log(count / N)
            
    # The new assignment cost is assign_K
    assign_cost = assign_K
    # Mixture Penalty
    #K = (1 if wins_a > 0 else 0) + (1 if wins_b > 0 else 0)
    #mix_penalty = 0.5 * (K - 1) * math.log(N) if (ADD_MIXTURE_PENALTY and K > 0) else 0

    total_dl = total_per_halo + lib_cost + assign_cost
    
    results.append({
        "ID_A": A['original_id'], 
        "ID_B": B['original_id'],
        "func_A": A["func"],
        "func_B": B["func"],
        "L_per_halo": total_per_halo,
        "L_lib": lib_cost,
        "L_assign": assign_cost, # Combined assignment and mixture penalty
        "Total_DL": total_dl,
        "A_counts": func_counts['A'],
        "B_counts": func_counts['B']
    })

# 5. Save Results to Formatted Text File (Pretty Table)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True) # Ensure the directory exists

# Sort by Total_DL (smaller to bigger)
results.sort(key=lambda x: x['Total_DL'])

with open(OUTPUT_FILE, "w") as f:
    # Write Header
    header = "{:^5} {:^5} |{:^50} {:^50} | {:^15} {:^15} {:^15} | {:^15}| {:^15} | {:^15}".format(
        "ID_A", "ID_B", "func_A", "func_B", "NegL+CodeL", "Library Cost", "Assignment Cost", "Total DL", "A_counts", "B_counts"
    )
    separator = "-" * len(header)
    f.write(header + "\n")
    f.write(separator + "\n")

    # Write Data
    for r in results:
        line = "{:^5} {:^5} |{:^50} {:^50} | {:^15.4f} {:^15.4f} {:^15.4f} | {:^15.4f}| {:^15.4f}| {:^15.4f}".format(
            r["ID_A"], r["ID_B"],
            r["func_A"], r["func_B"],
            r["L_per_halo"],
            r["L_lib"],
            r["L_assign"],
            r["Total_DL"],
            r["A_counts"],
            r["B_counts"]
        )
        f.write(line + "\n")

print(f"\nDone. Results saved to {OUTPUT_FILE}")
