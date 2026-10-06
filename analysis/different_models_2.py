import os
import math
import numpy as np
from collections import defaultdict

# -------------------------------
# CONFIG
# -------------------------------
add_mixture_penalty = True  # adds 0.5*(K-1)*log(N) to Option A (recommended)
use_nats = True             # all logs below are natural logs (nats)

# -------------------------------
# ACCUMULATE PER-HALO CHOICES
# -------------------------------
total_neg = 0.0
total_codelen = 0.0

cluster_names_file = 'all_clusters.txt'
with open(cluster_names_file, 'r') as f:
    names = f.read().splitlines()

# Track the chosen function per cluster
chosen_funcs = []                 # list of function labels, one per halo
func_counts = defaultdict(int)    # counts n_m
func_cost = {}                    # aifeyn/library cost per unique function (pay once)
# If two different functions can share the exact same numeric cost, we still dedupe by func label.

for i, name in enumerate(names):
    results_folder = 'esr/fitting/output/output_WL_' + name
    files = [f for f in os.listdir(results_folder) if f.startswith('final_') and f.endswith('.dat')]

    best_func = None
    best_DL = float('inf')
    best_data = None

    for file in files:
        file_path = os.path.join(results_folder, file)
        with open(file_path, 'r') as f:
            best_line = f.readlines()[0]  # assuming first line is the best
        parts = best_line.strip().split(';')

        func = parts[1]
        DL = float(parts[2])
        DL_no_aifeyn = DL - float(parts[6])
        if DL_no_aifeyn < best_DL:
            best_DL = DL
            best_func = func
            best_data = parts

    # Extract pieces for the best function for this cluster
    negloglike_value = float(best_data[4])
    codelen_value    = float(best_data[5])
    aifeyn_value     = float(best_data[6])  # complexity of the function (library cost)

    # Accumulate only the per-halo parts here:
    total_neg     += negloglike_value
    total_codelen += codelen_value

    # Record the chosen function
    chosen_funcs.append(best_func)
    func_counts[best_func] += 1

    # Store the function's one-time cost (if multiple clusters pick the same func, we reuse cost)
    if best_func not in func_cost:
        func_cost[best_func] = aifeyn_value

    print(f"Cluster {name}: Best Function: {best_func}, NegLogLike: {negloglike_value}, "
          f"Codelen: {codelen_value}, Aifeyn: {aifeyn_value}, DL(file): {best_DL}")

# -------------------------------
# LIBRARY COST (pay each used form ONCE)
# -------------------------------
unique_funcs = list(func_cost.keys())
print(unique_funcs)
K = len(unique_funcs)           # number of used functions
N = len(chosen_funcs)           # number of haloes/clusters

library_cost = sum(func_cost[f] for f in unique_funcs)

# -------------------------------
# ASSIGNMENT TERM — OPTION A (enumerative, nats)
# log(N!) - sum_m log(n_m!)
# -------------------------------
assign_A = math.lgamma(N + 1) - sum(math.lgamma(count + 1) for count in func_counts.values())

# Optional MDL mixture penalty for unknown mixing weights
mix_penalty = 0.5 * (K - 1) * np.log(N) if (add_mixture_penalty and K > 0 and N > 0) else 0.0

# -------------------------------
# ASSIGNMENT TERM — OPTION B (entropy, nats)
# N * H(pi), with H using natural logs
# -------------------------------
pi = np.array([count / N for count in func_counts.values()], dtype=float)
# Guard against zeros (shouldn't happen since we only included used funcs)
pi = pi[pi > 0]
assign_B = -N * np.sum(pi * np.log(pi)) if N > 0 and pi.size > 0 else 0.0

# -------------------------------
# TOTALS
# -------------------------------
total_per_halo = total_neg + total_codelen  # DO NOT include library cost per halo

# Option A: enumerative + (optional) mixture penalty
total_DL_optionA = total_per_halo + library_cost + assign_A + mix_penalty

# Option B: entropy
total_DL_optionB = total_per_halo + library_cost + assign_B

print("\n--- SUMMARY (nats) ---")
print(f"N (clusters): {N}, K (used funcs): {K}")
print(f"Per-halo sum (negloglike + codelen): {total_per_halo:.6f}")
print(f"Library cost (once per used func):  {library_cost:.6f}")
print(f"Assignment A (enumerative):         {assign_A:.6f}")
if add_mixture_penalty:
    print(f" + Mixture penalty 0.5*(K-1)logN:  {mix_penalty:.6f}")
print(f"Assignment B (entropy N*H(pi)):      {assign_B:.6f}")
print(f"\nTOTAL DL — Option A: {total_DL_optionA:.6f}")
print(f"TOTAL DL — Option B: {total_DL_optionB:.6f}")

