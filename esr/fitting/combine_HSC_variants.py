import os
import numpy as np


############################################
# Read one ESR codelen_matches file
############################################

def read_full_run(path, functions):

    data = {}

    if not os.path.exists(path):
        print("Missing:", path)
        return data

    with open(path) as f:
        lines = [ln for ln in f if ln.strip() and not ln.startswith('#')]

    for i, line in enumerate(lines):

        if i >= len(functions):
            break

        parts = line.split()

        data[functions[i]] = {
            "negloglike": float(parts[0]),
            "codelen": float(parts[1]),
            "params": parts[3:7],        # p0 p1 p2 p3
            "delta": parts[7:11],       # d0 d1 d2 d3
            "Nconv": parts[-3],
            "Niter": parts[-2],
            "time": parts[-1],
        }

    return data


############################################
# Combine across clusters
############################################

def combine_all_clusters(cluster_names,
                         comp,
                         full_functions,
                         rerun_functions,
                         base_dir):

    total_negloglike = np.zeros(len(full_functions))
    total_codelen = np.zeros(len(full_functions))
    seen_counts = np.zeros(len(full_functions), dtype=int)

    stored_params = {}
    stored_delta = {}
    stored_extra = {}

    for name in cluster_names:

        print("Processing cluster:", name)

        base = os.path.join(base_dir, f"output_WL_{name}")

        original_path = f"{base}_600/codelen_matches_comp{comp}.dat"
        rerun_path    = f"{base}_rerun_100_2/codelen_matches_comp{comp}.dat"

        original = read_full_run(original_path, full_functions)
        
        # Only load rerun if it exists
        if rerun_functions is not None and os.path.exists(rerun_path):
            rerun = read_full_run(rerun_path, rerun_functions)
        else:
            rerun = {}

        for i, fn in enumerate(full_functions):

            # Keep the variant with the lowest negloglike
            original_entry = original.get(fn)
            rerun_entry = rerun.get(fn) if fn in rerun else None

            if original_entry is None and rerun_entry is None:
                best_entry = None
            elif original_entry is None:
                best_entry = rerun_entry
            elif rerun_entry is None:
                best_entry = original_entry
            else:
                # Both exist, pick the one with lower negloglike
                if original_entry["negloglike"] <= rerun_entry["negloglike"]:
                    best_entry = original_entry
                else:
                    best_entry = rerun_entry

            if best_entry is None:
                continue

            total_negloglike[i] += best_entry["negloglike"]
            total_codelen[i] += best_entry["codelen"]
            seen_counts[i] += 1

            # store params only once
            if fn not in stored_params:
                stored_params[fn] = best_entry["params"]
                stored_delta[fn] = best_entry["delta"]
                stored_extra[fn] = (
                    best_entry["Nconv"],
                    best_entry["Niter"],
                    best_entry["time"]
                )

    return total_negloglike, total_codelen, stored_params, stored_delta, stored_extra, seen_counts


############################################
# Write ESR-style output file
############################################

def write_output(output_path,
                 full_functions,
                 total_negloglike,
                 total_codelen,
                 stored_params,
                 stored_delta,
                 stored_extra,
                 seen_counts):

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as f:

        for i, fn in enumerate(full_functions):

            if fn in stored_params:
                params = stored_params[fn]
                delta = stored_delta[fn]
                Nconv, Niter, time = stored_extra[fn]
            else:
                params = [0.0, 0.0, 0.0, 0.0]
                delta = [0.0, 0.0, 0.0, 0.0]
                Nconv, Niter, time = (0.0, 0.0, 0.0)

            if seen_counts[i] == 0:
                neglog_val = np.inf
                codelen_val = np.inf
            else:
                neglog_val = total_negloglike[i]
                codelen_val = total_codelen[i]

            # Format values in scientific notation with 7 decimal places
            neglog_str = f"{neglog_val:.7e}"
            codelen_str = f"{codelen_val:.7e}"
            func_idx_str = f"{float(i):.7e}"
            
            # Format params (4 values)
            params_str = " ".join([f"{float(p):.7e}" for p in params])
            
            # Format delta (4 values)
            delta_str = " ".join([f"{float(d):.7e}" for d in delta])
            
            # Format extra info
            nconv_str = f"{float(Nconv):.7e}"
            niter_str = f"{float(Niter):.7e}"
            time_str = f"{float(time):.7e}"

            line = (
                f"{neglog_str} {codelen_str} {func_idx_str} "
                f"{params_str} {delta_str} "
                f"{nconv_str} {niter_str} {time_str}"
            )

            f.write(line + "\n")

    print("Saved:", output_path)
    print(f"Rows written: {len(full_functions)}")
    print(f"Rows with no data across clusters (set to inf): {int(np.sum(seen_counts == 0))}")


############################################
# MAIN
############################################

if __name__ == "__main__":

    comp = 10

    base_dir = "esr/fitting/output"

    # Load cluster names
    cluster_name_file = "all_clusters.txt"
    with open(cluster_name_file) as f:
        cluster_names = [ln.strip() for ln in f if ln.strip()]

    # Load full equation list
    full_eq_file = f"esr/function_library/best_funcs/compl_{comp}/all_equations_{comp}.txt"
    with open(full_eq_file) as f:
        full_functions = [ln.strip() for ln in f if ln.strip()]

    # Load rerun equation list only for comp 9 and 10
    rerun_functions = None
    if comp in [9, 10]:
        rerun_eq_file = f"esr/function_library/rerun_100_2/compl_{comp}/all_equations_{comp}.txt"
        if os.path.exists(rerun_eq_file):
            with open(rerun_eq_file) as f:
                rerun_functions = [ln.strip() for ln in f if ln.strip()]
            print(f"Loaded {len(rerun_functions)} rerun functions for comp {comp}")
        else:
            print(f"Rerun functions file not found for comp {comp}")
    else:
        print(f"No rerun available for comp {comp} (only available for 9 and 10)")
    
    print(f"Loaded {len(full_functions)} full functions")

    # Combine
    neglog, codelen, params, delta, extra, seen_counts = combine_all_clusters(
        cluster_names,
        comp,
        full_functions,
        rerun_functions,
        base_dir
    )

    # Output path
    output_path = f"{base_dir}/combining_clusters/codelen_matches_comp{comp}_HSC_funcs.dat"

    write_output(
        output_path,
        full_functions,
        neglog,
        codelen,
        params,
        delta,
        extra,
        seen_counts
    )
