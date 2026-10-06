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
                         best_functions,
                         base_dir):

    total_negloglike = np.zeros(len(full_functions))
    total_codelen = np.zeros(len(full_functions))

    stored_params = {}
    stored_delta = {}
    stored_extra = {}
    
    # Debug: track problematic functions
    debug_funcs = [
        '(-a1/x - x + 1/a0)/x',
        '(a0 + a1/x - x)/x',
        '(a0 - a1/x - x)/x',
        '(a0 - x + 1/(a1*x))/x',
        '(a0/x + a1 - x)/x',
        '(a0/x - a1 - x)/x',
        '(a0/x - x + 1/a1)/x',
        '(a0/x - x - 1/a1)/x',
        '(a1 - x + 1/(a0*x))/x',
        '(a1/x - x + 1/a0)/x'
    ]
    debug_info = {f: {'bad_clusters': []} for f in debug_funcs}

    for name in cluster_names:

        print("Processing cluster:", name)

        base = os.path.join(base_dir, f"output_WL_{name}")

        original_path = f"{base}/codelen_matches_comp{comp}.dat"
        best1_path    = f"{base}_best_funcs/codelen_matches_comp{comp}.dat"
        best2_path    = f"{base}_best_funcs_2/codelen_matches_comp{comp}.dat"

        original = read_full_run(original_path, full_functions)
        best1    = read_full_run(best1_path, best_functions)
        best2    = read_full_run(best2_path, best_functions)

        for i, fn in enumerate(full_functions):

            # Decide which run to use
            if fn in best1:
                candidates = []

                if fn in best1:
                    candidates.append(best1[fn])
                if fn in best2:
                    candidates.append(best2[fn])

                best_entry = min(candidates, key=lambda x: x["negloglike"])

            else:
                best_entry = original.get(fn)

            if best_entry is None:
                continue
            
            # Debug: track which clusters give INF/NAN
            if fn in debug_funcs:
                nll = best_entry["negloglike"]
                cl = best_entry["codelen"]
                
                if np.isinf(nll) or np.isinf(cl) or np.isnan(nll) or np.isnan(cl):
                    debug_info[fn]['bad_clusters'].append(name)

            total_negloglike[i] += best_entry["negloglike"]
            total_codelen[i] += best_entry["codelen"]

            # store params only once
            if fn not in stored_params:
                stored_params[fn] = best_entry["params"]
                stored_delta[fn] = best_entry["delta"]
                stored_extra[fn] = (
                    best_entry["Nconv"],
                    best_entry["Niter"],
                    best_entry["time"]
                )
    
    # Print debug summary
    print("\n" + "="*60)
    print("CLUSTERS WITH INF/NAN VALUES:")
    print("="*60)
    for fn in debug_funcs:
        bad_clusters = debug_info[fn]['bad_clusters']
        if bad_clusters:
            print(f"\n{fn}: {len(bad_clusters)} bad clusters")
            print(f"  {', '.join(bad_clusters)}")
        else:
            print(f"\n{fn}: No INF/NAN clusters")
    print("="*60 + "\n")

    return total_negloglike, total_codelen, stored_params, stored_delta, stored_extra


############################################
# Write ESR-style output file
############################################

def write_output(output_path,
                 full_functions,
                 total_negloglike,
                 total_codelen,
                 stored_params,
                 stored_delta,
                 stored_extra):

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as f:

        for i, fn in enumerate(full_functions):

            if fn not in stored_params:
                continue

            params = stored_params[fn]
            delta = stored_delta[fn]
            Nconv, Niter, time = stored_extra[fn]

            # Format values in scientific notation with 7 decimal places
            neglog_str = f"{total_negloglike[i]:.7e}"
            codelen_str = f"{total_codelen[i]:.7e}"
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


############################################
# MAIN
############################################

if __name__ == "__main__":

    comp = 9

    base_dir = "../esr/fitting/output"

    # Load cluster names
    with open("esr/data/cluster_names.txt") as f:
        cluster_names = [ln.strip() for ln in f if ln.strip()]

    # Load full equation list
    full_eq_file = f"esr/function_library/core_maths/compl_{comp}/all_equations_{comp}.txt"
    with open(full_eq_file) as f:
        full_functions = [ln.strip() for ln in f if ln.strip()]

    # Load best_funcs equation list
    best_eq_file = f"esr/function_library/best_funcs/compl_{comp}/all_equations_{comp}.txt"
    if os.path.exists(best_eq_file):
        with open(best_eq_file) as f:
            best_functions = [ln.strip() for ln in f if ln.strip()]
    else:
        print(f"WARNING: missing best funcs file: {best_eq_file}; proceeding without best functions")
        best_functions = []

    # Combine
    neglog, codelen, params, delta, extra = combine_all_clusters(
        cluster_names,
        comp,
        full_functions,
        best_functions,
        base_dir
    )

    # Output path (your requested location)
    output_path = (
        "../esr/fitting/output/"
        "combining_clusters/codelen_matches_comp7.dat"
    )

    write_output(
        output_path,
        full_functions,
        neglog,
        codelen,
        params,
        delta,
        extra
    )