import os
import numpy as np
from typing import Optional


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

    for line in lines:
        parts = line.split()
        if len(parts) < 11:
            continue

        try:
            fn_idx = int(float(parts[2]))
        except ValueError:
            continue

        if fn_idx < 0 or fn_idx >= len(functions):
            continue

        neglog = float(parts[0])
        codelen = float(parts[1])
        if np.isinf(neglog) or np.isinf(codelen) or np.isnan(neglog) or np.isnan(codelen):
            continue

        fn = functions[fn_idx]
        data[fn] = {
            "negloglike": neglog,
            "codelen": codelen,
            "params": parts[3:7],        # p0 p1 p2 p3
            "delta": parts[7:11],       # d0 d1 d2 d3
            "Nconv": parts[-3],
            "Niter": parts[-2],
            "time": parts[-1],
        }

    return data


############################################
# Combine across complexities
############################################

def _read_detailed_results(path):
    if not os.path.exists(path):
        print(f"Missing detailed results: {path}")
        return {}

    data = {}
    with open(path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            if line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split(";")]
            if len(parts) < 10:
                continue
            if parts[0] == "#":
                continue
            fn = parts[1]
            data[fn] = {
                "hsc_negloglike": float(parts[6]),
                "hsc_codelen": float(parts[7]),
                "clash_negloglike": float(parts[8]),
                "clash_codelen": float(parts[9]),
            }
    return data


def combine_all_complexities(complexities,
                             all_functions,
                             base_dir):

    best_negloglike = np.full(len(all_functions), np.nan)
    best_codelen = np.full(len(all_functions), np.nan)
    best_aifeyn = np.full(len(all_functions), np.nan)
    best_hsc_negloglike = np.full(len(all_functions), np.nan)
    best_hsc_codelen = np.full(len(all_functions), np.nan)
    best_clash_negloglike = np.full(len(all_functions), np.nan)
    best_clash_codelen = np.full(len(all_functions), np.nan)

    stored_params = {}
    stored_delta = {}
    stored_extra = {}
    stored_complexity: list[Optional[int]] = [None] * len(all_functions)  # Track which complexity each function came from
    stored_source: list[Optional[str]] = [None] * len(all_functions)  # Track which dataset provided the best entry

    sources = [
        {
            "name": "HSC_best",
            "label": "HSC funcs",
            "results_dir": f"{base_dir}/results/HSC_funcs",
            "unique_dir": f"{base_dir}/HSC_best_funcs_all_clusters_funcs",
        },
        {
            "name": "HSC",
            "label": "CLASH funcs",
            "results_dir": f"{base_dir}/results",
            "unique_dir": f"{base_dir}/HSC_funcs",
        },
    ]

    for comp in complexities:

        print(f"Processing complexity {comp}...")

        aifeyn_map = {}
        aifeyn_file = f"{base_dir}/funcs/compl_{comp}/aifeyn_{comp}.txt"
        all_fcn_file = f"{base_dir}/funcs/compl_{comp}/all_equations_{comp}.txt"
        if os.path.exists(aifeyn_file) and os.path.exists(all_fcn_file):
            with open(all_fcn_file) as f:
                all_fcn = [ln.strip() for ln in f if ln.strip()]
            aifeyn_vals = np.atleast_1d(np.genfromtxt(aifeyn_file))
            if len(aifeyn_vals) != len(all_fcn):
                print(
                    f"WARNING: aifeyn length {len(aifeyn_vals)} != all_equations length {len(all_fcn)} for comp {comp}"
                )
            for i, fn in enumerate(all_fcn):
                if i < len(aifeyn_vals):
                    aifeyn_map[fn] = aifeyn_vals[i]
        else:
            print(f"WARNING: missing aifeyn/all_equations for comp {comp}; DL will omit aifeyn")

        for source in sources:
            codelen_path = f"{source['results_dir']}/codelen_matches_comp{comp}_combined.dat"
            detailed_path = f"{source['results_dir']}/detailed_results_comp{comp}.dat"
            comp_unique_file = f"{source['unique_dir']}/compl_{comp}/unique_equations_{comp}.txt"

            if not os.path.exists(codelen_path) or not os.path.exists(comp_unique_file):
                print(f"Skipping {source['name']} comp {comp}: missing inputs")
                continue

            with open(comp_unique_file) as f:
                comp_functions = [ln.strip() for ln in f if ln.strip()]

            comp_data = read_full_run(codelen_path, comp_functions)
            comp_detailed = _read_detailed_results(detailed_path)

            if not comp_data:
                print(f"No data found for {source['name']} comp {comp}")
                continue

            for i, fn in enumerate(all_functions):

                if fn not in comp_data:
                    continue

                entry = comp_data[fn]
                aifeyn_val = aifeyn_map.get(fn, 0.0)
                dl_val = entry["negloglike"] + entry["codelen"] + aifeyn_val

                prev_comp = stored_complexity[i]
                if prev_comp is None or comp < prev_comp:
                    stored_complexity[i] = comp

                if np.isnan(best_negloglike[i]) or dl_val < (best_negloglike[i] + best_codelen[i] + best_aifeyn[i]):
                    best_negloglike[i] = entry["negloglike"]
                    best_codelen[i] = entry["codelen"]
                    best_aifeyn[i] = aifeyn_val
                    detailed_entry = comp_detailed.get(fn)
                    if detailed_entry is not None:
                        best_hsc_negloglike[i] = detailed_entry["hsc_negloglike"]
                        best_hsc_codelen[i] = detailed_entry["hsc_codelen"]
                        best_clash_negloglike[i] = detailed_entry["clash_negloglike"]
                        best_clash_codelen[i] = detailed_entry["clash_codelen"]
                    stored_params[fn] = entry["params"]
                    stored_delta[fn] = entry["delta"]
                    stored_extra[fn] = (
                        entry["Nconv"],
                        entry["Niter"],
                        entry["time"]
                    )
                    stored_source[i] = source["label"]

    return (
        best_negloglike,
        best_codelen,
        best_aifeyn,
        best_hsc_negloglike,
        best_hsc_codelen,
        best_clash_negloglike,
        best_clash_codelen,
        stored_params,
        stored_delta,
        stored_extra,
        stored_complexity,
        stored_source,
    )


############################################
# Deduplicate functions
############################################

def deduplicate_functions(all_functions,
                         total_negloglike,
                         total_codelen,
                         total_aifeyn,
                         total_hsc_negloglike,
                         total_hsc_codelen,
                         total_clash_negloglike,
                         total_clash_codelen,
                         stored_params,
                         stored_delta,
                         stored_extra,
                         stored_complexity,
                         stored_source,
                         dl_tolerance=0.001):
    """
    Remove duplicate functions based on:
    1. Exact string match
    2. DL values within dl_tolerance
    
    For duplicates, keep the one with lowest DL.
    """
    # Calculate DL for all functions
    function_data = []
    for i, fn in enumerate(all_functions):
        if fn not in stored_params:
            continue
        dl = total_negloglike[i] + total_codelen[i] + total_aifeyn[i]
        function_data.append({
            'index': i,
            'function': fn,
            'DL': dl,
            'negloglike': total_negloglike[i],
            'codelen': total_codelen[i]
        })
    
    # Sort by DL
    function_data.sort(key=lambda x: x['DL'])
    
    # Track what to keep
    kept_indices = set()
    kept_strings = set()
    kept_dls = []
    
    duplicates_removed = 0
    
    for item in function_data:
        fn = item['function']
        dl = item['DL']
        idx = item['index']
        
        # Check for exact string duplicate
        if fn in kept_strings:
            duplicates_removed += 1
            continue
        
        # Check for DL duplicate (within tolerance)
        is_dl_duplicate = False
        for kept_dl in kept_dls:
            if abs(dl - kept_dl) <= dl_tolerance:
                is_dl_duplicate = True
                break
        
        if is_dl_duplicate:
            duplicates_removed += 1
            continue
        
        # Keep this function
        kept_indices.add(idx)
        kept_strings.add(fn)
        kept_dls.append(dl)
    
    # Build deduplicated lists
    dedup_functions = [all_functions[i] for i in sorted(kept_indices)]
    dedup_negloglike = np.array([total_negloglike[i] for i in sorted(kept_indices)])
    dedup_codelen = np.array([total_codelen[i] for i in sorted(kept_indices)])
    
    print(f"\nDEDUPLICATION SUMMARY:")
    print(f"Original functions: {len([fn for fn in all_functions if fn in stored_params])}")
    print(f"Duplicates removed: {duplicates_removed}")
    print(f"Unique functions kept: {len(dedup_functions)}")
    
    dedup_aifeyn = np.array([total_aifeyn[i] for i in sorted(kept_indices)])
    dedup_hsc_negloglike = np.array([total_hsc_negloglike[i] for i in sorted(kept_indices)])
    dedup_hsc_codelen = np.array([total_hsc_codelen[i] for i in sorted(kept_indices)])
    dedup_clash_negloglike = np.array([total_clash_negloglike[i] for i in sorted(kept_indices)])
    dedup_clash_codelen = np.array([total_clash_codelen[i] for i in sorted(kept_indices)])
    dedup_source = [stored_source[i] for i in sorted(kept_indices)]

    dedup_complexity = [stored_complexity[i] for i in sorted(kept_indices)]

    return (
        dedup_functions,
        dedup_negloglike,
        dedup_codelen,
        dedup_aifeyn,
        dedup_hsc_negloglike,
        dedup_hsc_codelen,
        dedup_clash_negloglike,
        dedup_clash_codelen,
        dedup_source,
        dedup_complexity,
    )


############################################
# Print combined results table
############################################

def print_combined_results(all_functions,
                          total_negloglike,
                          total_codelen,
                          total_aifeyn,
                          total_hsc_negloglike,
                          total_hsc_codelen,
                          total_clash_negloglike,
                          total_clash_codelen,
                          total_source,
                          stored_complexity,
                          limit=200):
    """
    Print combined results in a table format with complexity column.
    """
    from prettytable import PrettyTable
    
    ptab = PrettyTable()
    # ptab.max_width["Func"] = 35
    # ptab.max_width["Src"] = 12
    ptab.field_names = [
        "#",
        "Func",
        "Src",
        "Comp",
        "DL",
        "-logL",
        "C",
        "AF",
        "C-logL",
        "C-C",
        "H-logL",
        "H-C",
    ]
    
    # Calculate DL and create list of results
    results = []
    for i, fn in enumerate(all_functions):
        # if fn not in stored_complexity:
        #     continue
        
        dl = total_negloglike[i] + total_codelen[i] + total_aifeyn[i]
        comp = stored_complexity[i] if not isinstance(stored_complexity, dict) else stored_complexity.get(fn)
        source = total_source[i] if not isinstance(total_source, dict) else total_source.get(fn)
        results.append(
            (
                i,
                fn,
                source,
                comp,
                dl,
                total_negloglike[i],
                total_codelen[i],
                total_aifeyn[i],
                total_clash_negloglike[i],
                total_clash_codelen[i],
                total_hsc_negloglike[i],
                total_hsc_codelen[i],
            )
        )
    
    # Sort by DL
    results.sort(key=lambda x: x[4])
    
    # Print top N
    for rank, (idx, fn, source, comp, dl, neglog, codelen, aifeyn, c_logl, c_code, h_logl, h_code) in enumerate(results[:limit]):
        ptab.add_row([
            rank,
            fn,
            source,
            comp,
            f'{dl:.2f}',
            f'{neglog:.2f}',
            f'{codelen:.2f}',
            f'{aifeyn:.2e}',
            f'{c_logl:.2f}',
            f'{c_code:.2f}',
            f'{h_logl:.2f}',
            f'{h_code:.2f}',
        ])
    
    print("\n" + "="*130)
    print(f"COMBINED RESULTS (Top {min(limit, len(results))} of {len(results)} functions)")
    print("="*130)
    print(ptab)
    print("="*130 + "\n")


def write_combined_results_table(output_path,
                                 all_functions,
                                 total_negloglike,
                                 total_codelen,
                                 total_aifeyn,
                                 total_hsc_negloglike,
                                 total_hsc_codelen,
                                 total_clash_negloglike,
                                 total_clash_codelen,
                                 total_source,
                                 stored_complexity,
                                 limit=1000):
    """
    Write a pretty table with all functions, sorted by DL.
    """
    from prettytable import PrettyTable

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    ptab = PrettyTable()
    ptab.max_width["Func"] = 50
    ptab.max_width["Src"] = 12
    ptab.field_names = [
        "#",
        "Func",
        "Src",
        "Comp",
        "DL",
        "-logL",
        "C",
        "AF",
        "C-logL",
        "C-C",
        "H-logL",
        "H-C",
    ]

    results = []
    for i, fn in enumerate(all_functions):
        # if fn not in stored_complexity:
        #     continue

        dl = total_negloglike[i] + total_codelen[i] + total_aifeyn[i]
        comp = stored_complexity[i] if not isinstance(stored_complexity, dict) else stored_complexity.get(fn)
        source = total_source[i] if not isinstance(total_source, dict) else total_source.get(fn)
        results.append(
            (
                i,
                fn,
                source,
                comp,
                dl,
                total_negloglike[i],
                total_codelen[i],
                total_aifeyn[i],
                total_clash_negloglike[i],
                total_clash_codelen[i],
                total_hsc_negloglike[i],
                total_hsc_codelen[i],
            )
        )

    results.sort(key=lambda x: x[4])

    for rank, (idx, fn, source, comp, dl, neglog, codelen, aifeyn, c_logl, c_code, h_logl, h_code) in enumerate(results[:limit]):
        ptab.add_row([
            rank,
            fn,
            source,
            comp,
            f'{dl:.2f}',
            f'{neglog:.2f}',
            f'{codelen:.2f}',
            f'{aifeyn:.2e}',
            f'{c_logl:.2f}',
            f'{c_code:.2f}',
            f'{h_logl:.2f}',
            f'{h_code:.2f}',
        ])

    with open(output_path, "w") as f:
        f.write(str(ptab))

    print(f"Saved combined table (top {min(limit, len(results))}): {output_path}")


############################################
# Write ESR-style output file
############################################

def write_output(output_path,
                 all_functions,
                 total_negloglike,
                 total_codelen,
                 stored_params,
                 stored_delta,
                 stored_extra):

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as f:

        for i, fn in enumerate(all_functions):

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

    # Complexities to combine
    complexities = list(range(5, 11))  # comp 7, 8, 9, 10

    base_dir = "../results/HSC_CLASH"

    # Load union of all functions across all complexities
    all_functions_set = set()
    for comp in complexities:
        fcn_files = [
            f"{base_dir}/results/HSC_funcs/{comp}_CLASH_funcs_in_HSC.txt",
            f"{base_dir}/results/{comp}_CLASH_funcs_in_HSC.txt",
        ]
        for fcn_file in fcn_files:
            if os.path.exists(fcn_file):
                with open(fcn_file) as f:
                    funcs = [ln.strip() for ln in f if ln.strip()]
                    all_functions_set.update(funcs)
    
    all_functions = sorted(list(all_functions_set))
    print(f"Found {len(all_functions)} unique functions across all complexities")

    # Combine
    neglog, codelen, aifeyn, hsc_neglog, hsc_codelen, clash_neglog, clash_codelen, params, delta, extra, complexity, source = combine_all_complexities(
        complexities,
        all_functions,
        base_dir
    )

    # Deduplicate functions
    dedup_functions, dedup_neglog, dedup_codelen, dedup_aifeyn, dedup_hsc_neglog, dedup_hsc_codelen, dedup_clash_neglog, dedup_clash_codelen, dedup_source, dedup_complexity = deduplicate_functions(
        all_functions,
        neglog,
        codelen,
        aifeyn,
        hsc_neglog,
        hsc_codelen,
        clash_neglog,
        clash_codelen,
        params,
        delta,
        extra,
        complexity,
        source,
        dl_tolerance=0.001
    )

    print_combined_results(
        dedup_functions,
        dedup_neglog,
        dedup_codelen,
        dedup_aifeyn,
        dedup_hsc_neglog,
        dedup_hsc_codelen,
        dedup_clash_neglog,
        dedup_clash_codelen,
        dedup_source,
        dedup_complexity,
        limit=2000,
    )

    table_path = f"../results/HSC_CLASH/combined_results_all_complexities.txt"
    write_combined_results_table(
        table_path,
        dedup_functions,
        dedup_neglog,
        dedup_codelen,
        dedup_aifeyn,
        dedup_hsc_neglog,
        dedup_hsc_codelen,
        dedup_clash_neglog,
        dedup_clash_codelen,
        dedup_source,
        dedup_complexity,
        limit=2000,
    )

    # Output path
    # output_path = f"../results/HSC_CLASH/results/codelen_matches_combined_CLASH_funcs.dat"
    output_path = f"../results/HSC_CLASH/HSC_funcs/codelen_matches_combined_CLASH_funcs.dat"
    write_output(
        output_path,
        dedup_functions,
        dedup_neglog,
        dedup_codelen,
        params,
        delta,
        extra
    )

    # --- Save top 500 functions overall, then split by complexity ---
    # Compute DL for all deduplicated functions
    dl_list = []
    for i, fn in enumerate(dedup_functions):
        dl = dedup_neglog[i] + dedup_codelen[i] + dedup_aifeyn[i]
        # comp = complexity.get(fn, None)
        comp = dedup_complexity[i] 
        dl_list.append((dl, fn, comp))

    # Sort all by DL and take rows 500:1000
    dl_list.sort(key=lambda x: x[0])
    top_500 = dl_list[0:1000]

    # Split these 500 by complexity
    comp_to_topfuncs = {}
    for _, fn, comp in top_500:
        if comp is not None:
            comp_to_topfuncs.setdefault(comp, []).append(fn)

    # Save each set to file
    for comp, fn_list in comp_to_topfuncs.items():
        out_path = f"../results/HSC_CLASH/HSC_funcs/HSC_CLASH_comp_{comp}.txt"
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f:
            for fn in fn_list:
                f.write(fn + "\n")
        print(f"Saved {len(fn_list)} functions for complexity {comp} (from top 500 overall) to {out_path}")
