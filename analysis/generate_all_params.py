"""
Generate a per-cluster parameter CSV (same shape as the old dm-esr/CLASH/all_params_CLASH.csv)
for every function in the merged final_combined pool (see combine_final_results.py).

One row per (function, cluster): Rank, Function, ClusterIndex, a0, a1, a2, a3, d0, d1, d2, d3.
a0-a3 are the fitted parameter values for that cluster, d0-d3 their uncertainties.

Requires a Python with a working numpy -- the params_comp*.pkl files are pickled numpy
arrays, and this machine's plain `python3` has a broken numpy/pandas binary. Use e.g.:
    python generate_all_params.py
"""
import argparse
import csv
import os
import pickle

from combine_final_results import build_combined_pool


def pkl_path(dirpath, source, file_comp):
    if source == "final":
        return os.path.join(dirpath, f"params_comp{file_comp}.pkl")
    return os.path.join(dirpath, f"params_comp{file_comp}best_funcs_fixed_cosmo.pkl")


def dat_path(dirpath, source, file_comp):
    if source == "final":
        return os.path.join(dirpath, f"final_{file_comp}.dat")
    return os.path.join(dirpath, f"final_{file_comp}best_funcs_fixed_cosmo.dat")


def load_pkl_checked(dirpath, source, file_comp):
    """Load a params pkl and sanity-check its row count against the .dat file's
    line count -- the row-index correspondence (pkl row i <-> .dat line i) is
    load-bearing for this whole script, so fail loudly if it's ever violated."""
    with open(pkl_path(dirpath, source, file_comp), "rb") as f:
        params = pickle.load(f)
    with open(dat_path(dirpath, source, file_comp)) as f:
        n_lines = sum(1 for _ in f)
    n_pkl = params["a0"].shape[0]
    if n_pkl != n_lines:
        raise RuntimeError(
            f"Row-count mismatch for source={source} file_comp={file_comp}: "
            f"pkl has {n_pkl} rows, .dat has {n_lines} lines. The idx-based "
            f"pkl lookup assumes these match 1:1; refusing to guess."
        )
    return params


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    default_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    parser.add_argument("--dir", default=default_dir, help="Directory with final_*.dat / params_comp*.pkl files")
    parser.add_argument("--top", type=int, default=None, help="Only the top N by rank (default: every function)")
    parser.add_argument("--out", default=None, help="Output CSV path (default: <dir>/all_params_combined.csv)")
    args = parser.parse_args()

    print("Rebuilding the merged, deduped function pool (see combine_final_results.py)...")
    kept, stats = build_combined_pool(args.dir)
    if args.top:
        kept = kept[: args.top]
    print(f"Generating per-cluster params for {len(kept)} functions.")

    out_path = args.out or os.path.join(args.dir, "all_params_combined.csv")

    pkl_cache = {}

    def get_pkl(source, file_comp):
        key = (source, file_comp)
        if key not in pkl_cache:
            print(f"  loading pkl for source={source} file_comp={file_comp} ...")
            pkl_cache[key] = load_pkl_checked(args.dir, source, file_comp)
        return pkl_cache[key]

    n_rows_written = 0
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Rank", "Function", "ClusterIndex", "a0", "a1", "a2", "a3", "d0", "d1", "d2", "d3"])
        for rank, row in enumerate(kept, start=1):
            params = get_pkl(row["source"], row["file_comp"])
            idx = row["idx"]
            n_clusters = params["a0"].shape[1]
            for c in range(n_clusters):
                writer.writerow([
                    rank,
                    row["function"],
                    c,
                    params["a0"][idx, c],
                    params["a1"][idx, c],
                    params["a2"][idx, c],
                    params["a3"][idx, c],
                    params["d0"][idx, c],
                    params["d1"][idx, c],
                    params["d2"][idx, c],
                    params["d3"][idx, c],
                ])
                n_rows_written += 1
            if rank % 5000 == 0:
                print(f"  ... {rank}/{len(kept)} functions written")

    print(f"\nWrote {out_path}: {len(kept)} functions x up to 20 clusters = {n_rows_written} rows")


if __name__ == "__main__":
    main()
