import re
import pickle
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
from esr.generation import simplifier


BASE_DIR = Path("../esr/fitting/output/combining_clusters")
SUFFIX = "best_funcs_change_snapping_merged"


def comp_from_name(name):
    m = re.search(r"final_(\d+)", name)
    if m is None:
        raise ValueError(f"Could not parse comp number from {name}")
    return int(m.group(1))


def load_final_file(path, comp):
    # The .dat file does NOT store `comp` — it is derived from the filename and
    # attached here. This matches combine_comp.py, which appends comp per row
    # (row.append(all_comp[i])) rather than reading it from the file. The four
    # columns between "katz" and "divergence" are the best-fit a0..a3 (combine
    # reads them as d[-6:-2]); they are unused here since params come from the pkl.
    cols = [
        "Rank", "Function", "DL", "Prel", "negloglike", "codelen",
        "ayfeyn", "katz", "a0", "a1", "a2", "a3",
        "divergence"
    ]

    df = pd.read_csv(path, sep=";", header=None)

    if len(df.columns) != len(cols):
        raise ValueError(
            f"Unexpected number of columns in {path.name}: "
            f"got {len(df.columns)}, expected {len(cols)}"
        )

    df.columns = cols
    df["comp"] = comp
    return df


def load_params_file(path):
    with open(path, "rb") as f:
        params = pickle.load(f)

    return {
        "p0": np.asarray(params["a0"]),
        "p1": np.asarray(params["a1"]),
        "p2": np.asarray(params["a2"]),
        "p3": np.asarray(params["a3"]),
        "d0": np.asarray(params["d0"]),
        "d1": np.asarray(params["d1"]),
        "d2": np.asarray(params["d2"]),
        "d3": np.asarray(params["d3"]),
    }


def collect_all_comp_data(base_dir=BASE_DIR, suffix=SUFFIX):
    base_dir = Path(base_dir)

    final_files = sorted(
        base_dir.glob(f"final_*{suffix}.dat"),
        key=lambda p: comp_from_name(p.name)
    )

    if not final_files:
        raise FileNotFoundError(f"No final_*{suffix}.dat files found in {base_dir}")

    all_dfs = []

    p0_all, p1_all, p2_all, p3_all = [], [], [], []
    d0_all, d1_all, d2_all, d3_all = [], [], [], []

    for final_path in final_files:
        comp = comp_from_name(final_path.name)
        params_path = base_dir / f"params_comp{comp}{suffix}.pkl"

        if not params_path.exists():
            print(f"Skipping comp {comp}: missing {params_path.name}")
            continue

        df = load_final_file(final_path, comp)
        params = load_params_file(params_path)

        if len(df) != len(params["p0"]):
            print(
                f"Warning: comp {comp} row mismatch: "
                f"table rows={len(df)}, params rows={len(params['p0'])}"
            )

        all_dfs.append(df)

        p0_all.append(params["p0"])
        p1_all.append(params["p1"])
        p2_all.append(params["p2"])
        p3_all.append(params["p3"])

        d0_all.append(params["d0"])
        d1_all.append(params["d1"])
        d2_all.append(params["d2"])
        d3_all.append(params["d3"])

    if not all_dfs:
        raise RuntimeError("No comp files were successfully loaded.")

    df_all = pd.concat(all_dfs, ignore_index=True)

    p0 = np.concatenate(p0_all)
    p1 = np.concatenate(p1_all)
    p2 = np.concatenate(p2_all)
    p3 = np.concatenate(p3_all)

    d0 = np.concatenate(d0_all)
    d1 = np.concatenate(d1_all)
    d2 = np.concatenate(d2_all)
    d3 = np.concatenate(d3_all)

    return df_all, p0, p1, p2, p3, d0, d1, d2, d3


def find_globalisable_functions(
    df,
    p0, p1, p2, p3,
    d0, d1, d2, d3,
    outdir=BASE_DIR,
    ref_idx=5,
    max_combination_size=4,
    use_katz=False,
):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    fcn_list = df["Function"].astype(str).to_numpy()
    DL = df["DL"].astype(float).to_numpy()
    comp = df["comp"].astype(int).to_numpy()

    params = np.array([p0, p1, p2, p3]).T
    uncertainties = np.array([d0, d1, d2, d3]).T

    if params.ndim != 3 or params.shape[-1] != 4:
        raise ValueError(f"Expected params shape (N, F, 4), got {params.shape}")

    sort_idx = np.argsort(DL)
    fcn_list_sorted = fcn_list[sort_idx]
    DL_sorted = DL[sort_idx]
    comp_sorted = comp[sort_idx]
    # params/uncertainties are (n_clusters, n_functions, 4). The DL sort is over
    # FUNCTIONS, so index axis 1 — not axis 0, which is clusters. combine_comp.py
    # avoids this by sorting p0..p3 (shape (n_functions, n_clusters)) *before* the
    # transpose. Indexing axis 0 here both scrambled the cluster axis and
    # (since n_functions >> n_clusters) raised an IndexError.
    params_sorted = params[:, sort_idx, :]
    unc_sorted = uncertainties[:, sort_idx, :]

    if ref_idx < 0 or ref_idx >= len(DL_sorted):
        raise IndexError(f"ref_idx={ref_idx} out of range for {len(DL_sorted)} rows")

    Delta_DL_local = DL_sorted - DL_sorted[ref_idx]
    candidates = []

    for i, fcn in enumerate(fcn_list_sorted):
        nparams = simplifier.count_params([fcn], 4)[0]

        for r in range(1, min(max_combination_size, nparams) + 1):
            for param_indices in combinations(range(nparams), r):
                gpar = params_sorted[:, i, :][:, param_indices]
                gunc = unc_sorted[:, i, :][:, param_indices]

                with np.errstate(divide="ignore", invalid="ignore"):
                    fisher_diag = 12.0 / (gunc ** 2)
                    fisher_term = 0.5 * np.log(fisher_diag) + np.log(np.abs(gpar))

                fisher_term[(gpar == 0) | (gunc == 0) | ~np.isfinite(fisher_term)] = 0.0

                Delta_MDL = np.sum(fisher_term) + len(param_indices) * (
                    np.log(2) - params_sorted.shape[0] / 2.0 * np.log(3)
                )

                delta_difference = Delta_MDL - Delta_DL_local[i]

                if delta_difference > 0:
                    candidates.append((delta_difference, fcn, list(param_indices), int(comp_sorted[i])))

    # candidates.sort(key=lambda x: x[0], reverse=True)

    suffix = "_katz" if use_katz else ""
    out_file = outdir / f"global_local_funcs_duplicate{suffix}.txt"

    with open(out_file, "w") as f:
        for delta_diff, func, idxs, c in candidates:
            f.write(f"{func}|{idxs}|{c}\n")

    return candidates, out_file


def main():
    df, p0, p1, p2, p3, d0, d1, d2, d3 = collect_all_comp_data(
        base_dir=BASE_DIR,
        suffix=SUFFIX,
    )

    print("Loaded rows:", len(df))
    print("Loaded parameter rows:", len(p0))

    candidates, out_file = find_globalisable_functions(
        df=df,
        p0=p0, p1=p1, p2=p2, p3=p3,
        d0=d0, d1=d1, d2=d2, d3=d3,
        outdir=BASE_DIR,
        ref_idx=5,
        max_combination_size=4,
        use_katz=False,
    )

    print("Saved to:", out_file)
    print("Number of globalisable candidates:", len(candidates))
    print("Top 30:")
    for row in candidates[:30]:
        print(row)

    target = "a0/(a1 - x**3)"
    found = [x for x in candidates if x[1] == target]
    print("\nTarget function:", target)
    print("Occurrences:", found[:10] if found else "not found")


if __name__ == "__main__":
    main()