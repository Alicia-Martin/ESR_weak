from pathlib import Path
from collections import defaultdict, deque
import pickle
import re
import pandas as pd

BASE = Path("")

COMBINED_TABLE = BASE / "combine_all_comp_all_clustersbest_funcs_change_snapping_merged.txt"
OUT_CSV = BASE / "all_params_new.csv"

# This matches the format of your uploaded all_params.csv:
# Rank, Function, ClusterIndex, a0, a1, d0, d1
PARAM_COLS = None

# If instead you want every parameter stored in the pickle, use:
# PARAM_COLS = None

COMPS = [7, 8, 9, 10]

FINAL_DAT = {
    c: BASE / f"final_{c}best_funcs_change_snapping_merged.dat"
    for c in COMPS
}

PARAM_PKL = {
    c: BASE / f"params_comp{c}best_funcs_change_snapping_merged.pkl"
    for c in COMPS
}


def parse_combined_prettytable(path):
    """
    Reads combine_all_comp_...txt and returns:
        Rank, Function, comp

    It expects the PrettyTable format with | separators.
    """
    rows = []

    with open(path, "r") as f:
        for line in f:
            line = line.rstrip("\n")

            if not line.startswith("|"):
                continue

            parts = [p.strip() for p in line.strip("|").split("|")]

            if len(parts) < 2 or parts[0] == "Rank":
                continue

            try:
                rank = int(parts[0])
                func = parts[1]
                comp = int(parts[-1])
            except ValueError:
                continue

            rows.append({
                "Rank": rank,
                "Function": func,
                "comp": comp,
            })

    return rows


def parse_final_dat(path):
    """
    Reads final_<comp>...dat.

    In these files:
        column 0 = local row index
        column 1 = function string
    """
    rows = []

    with open(path, "r") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            parts = line.split(";")

            if len(parts) < 2:
                continue

            local_index = int(parts[0])
            func = parts[1].strip()

            rows.append((local_index, func))

    return rows


def normalise_func(s):
    """
    Only remove whitespace.
    Do not algebraically simplify or rewrite expressions.
    """
    return re.sub(r"\s+", "", s)


def build_lookup_by_comp():
    """
    For each comp, map:

        function string -> local row index in params array

    A deque is used in case the same function appears more than once
    in the same comp file.
    """
    lookup = {}

    for comp, path in FINAL_DAT.items():
        by_func = defaultdict(deque)

        for local_index, func in parse_final_dat(path):
            by_func[normalise_func(func)].append(local_index)

        lookup[comp] = by_func

    return lookup


def make_all_params():
    ranking = parse_combined_prettytable(COMBINED_TABLE)
    lookup = build_lookup_by_comp()

    params = {}

    for comp, path in PARAM_PKL.items():
        with open(path, "rb") as f:
            params[comp] = pickle.load(f)

    if PARAM_COLS is None:
        param_cols = sorted(
            {k for d in params.values() for k in d.keys()},
            key=lambda k: (k[0], int(k[1:]) if k[1:].isdigit() else 999),
        )
    else:
        param_cols = PARAM_COLS

    out_rows = []
    missing = []

    for row in ranking:
        rank = row["Rank"]
        func = row["Function"]
        comp = row["comp"]

        key = normalise_func(func)

        if comp not in lookup or key not in lookup[comp] or not lookup[comp][key]:
            missing.append((rank, comp, func))
            continue

        local_index = lookup[comp][key].popleft()

        # Number of clusters is the second dimension of the arrays
        n_clusters = next(iter(params[comp].values())).shape[1]

        for cluster_index in range(n_clusters):
            out = {
                "Rank": rank,
                "Function": func,
                "ClusterIndex": cluster_index,
            }

            for col in param_cols:
                if col in params[comp]:
                    out[col] = params[comp][col][local_index, cluster_index]

            out_rows.append(out)

    if missing:
        print("WARNING: some functions were not found in the matching final_<comp>.dat:")
        for rank, comp, func in missing[:20]:
            print(f"  Rank {rank}, comp {comp}: {func}")

        if len(missing) > 20:
            print(f"  ... and {len(missing) - 20} more")

    df = pd.DataFrame(out_rows)

    ordered_cols = ["Rank", "Function", "ClusterIndex"] + [
        c for c in param_cols if c in df.columns
    ]

    df = df[ordered_cols]
    df.to_csv(OUT_CSV, index=False)

    print(f"Saved: {OUT_CSV}")
    print(f"Rows: {len(df)}")
    print(f"Ranks: {df['Rank'].min()}..{df['Rank'].max()}")
    print(f"Columns: {list(df.columns)}")

    return df


if __name__ == "__main__":
    make_all_params()