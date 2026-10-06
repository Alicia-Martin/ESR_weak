"""
Same ranked, deduped pool as combine_final_results.py, truncated to the top
1000 functions by DL, then split by complexity into per-comp folders.

Outputs (written under --outdir, one folder per complexity present in the
top 1000):
  compl_<N>/functions_<N>.txt  -- fixed-width table, human readable,
                                   functions of complexity N ranked best-first

Each row keeps its OverallRank (position 1-1000 in the full top-1000 list)
alongside a per-complexity Rank, so results stay traceable to the combined
ranking.
"""
import argparse
import math
import os
from collections import defaultdict

from combine_final_results import build_combined_pool, write_fixed_width_table

TOP_N = 1000


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument("--dir", default=os.path.join(script_dir, "results_fixed_match"),
                         help="Directory with final_*.dat files")
    parser.add_argument("--outdir", default=script_dir,
                         help="Directory under which compl_<N> folders are written")
    args = parser.parse_args()

    kept, stats = build_combined_pool(args.dir)
    print(f"Total combined pool: {len(kept)} rows")
    top = kept[:TOP_N]
    print(f"Taking top {len(top)} by DL")

    # Prel recomputed over the top-1000 subset only, same formula as combine_final_results.py.
    dl_min = top[0]["DL"]
    raw_prel = []
    for row in top:
        delta = row["DL"] - dl_min
        try:
            p = math.exp(-delta)
        except OverflowError:
            p = 0.0
        if not math.isfinite(p):
            p = 0.0
        raw_prel.append(p)
    prel_sum = sum(raw_prel) or 1.0

    by_comp = defaultdict(list)
    for overall_rank, (row, p) in enumerate(zip(top, raw_prel), start=1):
        by_comp[row["comp"]].append((overall_rank, row, p / prel_sum))

    columns = [
        "OverallRank", "Rank", "Function", "DL", "Prel", "negloglike", "codelen",
        "ayfeyn", "katz", "a0", "a0_unc", "a1", "a1_unc", "divergence", "source",
    ]

    print("\nComplexity breakdown of top 1000:")
    for comp in sorted(by_comp):
        entries = by_comp[comp]
        comp_dir = os.path.join(args.outdir, f"compl_{comp}")
        os.makedirs(comp_dir, exist_ok=True)

        table_rows = []
        for local_rank, (overall_rank, row, prel) in enumerate(entries, start=1):
            table_rows.append([
                overall_rank, local_rank, row["function"], row["DL"], prel,
                row["negloglike"], row["codelen"], row["ayfeyn"], row["katz"],
                row["a0"], row["a0_unc"], row["a1"], row["a1_unc"],
                row["divergence"], row["source"],
            ])

        out_path = os.path.join(comp_dir, f"functions_{comp}.txt")
        write_fixed_width_table(columns, table_rows, out_path)
        print(f"  comp {comp:2d}: {len(entries):4d} functions -> {out_path}")


if __name__ == "__main__":
    main()
