"""
Same ranked, deduped pool as combine_final_results.py, truncated to the top 100
functions by DL.

Outputs (written next to the inputs, in --dir):
  final_combined_top100.dat  -- tab-separated, machine readable
  final_combined_top100.txt  -- fixed-width table, human readable
"""
import argparse
import math
import os

from combine_final_results import build_combined_pool, write_fixed_width_table, write_tsv

TOP_N = 100


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    default_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    parser.add_argument("--dir", default=default_dir, help="Directory with final_*.dat files")
    args = parser.parse_args()

    kept, _ = build_combined_pool(args.dir)
    kept = kept[:TOP_N]

    # Prel recomputed over the top-100 subset only, same formula as combine_final_results.py.
    dl_min = kept[0]["DL"]
    raw_prel = []
    for row in kept:
        delta = row["DL"] - dl_min
        try:
            p = math.exp(-delta)
        except OverflowError:
            p = 0.0
        if not math.isfinite(p):
            p = 0.0
        raw_prel.append(p)
    prel_sum = sum(raw_prel) or 1.0

    table_rows = []
    for rank, (row, p) in enumerate(zip(kept, raw_prel), start=1):
        table_rows.append([
            rank,
            row["function"],
            row["DL"],
            p / prel_sum,
            row["negloglike"],
            row["codelen"],
            row["ayfeyn"],
            row["katz"],
            row["a0"],
            row["a0_unc"],
            row["a1"],
            row["a1_unc"],
            row["divergence"],
            row["comp"],
            row["source"],
            row["file_comp"],
            row["idx"],
        ])

    columns = [
        "Rank", "Function", "DL", "Prel", "negloglike", "codelen", "ayfeyn", "katz",
        "a0", "a0_unc", "a1", "a1_unc", "divergence", "comp", "source", "file_comp", "idx",
    ]

    out_dat = os.path.join(args.dir, "final_combined_top100.dat")
    write_tsv(columns, table_rows, out_dat)
    print(f"Wrote {out_dat} ({len(table_rows)} rows)")

    out_txt = os.path.join(args.dir, "final_combined_top100.txt")
    write_fixed_width_table(columns, table_rows, out_txt)
    print(f"Wrote {out_txt} ({len(table_rows)} rows)")


if __name__ == "__main__":
    main()
