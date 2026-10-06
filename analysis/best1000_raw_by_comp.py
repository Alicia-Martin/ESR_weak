"""
Top 1000 functions by DL, sourced ONLY from the raw final_N.dat files (the
full exhaustive-search run per complexity) -- unlike combine_final_results.py,
this deliberately does NOT pull in or override with the
final_Nbest_funcs_fixed_cosmo.dat rerun rows. Intended to produce a plain
function list per complexity to feed into a fresh rerun (e.g. with the new
esd_fixed.py).

Rules:
  - Only final_N.dat files are read (comp 1-10, whatever's present in --dir).
  - On a function string appearing more than once (within or across files),
    the lowest-DL row is kept.
  - Rows whose negloglike matches (to 3 decimals) an already-kept, better-DL
    row are dropped as degenerate duplicate fits (same rule as
    combine_final_results.py).

Outputs (written under --outdir, one folder per complexity present in the
top 1000):
  compl_<N>/functions_<N>.txt  -- plain function list, one per line,
                                   ranked best (lowest DL) first
"""
import argparse
import math
import os
from collections import defaultdict

from combine_final_results import discover_files, build_by_fn

TOP_N = 1000


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument("--dir", default=os.path.join(script_dir, "results_fixed_match"),
                         help="Directory with final_*.dat files")
    parser.add_argument("--outdir", default=script_dir,
                         help="Directory under which compl_<N> folders are written")
    args = parser.parse_args()

    final_files, bestfuncs_files = discover_files(args.dir)
    print(f"Found final_N.dat for comp: {sorted(final_files)}")
    print(f"Ignoring {sum(1 for _ in bestfuncs_files)} best_funcs_fixed_cosmo file(s) "
          f"for comp: {sorted(bestfuncs_files)} (not used)")

    final_by_fn, final_counts = build_by_fn(final_files)
    for row in final_by_fn.values():
        row["comp"] = row["file_comp"]

    pool = list(final_by_fn.values())
    pool.sort(key=lambda r: r["DL"])

    kept = []
    seen_nll = {}
    n_dedup_dropped = 0
    for row in pool:
        key = round(row["negloglike"], 3)
        if key in seen_nll:
            n_dedup_dropped += 1
            continue
        seen_nll[key] = row["function"]
        kept.append(row)

    print(f"\nValid rows read: {sum(final_counts.values())}")
    print(f"Dropped as negloglike duplicates (tol 1e-3): {n_dedup_dropped}")
    print(f"Deduped pool: {len(kept)} rows")

    top = kept[:TOP_N]
    print(f"Taking top {len(top)} by DL (raw final_N.dat only, no bestfuncs override)")

    by_comp = defaultdict(list)
    for row in top:
        by_comp[row["comp"]].append(row)

    print("\nComplexity breakdown of top 1000:")
    for comp in sorted(by_comp):
        entries = by_comp[comp]
        comp_dir = os.path.join(args.outdir, f"compl_{comp}")
        os.makedirs(comp_dir, exist_ok=True)

        out_path = os.path.join(comp_dir, f"functions_{comp}.txt")
        with open(out_path, "w") as f:
            for row in entries:
                f.write(row["function"] + "\n")
        print(f"  comp {comp:2d}: {len(entries):4d} functions -> {out_path}")


if __name__ == "__main__":
    main()
