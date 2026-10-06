"""Candidate list for the truncation check: old-pool ranks [--start, --end), instead of
best1000_raw_by_comp.py's fixed top-1000.

Same pool-building rules as best1000_raw_by_comp.py (raw final_N.dat only, negloglike-
duplicate dedup, no fixed_cosmo override) -- this is deliberately the SAME old, pre-fix
ranking the current results_sqrt_grid/ shortlist was drawn from (top 1000 of this same
sorted pool), so a --start 1000 slice is exactly "the functions that just missed the cut".

Outputs (written under --outdir):
  compl_<N>/functions_<N>.txt  -- overall_rank;idx;file_comp;function;DL;negloglike;codelen,
    one per line, best (lowest DL) first. idx/file_comp identify the row's position in the
    source final_<file_comp>.dat -- needed to look up per-cluster params in
    results/params_comp<file_comp>.pkl for triage_tail_probe.py.
  top1000_cutoff_DL.txt -- the DL of the 1000th-ranked function in this same pool (i.e.
    the boundary the current results_sqrt_grid/ shortlist was cut at), so triage doesn't
    need to re-parse all of final_*.dat just to know what "made the cut" meant.
"""
import argparse
import os
from collections import defaultdict

from combine_final_results import discover_files, build_by_fn


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument("--dir", default=os.path.join(script_dir, "results_fixed_match"),
                         help="Directory with final_*.dat files")
    parser.add_argument("--outdir", default=os.path.join(script_dir, "tail_probe_1000_3000"),
                         help="Directory under which compl_<N> folders are written")
    parser.add_argument("--start", type=int, default=1000,
                         help="0-based start rank (inclusive) in the deduped, DL-sorted pool")
    parser.add_argument("--end", type=int, default=3000,
                         help="0-based end rank (exclusive)")
    args = parser.parse_args()

    final_files, bestfuncs_files = discover_files(args.dir)
    print(f"Found final_N.dat for comp: {sorted(final_files)}")
    print(f"Ignoring {len(bestfuncs_files)} best_funcs_fixed_cosmo file(s) "
          f"for comp: {sorted(bestfuncs_files)} (not used -- matches best1000_raw_by_comp.py)")

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

    if args.start >= len(kept):
        print(f"\n--start {args.start} is beyond the deduped pool size ({len(kept)}); nothing to write.")
        return

    window = kept[args.start:args.end]
    print(f"Taking ranks [{args.start}, {min(args.end, len(kept))}) by DL: {len(window)} functions")

    by_comp = defaultdict(list)
    for overall_rank, row in enumerate(window, start=args.start):
        by_comp[row["comp"]].append((overall_rank, row))

    print(f"\nComplexity breakdown of ranks [{args.start}, {args.end}):")
    for comp in sorted(by_comp):
        entries = by_comp[comp]
        comp_dir = os.path.join(args.outdir, f"compl_{comp}")
        os.makedirs(comp_dir, exist_ok=True)

        out_path = os.path.join(comp_dir, f"functions_{comp}.txt")
        with open(out_path, "w") as f:
            for overall_rank, row in entries:
                f.write(f"{overall_rank};{row['idx']};{row['file_comp']};{row['function']};"
                        f"{row['DL']};{row['negloglike']};{row['codelen']}\n")
        print(f"  comp {comp:2d}: {len(entries):4d} functions -> {out_path}")

    os.makedirs(args.outdir, exist_ok=True)
    cutoff_path = os.path.join(args.outdir, "top1000_cutoff_DL.txt")
    if len(kept) >= 1000:
        with open(cutoff_path, "w") as f:
            f.write(f"{kept[999]['DL']}\n")
        print(f"\nTop-1000 cutoff DL (rank 999 in this pool): {kept[999]['DL']:.4f} -> {cutoff_path}")


if __name__ == "__main__":
    main()
