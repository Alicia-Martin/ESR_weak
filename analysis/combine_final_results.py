"""
Combine all CLASH ESR final_N.dat complexity files (comp 3-10) with the
final_Nbest_funcs_fixed_cosmo.dat reruns (comp 6-10) into one ranked, deduped table.

Rules:
  - If a function string appears in a best_funcs_fixed_cosmo file, use that row
    instead of the final_N.dat row (the rerun fixes convergence problems in the
    original fit; see quality_reports/plans/zippy-forging-kitten.md for a worked
    example where this changes negloglike by ~8 orders of magnitude).
  - Complexity is taken from the final_N.dat file the function string matches, not
    from the best_funcs filename, since the two don't always agree.
  - Rows whose negloglike matches (to 3 decimals) an already-kept, better-DL row
    are dropped as degenerate duplicate fits.

Outputs (written next to the inputs, in --dir):
  final_combined.dat  -- tab-separated, machine readable
  final_combined.txt  -- fixed-width table, human readable
"""
import argparse
import math
import os
import re

FINAL_RE = re.compile(r"^final_(\d+)\.dat$")
BESTFUNCS_RE = re.compile(r"^final_(\d+)best_funcs_fixed_cosmo\.dat$")

FIELDS = [
    "idx", "function", "DL", "_dup_negloglike", "negloglike", "codelen",
    "ayfeyn", "katz", "a0", "a0_unc", "a1", "a1_unc", "divergence",
]


def parse_dat(path):
    rows = []
    with open(path) as f:
        for line_no, line in enumerate(f):
            parts = line.rstrip("\n").split(";")
            if len(parts) != len(FIELDS):
                continue
            row = dict(zip(FIELDS, parts))
            if row["function"] == "":
                continue
            try:
                dl = float(row["DL"])
            except ValueError:
                continue
            if not math.isfinite(dl):
                continue
            row["idx"] = line_no
            row["DL"] = dl
            row["negloglike"] = float(row["negloglike"])
            row["codelen"] = float(row["codelen"])
            row["ayfeyn"] = float(row["ayfeyn"])
            row["katz"] = float(row["katz"])
            row["a0"] = float(row["a0"])
            row["a0_unc"] = float(row["a0_unc"])
            row["a1"] = float(row["a1"])
            row["a1_unc"] = float(row["a1_unc"])
            rows.append(row)
    return rows


def discover_files(dirpath):
    final_files = {}       # comp -> path
    bestfuncs_files = {}    # comp -> path
    for fname in os.listdir(dirpath):
        m = FINAL_RE.match(fname)
        if m:
            final_files[int(m.group(1))] = os.path.join(dirpath, fname)
            continue
        m = BESTFUNCS_RE.match(fname)
        if m:
            bestfuncs_files[int(m.group(1))] = os.path.join(dirpath, fname)
    return final_files, bestfuncs_files


def build_by_fn(files_by_comp):
    """dict: function string -> row (dict), keeping the lowest-DL row on collision.

    Every row is tagged with:
      file_comp -- the complexity of the file it was physically parsed from
      idx       -- its 0-based line position in that file
    These two are what you need to look up the matching row in
    params_comp{file_comp}.pkl / params_comp{file_comp}best_funcs_fixed_cosmo.pkl;
    they're preserved even when `comp` (the reported/display complexity) gets
    overridden to a matched final file's complexity in build_combined_pool.
    """
    by_fn = {}
    counts = {}
    for comp, path in sorted(files_by_comp.items()):
        rows = parse_dat(path)
        counts[comp] = len(rows)
        for row in rows:
            row["file_comp"] = comp
            fn = row["function"]
            if fn not in by_fn or row["DL"] < by_fn[fn]["DL"]:
                by_fn[fn] = row
    return by_fn, counts


def build_combined_pool(dirpath):
    """Run the full merge (bestfuncs override + negloglike dedup) and return the
    DL-sorted, deduped list of row dicts plus a stats dict, in the same order
    combine_final_results.main() ranks them. Each row keeps `idx`/`file_comp`
    so callers (e.g. generate_all_params.py) can map back to the source pkl.
    """
    final_files, bestfuncs_files = discover_files(dirpath)
    final_by_fn, final_counts = build_by_fn(final_files)
    bestfuncs_by_fn, bestfuncs_counts = build_by_fn(bestfuncs_files)

    for row in final_by_fn.values():
        row["comp"] = row["file_comp"]
        row["source"] = "final"

    # Override: bestfuncs beats final on function-string match.
    n_overridden = 0
    n_comp_inferred = 0
    resolved_bestfuncs = []
    for fn, row in bestfuncs_by_fn.items():
        if fn in final_by_fn:
            row["comp"] = final_by_fn[fn]["comp"]
            row["comp_source"] = "matched_final_file"
            del final_by_fn[fn]
            n_overridden += 1
        else:
            row["comp"] = row["file_comp"]
            row["comp_source"] = "inferred_from_filename"
            n_comp_inferred += 1
        row["source"] = "bestfuncs"
        resolved_bestfuncs.append(row)

    pool = list(final_by_fn.values()) + resolved_bestfuncs
    pool.sort(key=lambda r: r["DL"])

    # Dedup by negloglike, in DL order (keep the better-ranked representative).
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

    stats = {
        "final_counts": final_counts,
        "bestfuncs_counts": bestfuncs_counts,
        "total_final_rows": sum(final_counts.values()),
        "total_bestfuncs_rows": sum(bestfuncs_counts.values()),
        "n_overridden": n_overridden,
        "n_comp_inferred": n_comp_inferred,
        "n_dedup_dropped": n_dedup_dropped,
    }
    return kept, stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    default_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    parser.add_argument("--dir", default=default_dir, help="Directory with final_*.dat files")
    args = parser.parse_args()

    kept, stats = build_combined_pool(args.dir)
    final_counts = stats["final_counts"]
    bestfuncs_counts = stats["bestfuncs_counts"]

    print(f"Found final_N.dat for comp: {sorted(final_counts)}")
    print(f"Found best_funcs_fixed_cosmo for comp: {sorted(bestfuncs_counts)}")
    print(f"\nValid rows: final={stats['total_final_rows']}, bestfuncs={stats['total_bestfuncs_rows']}")

    # Recompute Prel over the surviving pool.
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

    out_dat = os.path.join(args.dir, "final_combined.dat")
    write_tsv(columns, table_rows, out_dat)
    print(f"\nWrote {out_dat} ({len(table_rows)} rows)")

    out_txt = os.path.join(args.dir, "final_combined.txt")
    write_fixed_width_table(columns, table_rows, out_txt)
    print(f"Wrote {out_txt} ({len(table_rows)} rows)")

    # --- Summary ---
    print("\n=== Summary ===")
    for comp in sorted(final_counts):
        bf = bestfuncs_counts.get(comp, 0)
        print(f"  comp {comp:2d}: final={final_counts[comp]:7d} rows"
              + (f", bestfuncs={bf} rows" if bf else ""))
    print(f"\nTotal valid input rows: {stats['total_final_rows'] + stats['total_bestfuncs_rows']}")
    print(f"Overridden by bestfuncs (function matched a final row): {stats['n_overridden']}")
    print(f"Bestfuncs rows with no final-pool match (comp inferred from filename): {stats['n_comp_inferred']}")
    print(f"Dropped as negloglike duplicates (tol 1e-3): {stats['n_dedup_dropped']}")
    print(f"Final combined row count: {len(kept)}")

    comp_i = columns.index("comp")
    source_i = columns.index("source")

    top_n = min(300, len(kept))
    top = table_rows[:top_n]
    n_bestfuncs_top = sum(1 for r in top if r[source_i] == "bestfuncs")
    n_comp5_top = sum(1 for r in top if r[comp_i] <= 5)
    n_neither_top = sum(1 for r in top if r[source_i] != "bestfuncs" and r[comp_i] > 5)
    print(f"\nOf the top {top_n} functions by DL:")
    print(f"  from bestfuncs: {n_bestfuncs_top}")
    print(f"  comp <= 5:      {n_comp5_top}")
    print(f"  neither (flag): {n_neither_top}")
    if n_neither_top:
        print("  Flagged rows (comp > 5, source=final) in top {}:".format(top_n))
        for r in top:
            if r[source_i] != "bestfuncs" and r[comp_i] > 5:
                print(f"    rank {r[0]}: {r[1]}  comp={r[comp_i]}  DL={r[2]:.3f}")


def stringify_row(row):
    return [f"{v:.6g}" if isinstance(v, float) else str(v) for v in row]


def write_tsv(columns, rows, path):
    with open(path, "w") as f:
        f.write("\t".join(columns) + "\n")
        for row in rows:
            f.write("\t".join(stringify_row(row)) + "\n")


def write_fixed_width_table(columns, rows, path):
    str_rows = [stringify_row(row) for row in rows]
    widths = [len(c) for c in columns]
    for str_row in str_rows:
        for i, v in enumerate(str_row):
            if len(v) > widths[i]:
                widths[i] = len(v)

    def fmt_row(values):
        return " | ".join(v.ljust(w) for v, w in zip(values, widths))

    header = fmt_row(columns)
    sep = "-+-".join("-" * w for w in widths)

    with open(path, "w") as f:
        f.write(header + "\n")
        f.write(sep + "\n")
        for str_row in str_rows:
            f.write(fmt_row(str_row) + "\n")


if __name__ == "__main__":
    main()
