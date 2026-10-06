import os
import re
from prettytable import PrettyTable


############################################
# Parse PrettyTable text file
############################################

def parse_prettytable_file(path):
    rows = []

    with open(path, "r") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.startswith("|"):
                continue
            parts = [p.strip() for p in line.split("|")[1:-1]]
            if not parts:
                continue
            # Skip header row
            if parts[0] == "#":
                continue
            if len(parts) < 7:
                continue
            rows.append(parts)

    return rows


############################################
# Combine tables
############################################

def combine_tables(table_paths):
    combined = []

    for path in table_paths:
        filename = os.path.basename(path)
        match = re.search(r"comp(\d+)", filename)
        if not match:
            print(f"Skipping (no comp in name): {path}")
            continue
        comp = match.group(1)

        rows = parse_prettytable_file(path)
        for row in rows:
            # Row format: #, Function, DL, -logL, Codelen, HSC(-logL+C), CLASH(-logL+C)
            combined.append([
                row[0],
                row[1],
                comp,
                row[2],
                row[3],
                row[4],
                row[5],
                row[6]
            ])

    return combined


############################################
# Write combined table
############################################

def write_combined_table(output_path, rows):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Sort by DL ascending
    rows.sort(key=lambda r: float(r[3]))

    # Drop duplicate rows with the same DL (keep first occurrence)
    deduped_rows = []
    seen_dl = set()
    for row in rows:
        dl_value = float(row[3])
        if dl_value in seen_dl:
            continue
        seen_dl.add(dl_value)
        deduped_rows.append(row)

    ptab = PrettyTable()
    ptab.field_names = ["#", "Function", "Complexity", "DL", "-logL", "Codelen", "HSC(-logL+C)", "CLASH(-logL+C)"]

    for rank, row in enumerate(deduped_rows):
        ptab.add_row([
            rank,
            row[1],
            row[2],
            row[3],
            row[4],
            row[5],
            row[6],
            row[7]
        ])

    with open(output_path, "w") as f:
        f.write(str(ptab))

    print(f"Saved combined table: {output_path}")


############################################
# MAIN
############################################

if __name__ == "__main__":
    # Update these paths as needed
    table_paths = [
        "../results/HSC_CLASH/results/combined_results_table_comp7.txt",
        "../results/HSC_CLASH/results/combined_results_table_comp8.txt",
        "../results/HSC_CLASH/results/combined_results_table_comp9.txt",
        "../results/HSC_CLASH/results/combined_results_table_comp10.txt",
    ]

    output_path = "../results/HSC_CLASH/results/combined_results_all_complexities.txt"

    combined_rows = combine_tables(table_paths)
    write_combined_table(output_path, combined_rows)
