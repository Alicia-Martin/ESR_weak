import numpy as np
import pandas as pd
from pathlib import Path
from prettytable import PrettyTable

# Input / output
in_csv = Path("clash_ranking_full.csv")
out_dir = Path("")
out_txt = out_dir / "clash_ranking_full_dedup.txt"
out_dat = out_dir / "clash_ranking_full_dedup.dat"

# Read CSV
df = pd.read_csv(in_csv)

# Rename to the table-style names
df = df.rename(
    columns={
        "function": "Function",
        "aifeyn": "ayfeyn",
    }
)

# Keep only the columns we need from the CSV
needed = ["Rank", "Function", "DL", "negloglike", "codelen", "ayfeyn", "comp"]
df = df[needed].copy()

# Make sure numeric columns are numeric
for col in ["Rank", "DL", "negloglike", "codelen", "ayfeyn", "comp"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Drop rows with bad values
df = df.dropna(subset=["Function", "DL", "negloglike", "codelen", "ayfeyn", "comp"])

# Remove duplicates with the same negloglike (rounded like your earlier code)
df["negloglike_round"] = df["negloglike"].round(2)
df = df.sort_values(["DL", "negloglike_round", "Function"], ascending=[True, True, True])
df = df.drop_duplicates(subset=["negloglike_round"], keep="first").copy()

# Re-rank by DL
df = df.sort_values("DL", ascending=True).reset_index(drop=True)
df["Rank"] = np.arange(1, len(df) + 1)

# Prel from DL, normalized
dl0 = df["DL"].min()
prel = np.exp(-(df["DL"] - dl0))
prel = prel / prel.sum()
df["Prel"] = prel

# Fill columns that are not present in the CSV
df["katz"] = "--"
df["Mean a0"] = "--"
df["Std a0"] = "--"
df["Mean a1"] = "--"
df["Std a1"] = "--"
df["divergence"] = "--"

# Match the table column order
df = df[
    [
        "Rank", "Function", "DL", "Prel", "negloglike", "codelen", "ayfeyn",
        "katz", "Mean a0", "Std a0", "Mean a1", "Std a1", "divergence", "comp"
    ]
].copy()

# Format for display / output
def fmt_num(x, nd=2):
    if isinstance(x, (int, float, np.integer, np.floating)):
        if pd.isna(x):
            return "--"
        return f"{x:.{nd}f}"
    return str(x)

# Write PrettyTable txt
table = PrettyTable()
table.field_names = [
    "Rank", "Function", "DL", "Prel", "negloglike", "codelen", "ayfeyn",
    "katz", "Mean a0", "Std a0", "Mean a1", "Std a1", "divergence", "comp"
]

for _, row in df.iterrows():
    table.add_row([
        int(row["Rank"]),
        str(row["Function"]),
        f"{row['DL']:.2f}",
        f"{row['Prel']:.2f}",
        f"{row['negloglike']:.2f}",
        f"{row['codelen']:.2f}",
        f"{row['ayfeyn']:.2f}",
        row["katz"],
        row["Mean a0"],
        row["Std a0"],
        row["Mean a1"],
        row["Std a1"],
        row["divergence"],
        int(row["comp"]),
    ])

with open(out_txt, "w") as f:
    f.write(table.get_string())

# Write .dat version with numeric headers 0..13, matching your example file style
dat_df = df.copy()
dat_df.columns = [str(i) for i in range(len(dat_df.columns))]
dat_df.to_csv(out_dat, sep="\t", index=False)

# Optional: save a cleaned CSV too
out_clean_csv = out_dir / "clash_ranking_full_dedup.csv"
df.to_csv(out_clean_csv, index=False)

print(f"Saved txt: {out_txt}")
print(f"Saved dat: {out_dat}")
print(f"Saved csv: {out_clean_csv}")
print(f"Rows kept after deduplication: {len(df)}")