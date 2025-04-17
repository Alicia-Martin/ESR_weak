import numpy as np
import pandas as pd

out_dir = "/Users/aliciamartin/Downloads"

df_1 = pd.read_fwf(out_dir +
    "/combine_all_comp_all_clusters_rerun.dat",
    sep = '\t',
    skiprows=[1],          # Skip the line with "+------+..."
    skipfooter=1,          # Skip the last line with "+------+..."
    header=0               # Use the first line as column headers
)

df_2 = pd.read_fwf(out_dir +
    "/combine_all_comp_all_clusters_rerun_2.dat",
    sep = '\t',
    skiprows=[1],          # Skip the line with "+------+..."
    skipfooter=1,          # Skip the last line with "+------+..."
    header=0               # Use the first line as column headers
)

print("df_1 columns:", df_1.columns.tolist())
print("df_2 columns:", df_2.columns.tolist())

new_columns = [
    "Rank", "Function", "DL", "Prel", "negloglike", "codelen", 
    "ayfeyn", "katz", "Mean_a0", "Std_a0", "Mean_a1", "Std_a1", 
    "divergence", "comp"
]

# Assign new column names to the DataFrame
df_1.columns = new_columns
df_2.columns = new_columns

print("df_1 columns:", df_1.columns.tolist())
print("df_2 columns:", df_2.columns.tolist())

# #compare the DL of all fucntions, taking into account they ar not in the same order
# merged_df = pd.merge(
#     df_1[["Function", "DL"]],       # Keep only Function and DL from df_1
#     df_2[["Function", "DL"]],       # Keep only Function and DL from df_2
#     on="Function",                  # Merge on the Function column
#     how="outer",                    # Keep all functions (even if not in both files)
#     suffixes=("_1", "_2")           # Add suffixes to distinguish DL columns
# )





