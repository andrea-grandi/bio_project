import pandas as pd

metadata_camelyon_path = "camelyon17_v1.0/metadata.csv"
metadata_seg_path = "cells_segmentation_output/metadata.csv"

df1 = pd.read_csv(metadata_camelyon_path)
df2 = pd.read_csv(metadata_seg_path)

df2.rename(columns={
    "Patient": "patient",
    "Node": "node",
    "X_coord": "x_coord",
    "Y_coord": "y_coord"
    }, inplace=True)

merged_df = pd.merge(df1, df2, on=["patient", "node", "x_coord", "y_coord"], how="left")

# Path to output metadata
output_path = "final_metadata.csv"
merged_df.to_csv(output_path, index=False)

print(f"Merged completed. Output file saved in {output_path}")
